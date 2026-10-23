"""Seed the local load-test database (P3e). Local Docker stack only, never production.

    uv run python loadtest/seed_local.py [--users 200]

1. Copies the replayed articles, story coverage and ingestion runs from
   ``firstread_replay`` (``make persona-replay`` or ``make replay``) into a separate
   ``firstread_loadtest`` database, time-shifted so the newest article is "now".
2. Builds the window and the pool snapshot (with the demo personas) with ingestion's
   own code and publishes it to Redis db 2.
3. Creates ``--users`` readers through the local auth shim: 4 in 5 activate a demo
   persona (20 live articles of history), the rest follow 3 topics and have no
   history yet (cold start).
4. Writes a fresh ES256 key's JWKS and one token per reader to
   ``loadtest/.tokens/`` (gitignored), for real JWT verification with
   ``AUTH_LOCAL_JWKS_FILE``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from redis.asyncio import Redis

from firstread_api import repo
from firstread_api.testing import Signer
from firstread_core.db import apply_migrations, connect, default_migrations_dir
from firstread_core.dbtest import with_database
from firstread_ingest import personas, pools, store, window
from firstread_ingest.config import Settings, load_personas
from firstread_ingest.pipeline import EMBEDDING_MODEL_ID

HERE = Path(__file__).resolve().parent
TOKENS = HERE / ".tokens"
SOURCE_DB, TARGET_DB, REDIS_DB = "firstread_replay", "firstread_loadtest", 2
COLD_TOPICS = (
    ["sports", "health", "entertainment"],
    ["business", "tech_science", "politics"],
    ["world", "travel", "food_drink"],
)
TOKEN_HOURS = 12
SESSION_READERS = 40  # persona readers k6 never touches, for session_cost.py
TABLES = {  # table -> timestamp columns shifted by the same amount
    "articles": ("published_at", "first_seen_at", "created_at"),
    "story_coverage": ("first_seen_at",),
    "ingestion_runs": ("started_at", "finished_at", "window_start", "window_end"),
}


def local_only(dsn: str) -> None:
    host = urlsplit(dsn).hostname
    if host not in ("localhost", "127.0.0.1"):
        raise SystemExit(f"refusing to seed {host!r}: load tests run on the local stack only")


async def copy_shifted(admin_dsn: str) -> float:
    """Recreate the target database and copy the replayed rows, shifted to now."""
    admin = await connect(admin_dsn)
    try:
        await admin.execute(f'drop database if exists "{TARGET_DB}" with (force)')
        await admin.execute(f'create database "{TARGET_DB}"')
    finally:
        await admin.close()
    src = await connect(with_database(admin_dsn, SOURCE_DB))
    dst = await connect(with_database(admin_dsn, TARGET_DB))
    try:
        await apply_migrations(dst, default_migrations_dir())
        newest = await src.fetchval("select max(first_seen_at) from public.articles")
        if newest is None:
            raise SystemExit(f"{SOURCE_DB} has no articles: run `make persona-replay` first")
        shift = datetime.now(UTC) - newest
        for table, cols in TABLES.items():
            names = [
                r["column_name"]
                for r in await src.fetch(
                    "select column_name from information_schema.columns "
                    "where table_schema = 'public' and table_name = $1 order by ordinal_position",
                    table,
                )
            ]
            buf = io.BytesIO()
            await src.copy_from_table(table, schema_name="public", output=buf, columns=names)
            buf.seek(0)
            await dst.copy_to_table(table, schema_name="public", source=buf, columns=names)
            sets = ", ".join(f"{c} = {c} + $1::interval" for c in cols)
            await dst.execute(f"update public.{table} set {sets}", shift)  # noqa: S608
        await dst.execute("vacuum analyze")
        return float(shift.total_seconds())
    finally:
        await src.close()
        await dst.close()


async def publish(dsn: str, redis_url: str) -> dict[str, object]:
    """Window → pool snapshot (+ personas) → Redis, exactly as an ingestion run does."""
    conn = await connect(dsn)
    redis = Redis.from_url(redis_url)
    try:
        await store.register_codecs(conn)
        await redis.flushdb()
        now = datetime.now(UTC)
        win = (await window.sync(conn, None, now, EMBEDDING_MODEL_ID)).snapshot
        snap = await pools.build_snapshot(conn, win, now, f"{now:%Y%m%dT%H%M%SZ}-loadtest")
        chosen = await personas.refresh_or_load(conn, win, now, snap.trending)
        snap.personas = [p.to_dict() for p in chosen.summaries]
        result = await pools.publish(redis, conn, snap)
        return {"window_rows": len(win), "blob_bytes": result.blob_bytes, "pools": result.sizes,
                "personas": {p.id: p.articles for p in chosen.summaries}}  # fmt: skip
    finally:
        await redis.aclose()
        await conn.close()


async def readers(dsn: str, n: int) -> list[dict[str, object]]:
    """``n`` readers via the auth shim: 4 in 5 on a persona, the rest cold."""
    specs = load_personas()
    conn = await connect(dsn)
    out: list[dict[str, object]] = []
    try:
        q = repo.Q(conn)
        for k in range(n):
            uid = uuid.uuid4()
            await conn.execute("insert into auth.users (id, is_anonymous) values ($1, true)", uid)
            await conn.execute("insert into public.profiles (user_id) values ($1)", uid)
            if k % 5:
                persona = specs[k % len(specs)].id
                if await repo.activate_persona(q, uid, persona, replace=True) != "ok":
                    raise SystemExit(f"could not activate {persona}")
                out.append({"uid": str(uid), "kind": "persona", "persona": persona})
            else:
                topics = COLD_TOPICS[(k // 5) % len(COLD_TOPICS)]
                await repo.set_topics(q, uid, topics)
                out.append({"uid": str(uid), "kind": "cold", "topics": topics})
    finally:
        await conn.close()
    return out


async def main(n_users: int) -> None:
    settings = Settings()
    admin_dsn = settings.database_url.get_secret_value()
    local_only(admin_dsn)
    dsn = with_database(admin_dsn, TARGET_DB)
    redis_url = settings.redis_url.get_secret_value().rsplit("/", 1)[0] + f"/{REDIS_DB}"
    shift_s = await copy_shifted(admin_dsn)
    print(f"copied {SOURCE_DB} -> {TARGET_DB}, shifted by {shift_s / 3600:.1f} h")
    published = await publish(dsn, redis_url)
    print(f"published the pool snapshot: {json.dumps(published)}")
    users = await readers(dsn, n_users)
    spare = [u for u in await readers(dsn, SESSION_READERS * 5 // 4) if u["kind"] == "persona"]
    signer = Signer(kid="loadtest-1")
    TOKENS.mkdir(exist_ok=True)
    signer.write_jwks(TOKENS / "jwks.json")
    for u in users + spare:
        u["token"] = signer.token(str(u["uid"]), exp_in=TOKEN_HOURS * 3600)
    (TOKENS / "tokens.json").write_text(json.dumps(users))
    (TOKENS / "sessions.json").write_text(json.dumps(spare[:SESSION_READERS]))
    print(f"{len(users)} + {SESSION_READERS} readers and tokens in {TOKENS} "
          f"(valid {TOKEN_HOURS} h)")  # fmt: skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=200)
    asyncio.run(main(parser.parse_args().users))
