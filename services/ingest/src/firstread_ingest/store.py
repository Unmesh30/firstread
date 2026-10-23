"""Postgres reads and writes for ingestion. Everything else in the pipeline is pure.

* Vectors are **read** with pgvector's asyncpg codec (binary, registered for the
  ``extensions`` schema) and **written** as text cast to ``extensions.halfvec(384)``
  inside one ``insert ... select from unnest(...)``, so a batch is one statement.
* ``on conflict (url_canonical) do nothing returning`` makes the upsert idempotent
  and tells us which rows were actually new.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np

# Egress accounting lives in core (the API meters its queries the same way).
from firstread_core.db import Metered as Metered
from firstread_core.db import rows_bytes as rows_bytes
from firstread_core.db import value_bytes as value_bytes
from firstread_core.embedding import EMBED_DIM
from firstread_ingest.dedupe import Recent
from firstread_ingest.sources.rss import FeedState


async def register_codecs(conn: Any) -> None:
    from pgvector.asyncpg import register_vector

    await register_vector(conn, schema="extensions")


@dataclass(frozen=True)
class ArticleRow:
    id: uuid.UUID
    url: str
    url_canonical: str
    title: str
    source_name: str
    source_domain: str
    image_url: str | None
    language: str
    source_country: str | None
    geo: str
    category: str
    category_confidence: float
    category_source: str
    published_at: datetime | None
    first_seen_at: datetime
    embedding: np.ndarray
    embedding_model: str
    text_source: str
    title_fingerprint: str
    story_cluster_id: uuid.UUID
    origin: str


def _vec_text(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v.tolist()) + "]"


async def load_recent(
    conn: Any, now: datetime, window: timedelta, fingerprint_window: timedelta
) -> Recent:
    rows = await conn.fetch(
        "select id, story_cluster_id, text_source, embedding, url_canonical, title_fingerprint, "
        "first_seen_at from public.articles where first_seen_at >= $1",
        now - max(window, fingerprint_window),
    )
    emb_rows = [r for r in rows if r["first_seen_at"] >= now - window]
    embeddings = (
        np.stack([np.asarray(r["embedding"].to_numpy(), np.float32) for r in emb_rows])
        if emb_rows
        else np.zeros((0, EMBED_DIM), np.float32)
    )
    fp_cut = now - fingerprint_window
    return Recent(
        ids=[r["id"] for r in emb_rows],
        clusters=[r["story_cluster_id"] for r in emb_rows],
        text_sources=[r["text_source"] for r in emb_rows],
        embeddings=embeddings,
        url_cluster={r["url_canonical"]: r["story_cluster_id"] for r in rows},
        fingerprint_cluster={
            r["title_fingerprint"]: r["story_cluster_id"]
            for r in rows
            if r["first_seen_at"] >= fp_cut
        },
    )


_INSERT = """
insert into public.articles (
  id, url, url_canonical, title, source_name, source_domain, image_url, language,
  source_country, geo, category, category_confidence, category_source, published_at,
  first_seen_at, embedding, embedding_model, text_source, title_fingerprint,
  story_cluster_id, origin)
select id, url, url_canonical, title, source_name, source_domain, image_url, language,
  source_country, geo, category, category_confidence, category_source, published_at,
  first_seen_at, embedding::extensions.halfvec(384), embedding_model, text_source,
  title_fingerprint, story_cluster_id, origin
from unnest(
  $1::uuid[], $2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::text[],
  $8::text[], $9::text[], $10::text[], $11::text[], $12::real[], $13::text[],
  $14::timestamptz[], $15::timestamptz[], $16::text[], $17::text[], $18::text[],
  $19::text[], $20::uuid[], $21::text[]
) as t(id, url, url_canonical, title, source_name, source_domain, image_url, language,
  source_country, geo, category, category_confidence, category_source, published_at,
  first_seen_at, embedding, embedding_model, text_source, title_fingerprint,
  story_cluster_id, origin)
on conflict (url_canonical) do nothing
returning url_canonical
"""


async def insert_articles(conn: Any, rows: Sequence[ArticleRow]) -> set[str]:
    """Insert new articles; return the canonical URLs that were actually inserted."""
    if not rows:
        return set()
    cols = [
        [r.id for r in rows],
        [r.url for r in rows],
        [r.url_canonical for r in rows],
        [r.title for r in rows],
        [r.source_name for r in rows],
        [r.source_domain for r in rows],
        [r.image_url for r in rows],
        [r.language for r in rows],
        [r.source_country for r in rows],
        [r.geo for r in rows],
        [r.category for r in rows],
        [r.category_confidence for r in rows],
        [r.category_source for r in rows],
        [r.published_at for r in rows],
        [r.first_seen_at for r in rows],
        [_vec_text(r.embedding) for r in rows],
        [r.embedding_model for r in rows],
        [r.text_source for r in rows],
        [r.title_fingerprint for r in rows],
        [r.story_cluster_id for r in rows],
        [r.origin for r in rows],
    ]
    out = await conn.fetch(_INSERT, *cols)
    return {r["url_canonical"] for r in out}


async def insert_coverage(conn: Any, rows: Sequence[tuple[uuid.UUID, str, datetime]]) -> None:
    """(story_cluster_id, source_domain, first_seen_at); the first sighting wins."""
    if not rows:
        return
    await conn.execute(
        "insert into public.story_coverage (story_cluster_id, source_domain, first_seen_at) "
        "select * from unnest($1::uuid[], $2::text[], $3::timestamptz[]) "
        "on conflict (story_cluster_id, source_domain) do nothing",
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
    )


async def load_feed_states(conn: Any) -> dict[str, FeedState]:
    rows = await conn.fetch(
        "select feed_id, etag, last_modified, last_status, consecutive_failures "
        "from public.feed_state"
    )
    return {
        r["feed_id"]: FeedState(
            r["etag"], r["last_modified"], r["last_status"], r["consecutive_failures"]
        )
        for r in rows
    }


async def save_feed_states(conn: Any, states: dict[str, FeedState], now: datetime) -> None:
    if not states:
        return
    ids = list(states)
    await conn.execute(
        "insert into public.feed_state (feed_id, etag, last_modified, last_status, "
        "last_fetched_at, consecutive_failures) "
        "select f, e, m, s, $5, c from unnest($1::text[], $2::text[], $3::text[], $4::int[], "
        "$6::int[]) as t(f, e, m, s, c) "
        "on conflict (feed_id) do update set etag = excluded.etag, "
        "last_modified = excluded.last_modified, last_status = excluded.last_status, "
        "last_fetched_at = excluded.last_fetched_at, "
        "consecutive_failures = excluded.consecutive_failures",
        ids,
        [states[i].etag for i in ids],
        [states[i].last_modified for i in ids],
        [states[i].last_status for i in ids],
        now,
        [states[i].consecutive_failures for i in ids],
    )


async def start_run(conn: Any, now: datetime) -> uuid.UUID:
    run_id: uuid.UUID = await conn.fetchval(
        "insert into public.ingestion_runs (started_at) values ($1) returning id", now
    )
    return run_id


async def finish_run(conn: Any, run_id: uuid.UUID, summary: dict[str, Any], now: datetime) -> None:
    await conn.execute(
        "update public.ingestion_runs set finished_at = $2, window_start = $3, window_end = $4, "
        "fetched = $5, inserted = $6, duplicates = $7, near_duplicates = $8, clustered = $9, "
        "errors = $10, gdelt_calls = $11, status = $12, details = $13::jsonb, "
        "redis_commands = $14 where id = $1",
        run_id,
        now,
        summary.get("window_start"),
        summary.get("window_end"),
        summary["fetched"],
        summary["inserted"],
        summary["duplicates"],
        summary["near_duplicates"],
        summary["clustered"],
        summary["errors"],
        summary.get("gdelt_calls", 0),
        summary["status"],
        json.dumps(summary["details"], default=str),
        summary.get("redis_commands", 0),
    )


async def load_cursor(conn: Any, source: str) -> str | None:
    value: str | None = await conn.fetchval(
        "select cursor from public.source_cursor where source = $1", source
    )
    return value


async def save_cursor(conn: Any, source: str, cursor: str, now: datetime) -> None:
    await conn.execute(
        "insert into public.source_cursor (source, cursor, updated_at) values ($1, $2, $3) "
        "on conflict (source) do update set cursor = excluded.cursor, "
        "updated_at = excluded.updated_at",
        source,
        cursor,
        now,
    )


RETENTION_BATCH = 5000


async def _delete_batched(
    conn: Any, table: str, column: str, before: datetime, extra: str = ""
) -> int:
    """Delete rows with ``column < before`` (and ``extra``, a constant SQL condition) in
    batches (by ctid), so no statement runs long."""
    total = 0
    cond = f"{column} < $1" + (f" and {extra}" if extra else "")
    while True:
        n = await conn.fetchval(
            f"with gone as (delete from public.{table} where ctid in ("  # noqa: S608 (constants)
            f"select ctid from public.{table} where {cond} limit {RETENTION_BATCH}) "
            "returning 1) select count(*) from gone",
            before,
        )
        total += int(n)
        if n < RETENTION_BATCH:
            return total


# ---- user data retention and the size guard (ADR 0016) ----

MB = 1_000_000
# Supabase Free goes read-only above 500 MB, and deleted rows free space only after
# vacuum, so the guard tightens early: each step applies above its size.
GUARD_STEPS = (350 * MB, 375 * MB, 400 * MB)


@dataclass(frozen=True)
class UserRetention:
    """How long user data is kept (days). The guard can only shorten these."""

    impressions: float = 7
    events: float = 30  # every event type except impressions
    slates: float = 30
    inactive_anonymous: float = 30  # app rows of anonymous readers not seen for this long
    articles: float = 14


def guard(size_bytes: int, base: UserRetention) -> tuple[UserRetention, int]:
    """Tighten retention by database size; returns (windows, step 0-3).

    * > 350 MB: impressions 2 days;
    * > 375 MB: also slates 7 days and other events 14 days;
    * > 400 MB: also articles 10 days (user_history is never touched: it is small and
      it is what readers' histories are made of).
    """
    step = sum(size_bytes > t for t in GUARD_STEPS)
    r = base
    if step >= 1:
        r = replace(r, impressions=min(r.impressions, 2))
    if step >= 2:
        r = replace(r, slates=min(r.slates, 7), events=min(r.events, 14))
    if step >= 3:
        r = replace(r, articles=min(r.articles, 10))
    return r, step


async def apply_user_retention(conn: Any, now: datetime, r: UserRetention) -> dict[str, int]:
    """Expire events, served slates and inactive anonymous readers' app rows."""
    day = timedelta(days=1)
    out = {
        "events_impressions": await _delete_batched(
            conn, "events", "created_at", now - r.impressions * day, "type = 'impression'"
        ),
        "events_other": await _delete_batched(
            conn, "events", "created_at", now - r.events * day, "type <> 'impression'"
        ),
        "served_slates": await _delete_batched(
            conn, "served_slates", "created_at", now - r.slates * day
        ),
    }
    # Anonymous readers who haven't come back: their app rows go (the auth user itself
    # is Supabase's to clean up, P5). The cascade from profiles isn't relied on.
    stale = await conn.fetch(
        "select user_id from public.profiles where is_anonymous and last_seen_at < $1",
        now - r.inactive_anonymous * day,
    )
    ids = [row["user_id"] for row in stale]
    if ids:
        for table in ("served_slates", "events", "user_history", "profiles"):
            await conn.execute(
                f"delete from public.{table} where user_id = any($1::uuid[])",  # noqa: S608
                ids,
            )
    out["inactive_anonymous_users"] = len(ids)
    return out


VACUUM_TABLES = ("articles", "story_coverage", "events", "served_slates")


async def vacuum(conn: Any, tables: Sequence[str] = VACUUM_TABLES) -> None:
    """Plain VACUUM (ANALYZE): makes deleted rows' space reusable so files stop growing.
    It doesn't shrink files (VACUUM FULL does, with a lock; runbook). Must run outside
    a transaction."""
    for t in tables:
        await conn.execute(f"vacuum (analyze) public.{t}")


async def apply_retention(
    conn: Any,
    now: datetime,
    articles: timedelta,
    coverage: timedelta,
    runs: timedelta,
    feed_ids: Sequence[str] | None = None,
    users: UserRetention | None = None,
) -> dict[str, int]:
    """Delete expired rows; return how many went from each table."""
    out = {
        "articles": await _delete_batched(conn, "articles", "first_seen_at", now - articles),
        "story_coverage": await _delete_batched(
            conn, "story_coverage", "first_seen_at", now - coverage
        ),
        "ingestion_runs": await _delete_batched(conn, "ingestion_runs", "started_at", now - runs),
    }
    if feed_ids is not None:
        gone = await conn.fetch(
            "delete from public.feed_state where not (feed_id = any($1::text[])) returning 1",
            list(feed_ids),
        )
        out["feed_state"] = len(gone)
    if users is not None:
        out |= await apply_user_retention(conn, now, users)
    return out


async def database_size(conn: Any) -> dict[str, int]:
    row = await conn.fetchrow(
        "select pg_database_size(current_database()) as db, "
        "pg_total_relation_size('public.articles') as articles, "
        "pg_total_relation_size('public.story_coverage') as story_coverage"
    )
    return dict(row)
