"""Postgres helpers shared by ingestion and the API (``firstread-core[db]``).

* :func:`connect` / :func:`create_pool` always pass ``statement_cache_size=0``:
  Supabase's pooler in transaction mode (port 6543) breaks asyncpg's prepared
  statement cache (CLAUDE.md gotcha). It costs little locally.
* :func:`apply_migrations` applies ``NNNN_name.sql`` files in order, each in its
  own transaction, under an advisory lock so two runners can't race. It records
  each file's SHA256 in ``schema_migrations`` and refuses to continue if an
  already-applied file has changed: migrations are append-only.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIGRATION_NAME = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
# Any constant works; it only has to be the same for every runner.
_LOCK_KEY = 0x46_52_53_54  # "FRST"

_BOOKKEEPING = """
create table if not exists public.schema_migrations (
  filename   text primary key,
  sha256     text not null,
  applied_at timestamptz not null default now()
);
alter table public.schema_migrations enable row level security;
"""


def default_migrations_dir() -> Path:
    """``$FIRSTREAD_MIGRATIONS_DIR``, or ``infra/db/migrations`` in this repo checkout."""
    env = os.environ.get("FIRSTREAD_MIGRATIONS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[4] / "infra" / "db" / "migrations"


class MigrationError(RuntimeError):
    """A migration file is malformed or an applied one was edited."""


@dataclass(frozen=True)
class Migration:
    filename: str
    sql: str
    sha256: str


async def connect(dsn: str, **kwargs: Any) -> Any:
    import asyncpg

    return await asyncpg.connect(dsn, statement_cache_size=0, **kwargs)


async def create_pool(dsn: str, **kwargs: Any) -> Any:
    import asyncpg

    return await asyncpg.create_pool(dsn, statement_cache_size=0, **kwargs)


def load_migrations(directory: Path) -> list[Migration]:
    files = sorted(p for p in directory.iterdir() if p.suffix == ".sql")
    bad = [p.name for p in files if not MIGRATION_NAME.match(p.name)]
    if bad:
        raise MigrationError(f"badly named migrations (want NNNN_name.sql): {bad}")
    numbers = [p.name[:4] for p in files]
    if len(set(numbers)) != len(numbers):
        raise MigrationError(f"duplicate migration numbers in {directory}")
    return [
        Migration(p.name, p.read_text(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files
    ]


async def apply_migrations(conn: Any, directory: Path) -> list[str]:
    """Apply pending migrations; return the filenames applied in this call."""
    migrations = load_migrations(directory)
    await conn.execute("select pg_advisory_lock($1)", _LOCK_KEY)
    try:
        await conn.execute(_BOOKKEEPING)
        rows = await conn.fetch("select filename, sha256 from public.schema_migrations")
        applied = {r["filename"]: r["sha256"] for r in rows}
        changed = [m.filename for m in migrations if applied.get(m.filename, m.sha256) != m.sha256]
        if changed:
            raise MigrationError(
                f"applied migrations were edited (add a new one instead): {changed}"
            )
        done = []
        for m in migrations:
            if m.filename in applied:
                continue
            async with conn.transaction():
                await conn.execute(m.sql)
                await conn.execute(
                    "insert into public.schema_migrations (filename, sha256) values ($1, $2)",
                    m.filename,
                    m.sha256,
                )
            done.append(m.filename)
        return done
    finally:
        await conn.execute("select pg_advisory_unlock($1)", _LOCK_KEY)
