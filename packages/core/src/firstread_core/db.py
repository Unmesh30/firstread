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
import uuid
from dataclasses import dataclass
from datetime import datetime
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


async def register_vector_codec(conn: Any) -> None:
    """pgvector's binary codecs (vector, halfvec), registered for the ``extensions``
    schema where pgvector lives on Supabase and locally (ADR 0008)."""
    from pgvector.asyncpg import register_vector

    await register_vector(conn, schema="extensions")


async def create_pool(dsn: str, *, vectors: bool = False, **kwargs: Any) -> Any:
    """An asyncpg pool for the transaction pooler (no statement cache); with
    ``vectors=True`` every connection gets the pgvector codecs."""
    import asyncpg

    if vectors:
        kwargs["init"] = register_vector_codec
    return await asyncpg.create_pool(dsn, statement_cache_size=0, **kwargs)


# Supabase's transaction-mode pooler. Migrations take a session advisory lock, which
# can't be released reliably through it, so they must use the direct/session port.
TRANSACTION_POOLER_PORT = 6543


def check_migration_dsn(dsn: str) -> None:
    """Refuse to migrate through the transaction pooler (ADR 0008)."""
    from urllib.parse import urlsplit

    if urlsplit(dsn).port == TRANSACTION_POOLER_PORT:
        raise MigrationError(
            "migrations must use the direct/session connection (port 5432), not the "
            "6543 transaction pooler: the runner's session advisory lock can't be "
            "released reliably through it"
        )


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


# ---- egress accounting (Supabase Free: 5 GB/month, pooler egress included; ADR 0011) ----


_ROW_OVERHEAD = 11  # DataRow message: type byte, length, field count
_FIELD_OVERHEAD = 4  # per-field length prefix


def value_bytes(v: Any) -> int:
    """Approximate wire size of one result value (binary protocol)."""
    if v is None:
        return 0
    if isinstance(v, bytes | bytearray | memoryview):
        return len(v)
    if isinstance(v, str):
        return len(v.encode())
    if isinstance(v, bool):
        return 1
    if isinstance(v, int | float | datetime):
        return 8
    if isinstance(v, uuid.UUID):
        return 16
    if isinstance(v, list | tuple):
        return 20 + sum(value_bytes(x) + _FIELD_OVERHEAD for x in v)
    dim = getattr(v, "dimensions", None)  # pgvector HalfVector / Vector
    if callable(dim):
        return 4 + 2 * int(dim())
    return 8


def rows_bytes(rows: Any) -> int:
    if rows is None:
        return 0
    if not isinstance(rows, list):
        rows = [rows]
    total = 0
    for r in rows:
        values = r.values() if hasattr(r, "values") else [r]
        total += _ROW_OVERHEAD + sum(value_bytes(v) + _FIELD_OVERHEAD for v in values)
    return total


class Metered:
    """A connection wrapper that estimates result bytes (Supabase bills egress).

    Everything else passes straight through to the asyncpg connection.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.bytes = 0
        self.queries = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def fetch(self, *args: Any, **kwargs: Any) -> Any:
        rows = await self._conn.fetch(*args, **kwargs)
        self.queries += 1
        self.bytes += rows_bytes(rows)
        return rows

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:
        row = await self._conn.fetchrow(*args, **kwargs)
        self.queries += 1
        self.bytes += rows_bytes(row)
        return row

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        v = await self._conn.fetchval(*args, **kwargs)
        self.queries += 1
        self.bytes += _ROW_OVERHEAD + value_bytes(v) + _FIELD_OVERHEAD
        return v

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        status = await self._conn.execute(*args, **kwargs)
        self.queries += 1
        self.bytes += len(str(status)) + 8  # CommandComplete
        return status
