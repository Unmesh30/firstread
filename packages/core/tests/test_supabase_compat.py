"""Migrations on the real Supabase Postgres image (CI job `supabase`).

Runs only when SUPABASE_TEST_URL points at a `supabase/postgres` container, where
`postgres` is not a superuser and the `anon` / `authenticated` roles exist, as on
hosted Supabase. Checks the migrations apply (twice), and that RLS lets anon read
articles but write nothing.
"""

import os

import asyncpg
import pytest

from firstread_core.db import apply_migrations, connect, load_migrations
from firstread_core.dbtest import MIGRATIONS

URL = os.environ.get("SUPABASE_TEST_URL")
pytestmark = [
    pytest.mark.supabase,
    pytest.mark.skipif(not URL, reason="SUPABASE_TEST_URL not set (CI job `supabase`)"),
]
TABLES = ("articles", "ingestion_runs", "feed_state", "story_coverage", "schema_migrations")
INSERTS = {"articles": "url", "feed_state": "feed_id"}


async def count_as(conn: asyncpg.Connection, role: str, table: str) -> int:
    assert table in TABLES
    async with conn.transaction():
        await conn.execute(f"set local role {role}")
        n: int = await conn.fetchval(f"select count(*) from public.{table}")  # noqa: S608 (constant)
    return n


async def insert_as(conn: asyncpg.Connection, role: str, table: str) -> None:
    async with conn.transaction():
        await conn.execute(f"set local role {role}")
        sql = f"insert into public.{table} ({INSERTS[table]}) values ('x')"  # noqa: S608 (constant)
        await conn.execute(sql)


async def test_migrations_apply_as_a_non_superuser_and_rls_holds() -> None:
    assert URL is not None
    conn = await connect(URL)
    try:
        assert not await conn.fetchval("select rolsuper from pg_roles where rolname = current_user")
        await apply_migrations(conn, MIGRATIONS)
        assert await apply_migrations(conn, MIGRATIONS) == []
        applied = await conn.fetchval("select count(*) from public.schema_migrations")
        assert applied == len(load_migrations(MIGRATIONS))
        for role in ("anon", "authenticated"):
            await count_as(conn, role, "articles")  # public read is allowed
            for table in TABLES[1:]:
                assert await count_as(conn, role, table) == 0  # RLS hides service tables
            for table in INSERTS:
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await insert_as(conn, role, table)
    finally:
        await conn.close()
