"""Migrations on the real Supabase Postgres image (CI job `supabase`).

Runs only when SUPABASE_TEST_URL points at a `supabase/postgres` container, where
`postgres` is not a superuser and the `anon` / `authenticated` roles exist, as on
hosted Supabase. Checks:

* the migrations apply (twice) and never touch the `auth` schema (0007's shim is a
  no-op where `auth` exists);
* RLS lets anon read articles but write nothing; service tables are invisible;
* user tables: deleting an auth user cascades; `authenticated` sees only its own
  rows (with the JWT claims set as PostgREST does); anon sees nothing; nobody writes;
* pgvector is 0.8+ (iterative index scans, P3d).
"""

import os
import uuid

import asyncpg
import pytest

from firstread_core.db import apply_migrations, connect, load_migrations
from firstread_core.dbtest import MIGRATIONS

URL = os.environ.get("SUPABASE_TEST_URL")
pytestmark = [
    pytest.mark.supabase,
    pytest.mark.skipif(not URL, reason="SUPABASE_TEST_URL not set (CI job `supabase`)"),
]
TABLES = (
    "articles",
    "ingestion_runs",
    "feed_state",
    "story_coverage",
    "source_cursor",
    "schema_migrations",
)
INSERTS = {"articles": "url", "feed_state": "feed_id"}
USER_TABLES = ("profiles", "user_history", "events", "served_slates")
AUTH_OBJECTS = (
    "select coalesce(string_agg(c.relname || ':' || c.relkind::text, ',' order by c.relname), '') "
    "|| '|' || coalesce((select string_agg(p.proname || ':' || md5(p.prosrc), ',' "
    "order by p.proname) from pg_proc p where p.pronamespace = 'auth'::regnamespace), '') "
    "from pg_class c where c.relnamespace = 'auth'::regnamespace"
)


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


async def as_user(conn: asyncpg.Connection, uid: uuid.UUID, sql: str) -> list[asyncpg.Record]:
    """Run ``sql`` as the authenticated role with ``uid``'s claims, like PostgREST."""
    async with conn.transaction():
        await conn.execute("set local role authenticated")
        await conn.execute(f"set local request.jwt.claim.sub = '{uid}'")
        await conn.execute(
            f"""set local request.jwt.claims = '{{"sub": "{uid}", "role": "authenticated"}}'"""
        )
        rows: list[asyncpg.Record] = await conn.fetch(sql)
    return rows


async def test_migrations_apply_as_a_non_superuser_and_rls_holds() -> None:
    assert URL is not None
    conn = await connect(URL)
    try:
        assert not await conn.fetchval("select rolsuper from pg_roles where rolname = current_user")
        auth_before = await conn.fetchval(AUTH_OBJECTS)
        await apply_migrations(conn, MIGRATIONS)
        assert await conn.fetchval(AUTH_OBJECTS) == auth_before  # auth is never touched
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


async def test_user_tables_cascade_and_are_owner_only() -> None:
    assert URL is not None
    conn = await connect(URL)
    try:
        await apply_migrations(conn, MIGRATIONS)
        version = await conn.fetchval(
            "select extversion from pg_extension where extname = 'vector'"
        )
        assert tuple(int(x) for x in version.split(".")[:2]) >= (0, 8)
        a, b = uuid.uuid4(), uuid.uuid4()
        for uid in (a, b):
            await conn.execute("insert into auth.users (id) values ($1)", uid)
            await conn.execute("insert into public.profiles (user_id) values ($1)", uid)
            await conn.execute(
                "insert into public.events (id, user_id, article_id, type) "
                "values ($1, $2, $3, 'click')",
                uuid.uuid4(), uid, uuid.uuid4(),
            )  # fmt: skip
        for table in ("profiles", "events"):
            seen = await as_user(conn, a, f"select user_id from public.{table}")  # noqa: S608
            assert [r["user_id"] for r in seen] == [a], table  # owner-only
        for table in USER_TABLES:
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await count_as_any(conn, "anon", table)  # anon: no grant at all
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await as_user(conn, a, f"delete from public.{table}")  # noqa: S608
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await as_user(conn, a, f"insert into public.profiles (user_id) values ('{a}')")  # noqa: S608
        await conn.execute("delete from auth.users where id = $1", a)
        for table in ("profiles", "events"):
            n = await conn.fetchval(f"select count(*) from public.{table} where user_id = $1", a)  # noqa: S608
            assert n == 0, table  # cascaded
        assert (
            await conn.fetchval("select count(*) from public.profiles where user_id = $1", b) == 1
        )
    finally:
        await conn.close()


async def count_as_any(conn: asyncpg.Connection, role: str, table: str) -> int:
    assert table in USER_TABLES
    async with conn.transaction():
        await conn.execute(f"set local role {role}")
        n: int = await conn.fetchval(f"select count(*) from public.{table}")  # noqa: S608 (constant)
    return n
