"""Migrations against a real Postgres + pgvector (throwaway database per test)."""

from pathlib import Path

import pytest

from firstread_core.db import MigrationError, apply_migrations, connect, load_migrations
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database


def test_repo_migrations_are_well_formed() -> None:
    names = [m.filename for m in load_migrations(MIGRATIONS)]
    assert names == sorted(names)
    assert names[0] == "0001_extensions.sql"


@pytest.mark.parametrize("bad", ["1_x.sql", "0001-x.sql", "0001_X.sql"])
def test_badly_named_migrations_are_rejected(tmp_path: Path, bad: str) -> None:
    (tmp_path / bad).write_text("select 1;")
    with pytest.raises(MigrationError, match="badly named"):
        load_migrations(tmp_path)


def test_duplicate_numbers_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("select 1;")
    (tmp_path / "0001_b.sql").write_text("select 1;")
    with pytest.raises(MigrationError, match="duplicate"):
        load_migrations(tmp_path)


@pytest.fixture
async def db() -> None:
    await skip_or_fail_if_unreachable_async()


pytestmark_integration = pytest.mark.integration


@pytestmark_integration
async def test_migrations_apply_once_then_are_a_no_op(db: None) -> None:
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            first = await apply_migrations(conn, MIGRATIONS)
            assert first == [m.filename for m in load_migrations(MIGRATIONS)]
            assert await apply_migrations(conn, MIGRATIONS) == []
            ext = await conn.fetchval(
                "select n.nspname from pg_extension e join pg_namespace n "
                "on n.oid = e.extnamespace where e.extname = 'vector'"
            )
            assert ext == "extensions"
        finally:
            await conn.close()


@pytestmark_integration
async def test_editing_an_applied_migration_is_refused(db: None, tmp_path: Path) -> None:
    (tmp_path / "0001_t.sql").write_text("create table public.t (x int);")
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, tmp_path)
            (tmp_path / "0001_t.sql").write_text("create table public.t (x bigint);")
            with pytest.raises(MigrationError, match="edited"):
                await apply_migrations(conn, tmp_path)
        finally:
            await conn.close()


@pytestmark_integration
async def test_every_public_table_has_row_level_security(db: None) -> None:
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            rows = await conn.fetch(
                "select c.relname, c.relrowsecurity from pg_class c "
                "join pg_namespace n on n.oid = c.relnamespace "
                "where n.nspname = 'public' and c.relkind in ('r', 'p')"
            )
            tables = {r["relname"]: r["relrowsecurity"] for r in rows}
            assert {"articles", "ingestion_runs", "feed_state", "story_coverage",
                    "schema_migrations"} <= set(tables)  # fmt: skip
            assert [t for t, rls in tables.items() if not rls] == []
        finally:
            await conn.close()


@pytestmark_integration
async def test_inner_product_search_uses_the_hnsw_index(db: None) -> None:
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            await conn.execute("set enable_seqscan = off")
            rows = await conn.fetch(
                "explain select id from public.articles "
                "order by embedding OPERATOR(extensions.<#>) $1::extensions.halfvec(384) limit 10",
                "[" + ",".join(["0.05"] * 384) + "]",
            )
            plan = "\n".join(r[0] for r in rows)
            assert "articles_embedding_hnsw" in plan, plan
        finally:
            await conn.close()
