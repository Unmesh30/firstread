"""The feed's ANN queries on the real Supabase Postgres image (CI job `supabase`).

pgvector must be 0.8+ (iterative index scans), and both query shapes must run there
as the non-superuser ``postgres`` role: ``SET LOCAL hnsw.*`` inside the transaction
(session settings don't survive Supabase's transaction pooler) and the exact scan.
"""

import os
from datetime import UTC, datetime, timedelta

import pytest

from firstread_api import repo
from firstread_api.candidates import Exclusions, ann
from firstread_api.testing import insert_article, unit
from firstread_core.db import apply_migrations, connect, register_vector_codec
from firstread_core.dbtest import MIGRATIONS

URL = os.environ.get("SUPABASE_TEST_URL")
pytestmark = [
    pytest.mark.supabase,
    pytest.mark.skipif(not URL, reason="SUPABASE_TEST_URL not set (CI job `supabase`)"),
]


async def test_both_ann_shapes_run_on_the_supabase_image() -> None:
    assert URL is not None
    conn = await connect(URL)
    try:
        await apply_migrations(conn, MIGRATIONS)
        await register_vector_codec(conn)
        version = await conn.fetchval(
            "select extversion from pg_extension where extname = 'vector'"
        )
        assert tuple(int(x) for x in version.split(".")[:2]) >= (0, 8)
        rows = [await insert_article(conn, unit(k), domain=f"s{k}.example") for k in range(5)]
        since = datetime.now(UTC) - timedelta(hours=72)
        try:
            for mode in ("hnsw", "exact"):
                got = await ann(repo.Q(conn), unit(2), since, Exclusions(ids=[rows[0].id]),
                                limit=3, mode=mode)  # fmt: skip
                assert got[0].id == rows[2].id
                assert rows[0].id not in {c.id for c in got}
            # SET LOCAL ended with the transaction: nothing leaks into the session.
            assert await conn.fetchval("show hnsw.iterative_scan") == "off"
        finally:
            await conn.execute(
                "delete from public.articles where id = any($1::uuid[])", [r.id for r in rows]
            )
    finally:
        await conn.close()
