"""Readiness against real Postgres + Redis (`make up` locally, service containers in CI).

Uses a throwaway, migrated database: the API's pool registers pgvector's codecs on
every connection, so an unmigrated database (no `extensions.halfvec`) is correctly
*not ready*. Skipped when the services are unreachable, unless REQUIRE_INTEGRATION=1.
"""

from pathlib import Path

import pytest

from firstread_api.testing import live_api
from firstread_core.dbtest import skip_or_fail_if_unreachable_async

pytestmark = pytest.mark.integration


async def test_readyz_against_real_services(tmp_path: Path) -> None:
    await skip_or_fail_if_unreachable_async()
    async with live_api(tmp_path) as api:
        resp = await api.client.get("/readyz")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ready", "checks": {"postgres": "ok", "redis": "ok"}}


async def test_pgvector_extension_available(tmp_path: Path) -> None:
    await skip_or_fail_if_unreachable_async()
    async with live_api(tmp_path) as api:
        conn = await api.db()
        try:
            version = await conn.fetchval(
                "select extversion from pg_extension where extname = 'vector'"
            )
        finally:
            await conn.close()
    assert version is not None
