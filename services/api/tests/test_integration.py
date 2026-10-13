"""Readiness against real Postgres + Redis (`make up` locally, service containers in CI).

Skipped when the services are unreachable, unless REQUIRE_INTEGRATION=1 (set in CI),
in which case an unreachable service is a failure, not a skip.
"""

import asyncio
import os

import asyncpg
import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from firstread_api.config import Settings
from firstread_api.main import create_app

pytestmark = pytest.mark.integration


async def _services_up(settings: Settings) -> bool:
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(settings.database_url.get_secret_value()), timeout=2
        )
        await conn.close()
        redis = Redis.from_url(settings.redis_url.get_secret_value())
        try:
            await asyncio.wait_for(redis.ping(), timeout=2)
        finally:
            await redis.aclose()
    except Exception:  # any failure means "not reachable"
        return False
    return True


@pytest.fixture
def live_settings() -> Settings:
    settings = Settings(app_env="test")
    if not asyncio.run(_services_up(settings)):
        if os.environ.get("REQUIRE_INTEGRATION") == "1":
            pytest.fail("Postgres/Redis unreachable but REQUIRE_INTEGRATION=1")
        pytest.skip("Postgres/Redis not reachable; run `make up` to enable")
    return settings


def test_readyz_against_real_services(live_settings: Settings) -> None:
    with TestClient(create_app(live_settings)) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ready", "checks": {"postgres": "ok", "redis": "ok"}}


def test_pgvector_extension_available(live_settings: Settings) -> None:
    async def check() -> str | None:
        conn = await asyncpg.connect(live_settings.database_url.get_secret_value())
        try:
            row: str | None = await conn.fetchval(
                "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
            )
            return row
        finally:
            await conn.close()

    assert asyncio.run(check()) is not None
