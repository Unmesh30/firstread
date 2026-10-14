"""Test support: a throwaway Postgres database per test (used by core and ingest tests).

``DATABASE_URL`` must point at a server where the user may ``create database``
(the local Docker stack and the CI service container both allow it). Without a
reachable server the tests skip, unless ``REQUIRE_INTEGRATION=1`` (CI), where an
unreachable server is a failure.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

from firstread_core.db import connect, default_migrations_dir

DEFAULT_DSN = "postgresql://firstread:firstread@localhost:5432/firstread"
MIGRATIONS = default_migrations_dir()


def admin_dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def with_database(dsn: str, name: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


async def server_reachable(dsn: str | None = None) -> bool:
    try:
        conn = await asyncio.wait_for(connect(dsn or admin_dsn()), timeout=3)
    except Exception:  # any failure means "not reachable"
        return False
    await conn.close()
    return True


def skip_or_fail_if_unreachable() -> None:
    """For sync tests (it runs its own event loop)."""
    _skip_or_fail(asyncio.run(server_reachable()))


async def skip_or_fail_if_unreachable_async() -> None:
    """For async tests, which already run inside an event loop."""
    _skip_or_fail(await server_reachable())


def _skip_or_fail(reachable: bool) -> None:
    import pytest

    if reachable:
        return
    if os.environ.get("REQUIRE_INTEGRATION") == "1":
        pytest.fail("Postgres unreachable but REQUIRE_INTEGRATION=1")
    pytest.skip("Postgres not reachable; run `make up` to enable")


@asynccontextmanager
async def temp_database() -> AsyncIterator[str]:
    """Create an empty database, yield its DSN, drop it afterwards."""
    name = f"fr_test_{uuid.uuid4().hex[:12]}"
    admin = await connect(admin_dsn())
    try:
        await admin.execute(f'create database "{name}"')
    finally:
        await admin.close()
    try:
        yield with_database(admin_dsn(), name)
    finally:
        admin = await connect(admin_dsn())
        try:
            await admin.execute(f'drop database if exists "{name}" with (force)')
        finally:
            await admin.close()
