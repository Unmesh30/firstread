"""Liveness and readiness checks.

`/healthz` answers "is the process alive?" and never touches dependencies.
`/readyz` answers "can this instance serve traffic?" by checking Postgres, Redis and
the encoder (loaded and verified at startup).
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import structlog
from pydantic import BaseModel

log = structlog.get_logger("firstread_api.health")

Check = Callable[[], Awaitable[None]]
CheckStatus = Literal["ok", "error", "timeout"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckStatus]
    model_tag: str | None = None  # the verified encoder release this instance serves


def model_check(loaded: bool) -> Check:
    """The encoder was loaded and verified at startup (it never changes afterwards)."""

    async def check() -> None:
        if not loaded:
            raise RuntimeError("model not loaded")

    return check


def postgres_check(pool: Any) -> Check:
    """Run `SELECT 1` on a pooled asyncpg connection."""

    async def check() -> None:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

    return check


def redis_check(client: Any) -> Check:
    """Send `PING` to Redis."""

    async def check() -> None:
        await client.ping()

    return check


async def _run_one(name: str, check: Check, timeout_s: float) -> CheckStatus:
    try:
        await asyncio.wait_for(check(), timeout=timeout_s)
    except TimeoutError:
        log.warning("readiness_check_failed", check=name, reason="timeout")
        return "timeout"
    except Exception as exc:
        # Log the type only: driver messages can include connection strings.
        log.warning("readiness_check_failed", check=name, reason=type(exc).__name__)
        return "error"
    return "ok"


async def run_checks(checks: dict[str, Check], timeout_s: float) -> ReadyResponse:
    """Run all checks concurrently; ready only if every check passes."""
    names = list(checks)
    results = await asyncio.gather(*(_run_one(n, checks[n], timeout_s) for n in names))
    statuses = dict(zip(names, results, strict=True))
    ready = all(s == "ok" for s in statuses.values())
    return ReadyResponse(status="ready" if ready else "not_ready", checks=statuses)
