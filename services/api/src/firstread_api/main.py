"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import asyncpg
from fastapi import Depends, FastAPI, Request, Response, status
from redis.asyncio import Redis

from firstread_api import __version__
from firstread_api.config import Settings, get_settings
from firstread_api.health import (
    Check,
    HealthResponse,
    ReadyResponse,
    postgres_check,
    redis_check,
    run_checks,
)
from firstread_api.logging import configure_logging, log, request_context_middleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # min_size=0: no connection is opened at startup, so the process starts (and
    # /healthz answers) even while the database is still booting.
    # statement_cache_size=0: required by the Supabase transaction pooler (port 6543).
    app.state.db_pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(),
        min_size=0,
        max_size=5,
        statement_cache_size=0,
    )
    app.state.redis = Redis.from_url(settings.redis_url.get_secret_value())
    log.info("startup", app_env=settings.app_env, version=__version__)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.db_pool.close()
        log.info("shutdown")


def get_readiness_checks(request: Request) -> dict[str, Check]:
    state = request.app.state
    return {"postgres": postgres_check(state.db_pool), "redis": redis_check(state.redis)}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="FirstRead API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(request_context_middleware)

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness: the process is up. Never checks dependencies."""
        return HealthResponse()

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        tags=["ops"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
    )
    async def readyz(
        response: Response,
        checks: Annotated[dict[str, Check], Depends(get_readiness_checks)],
    ) -> ReadyResponse:
        """Readiness: Postgres and Redis are reachable."""
        result = await run_checks(checks, settings.readyz_timeout_s)
        if result.status != "ready":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    return app


app = create_app()
