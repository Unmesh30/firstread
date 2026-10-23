"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from redis.asyncio import Redis

from firstread_api import __version__, errors
from firstread_api.auth import JWKSCache, Verifier
from firstread_api.config import Settings, get_settings
from firstread_api.deps import Services
from firstread_api.logging import (
    REQUEST_ID_HEADER,
    configure_logging,
    log,
    request_context_middleware,
)
from firstread_api.ratelimit import RateLimiter
from firstread_api.redis_state import CountingRedis
from firstread_api.routes import events, me, ops, topics
from firstread_core.db import create_pool


def operation_id(route: APIRoute) -> str:
    """Stable operationIds (P4 generates its client from them): the explicit
    ``operation_id`` when set, otherwise the route's name."""
    return route.operation_id or route.name


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # min_size=0: no connection is opened at startup, so the process starts (and
    # /healthz answers) even while the database is still booting.
    pool = await create_pool(
        settings.database_url.get_secret_value(),
        vectors=True,
        min_size=0,
        max_size=settings.db_pool_max,
    )
    redis = CountingRedis(Redis.from_url(settings.redis_url.get_secret_value()))
    per_min = {
        "ip": settings.rl_ip_per_min,
        "feed": settings.rl_user_feed_per_min,
        "events": settings.rl_user_events_per_min,
        "writes": settings.rl_user_writes_per_min,
    }
    app.state.services = Services(
        settings=settings,
        pool=pool,
        redis=redis,
        verifier=Verifier(settings, JWKSCache(settings)),
        limits={k: RateLimiter(v, settings.rl_max_keys) for k, v in per_min.items()},
    )
    log.info("startup", app_env=settings.app_env, version=__version__,
             model_version=settings.model_version)  # fmt: skip
    try:
        yield
    finally:
        await redis.aclose()
        await pool.close()
        log.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="FirstRead API",
        version=__version__,
        lifespan=lifespan,
        generate_unique_id_function=operation_id,
    )
    app.state.settings = settings
    errors.install(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # bearer tokens, not cookies
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER, "Server-Timing", "Retry-After"],
        max_age=600,
    )
    app.middleware("http")(request_context_middleware)
    for r in (ops.router, topics.router, me.router, events.router):
        app.include_router(r)
    return app


app = create_app()
