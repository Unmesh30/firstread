"""Shared request dependencies: services on app.state, auth, rate limits, profiles."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

import structlog
from fastapi import Depends, Request

from firstread_api import repo
from firstread_api.articles_cache import ArticlesCache
from firstread_api.auth import Principal, Verifier, authenticate
from firstread_api.config import Settings
from firstread_api.encoder import MicroBatcher
from firstread_api.errors import ApiError
from firstread_api.logging import uid_hash
from firstread_api.pools_cache import PoolsCache
from firstread_api.ratelimit import RateLimiter
from firstread_api.redis_state import CountingRedis
from firstread_core.ranking import RankingConfig, load_config

Scope = Literal["feed", "events", "writes"]


@dataclass
class Services:
    settings: Settings
    pool: Any
    redis: CountingRedis
    verifier: Verifier
    limits: dict[str, RateLimiter]
    pools: PoolsCache
    articles: ArticlesCache
    # The pinned user encoder behind the micro-batcher; None only when API_LOAD_MODEL
    # is false (tests that never rank; prod refuses it).
    batcher: MicroBatcher | None = None
    ranking: RankingConfig = field(default_factory=load_config)
    # user -> the day this instance last ensured the profile (at most once a day).
    ensured: OrderedDict[uuid.UUID, date] = field(default_factory=OrderedDict)


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def limit_ip(request: Request, svc: Services = Depends(services)) -> None:  # noqa: B008
    svc.limits["ip"].check(client_ip(request), "ip")


def user(scope: Scope) -> Callable[..., Awaitable[Principal]]:
    """Authenticated reader, rate-limited per user for ``scope``."""

    async def dep(request: Request, svc: Services = Depends(services)) -> Principal:  # noqa: B008
        t0 = time.perf_counter()
        principal = await authenticate(request, svc.settings, svc.verifier)
        request.state.auth_s = time.perf_counter() - t0  # the feed's first stage
        svc.limits[scope].check(str(principal.user_id), scope)
        structlog.contextvars.bind_contextvars(
            uid_hash=uid_hash(principal.user_id, svc.settings.log_uid_salt.get_secret_value())
        )
        return principal

    return dep


async def ensure_profile(svc: Services, q: repo.Q, who: Principal) -> None:
    """Create/touch the profile at most once a day per user per instance."""
    today = datetime.now(UTC).date()
    if svc.ensured.get(who.user_id) == today:
        svc.ensured.move_to_end(who.user_id)
        return
    local = svc.settings.app_env in ("local", "test")
    await repo.ensure_profile(q, who, local_auth=local)
    svc.ensured[who.user_id] = today
    if len(svc.ensured) > svc.settings.rl_max_keys:
        svc.ensured.popitem(last=False)


def forget_profile(svc: Services, uid: uuid.UUID) -> None:
    svc.ensured.pop(uid, None)


def not_found(what: str) -> ApiError:
    return ApiError(404, f"{what} not found")
