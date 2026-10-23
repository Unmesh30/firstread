"""The reader's own profile: GET/DELETE /v1/me, PUT /v1/me/topics."""

import structlog
from fastapi import APIRouter, Depends, Request, Response

from firstread_api import redis_state, repo
from firstread_api.auth import Principal
from firstread_api.deps import Services, ensure_profile, forget_profile, limit_ip, services, user
from firstread_api.errors import ERROR_RESPONSES, ApiError, request_id_of
from firstread_api.schemas import HistoryItem, MeResponse, TopicsUpdate

log = structlog.get_logger("firstread_api.me")
router = APIRouter(tags=["me"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)


async def me_response(request: Request, svc: Services, who: Principal) -> MeResponse:
    async with svc.pool.acquire() as conn:
        q = repo.Q(conn)
        await ensure_profile(svc, q, who)
        data = await repo.me(q, who.user_id)
    p = data["profile"]
    return MeResponse(
        request_id=request_id_of(request),
        model_version=svc.settings.model_version,
        user_id=who.user_id,
        is_anonymous=who.is_anonymous,
        topics=list(p["topics"]),
        hidden_sources=list(p["hidden_sources"]),
        active_persona=p["active_persona"],
        history_count=int(p["n"]),
        recent_history=[HistoryItem(**r) for r in data["recent"]],
    )


@router.get("/v1/me", response_model=MeResponse, operation_id="getMe")
async def get_me(
    request: Request,
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("feed")),  # noqa: B008
) -> MeResponse:
    """Profile summary: followed topics, hidden sources, persona, recent history."""
    return await me_response(request, svc, who)


@router.put("/v1/me/topics", response_model=MeResponse, operation_id="setMyTopics")
async def set_my_topics(
    body: TopicsUpdate,
    request: Request,
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("writes")),  # noqa: B008
) -> MeResponse:
    """Set followed topics (onboarding): 3 to 13 distinct topic ids."""
    async with svc.pool.acquire() as conn:
        q = repo.Q(conn)
        await ensure_profile(svc, q, who)
        await repo.set_topics(q, who.user_id, body.topics)
    try:  # best effort, like the events route: topics are saved in Postgres already
        await svc.redis.delete(redis_state.slate_key(who.user_id))  # the next page re-ranks
    except Exception as exc:
        log.warning("topics_slate_delete_failed", error=type(exc).__name__)
    return await me_response(request, svc, who)


@router.delete(
    "/v1/me", status_code=204, operation_id="deleteMe", response_class=Response,
    responses={204: {"description": "All of this reader's data was deleted"}},
)  # fmt: skip
async def delete_me(
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("writes")),  # noqa: B008
) -> Response:
    """Delete this reader's profile, history, events, served slates and cached state.

    204 only when every copy is gone; 503 with ``Retry-After`` if Redis couldn't be
    cleared (the Postgres rows are already gone; retrying is safe).

    The Supabase Auth user itself is removed in P5 (that needs the service key)."""
    async with svc.pool.acquire() as conn:
        await repo.delete_user_data(repo.Q(conn), who.user_id)
    # Before Redis: whatever happens next, this instance must recreate the profile.
    forget_profile(svc, who.user_id)
    try:
        await redis_state.delete_user(svc.redis, who.user_id)
    except Exception as exc:
        # A 204 must mean every copy is gone (u:{uid}:hist holds vectors for 7 days).
        # Both deletes are idempotent, so the client simply retries.
        log.warning("delete_me_redis_failed", error=type(exc).__name__)
        raise ApiError(503, "could not delete cached data; retry", "unavailable",
                       headers={"Retry-After": "5"}) from None  # fmt: skip
    return Response(status_code=204)
