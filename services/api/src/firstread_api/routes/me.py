"""The reader's own profile: GET/DELETE /v1/me, PUT /v1/me/topics."""

from fastapi import APIRouter, Depends, Request, Response

from firstread_api import redis_state, repo
from firstread_api.auth import Principal
from firstread_api.deps import Services, ensure_profile, forget_profile, limit_ip, services, user
from firstread_api.errors import ERROR_RESPONSES, request_id_of
from firstread_api.schemas import HistoryItem, MeResponse, TopicsUpdate

router = APIRouter(tags=["me"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)


async def _me(request: Request, svc: Services, who: Principal) -> MeResponse:
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
    return await _me(request, svc, who)


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
    await svc.redis.delete(redis_state.slate_key(who.user_id))  # the next page re-ranks
    return await _me(request, svc, who)


@router.delete(
    "/v1/me", status_code=204, operation_id="deleteMe", response_class=Response,
    responses={204: {"description": "All of this reader's data was deleted"}},
)  # fmt: skip
async def delete_me(
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("writes")),  # noqa: B008
) -> Response:
    """Delete this reader's profile, history, events, served slates and cached state.

    The Supabase Auth user itself is removed in P5 (that needs the service key)."""
    async with svc.pool.acquire() as conn:
        await repo.delete_user_data(repo.Q(conn), who.user_id)
    await redis_state.delete_user(svc.redis, who.user_id)
    forget_profile(svc, who.user_id)
    return Response(status_code=204)
