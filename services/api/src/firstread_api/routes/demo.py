"""Demo readers: GET /v1/demo/personas (public), POST .../{id}/activate (ADR 0018)."""

from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Path, Query, Request, Response

from firstread_api import redis_state, repo
from firstread_api.auth import Principal
from firstread_api.deps import Services, ensure_profile, limit_ip, services, user
from firstread_api.errors import ERROR_RESPONSES, ApiError, ErrorResponse, request_id_of
from firstread_api.routes.me import me_response
from firstread_api.schemas import MeResponse, PersonaOut, PersonaSampleOut, PersonasResponse
from firstread_core.personas import parse_all

log = structlog.get_logger("firstread_api.demo")
router = APIRouter(tags=["demo"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)
PUBLIC_CACHE = "public, max-age=60"


def no_snapshot() -> ApiError:
    return ApiError(503, "no pool snapshot yet", "unavailable", headers={"Retry-After": "5"})


@router.get("/v1/demo/personas", response_model=PersonasResponse, operation_id="listPersonas")
async def list_personas(
    request: Request,
    response: Response,
    svc: Services = Depends(services),  # noqa: B008
) -> PersonasResponse:
    """The demo readers, with their topics and sample headlines. Served from the
    in-process pool snapshot: no database read."""
    snap = await svc.pools.get()
    if snap is None:
        raise no_snapshot()
    response.headers["Cache-Control"] = PUBLIC_CACHE
    return PersonasResponse(
        request_id=request_id_of(request),
        model_version=svc.settings.model_version,
        personas=[
            PersonaOut(
                id=p.id,
                name=p.name,
                description=p.description,
                topics=list(p.topics),
                articles=p.articles,
                refreshed_at=datetime.fromtimestamp(p.refreshed_at, UTC),
                samples=[
                    PersonaSampleOut(title=s.title, source_name=s.source_name, category=s.category)
                    for s in p.samples
                ],
            )
            for p in parse_all(snap.personas)
        ],
    )


@router.post(
    "/v1/demo/personas/{persona_id}/activate",
    response_model=MeResponse,
    operation_id="activatePersona",
    responses={409: {"model": ErrorResponse}},
)
async def activate_persona(
    request: Request,
    persona_id: Annotated[str, Path(pattern=r"^[a-z0-9-]{1,40}$")],
    replace: Annotated[bool, Query()] = False,
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("writes")),  # noqa: B008
) -> MeResponse:
    """Load a demo reader's history and topics into this session.

    A reader with their own history or topics (and no persona yet) gets **409
    ``history_exists``** unless ``replace=true``: the app asks first. Switching from
    one persona to another needs no flag. Hidden sources and articles are kept."""
    async with svc.pool.acquire() as conn:
        q = repo.Q(conn)
        await ensure_profile(svc, q, who)
        result = await repo.activate_persona(q, who.user_id, persona_id, replace)
    if result == "unknown":
        raise ApiError(404, "persona not found")
    if result == "empty":
        raise ApiError(503, "this persona has no articles yet", "unavailable",
                       headers={"Retry-After": "60"})  # fmt: skip
    if result == "history_exists":
        raise ApiError(409, "this reader has their own history or topics; "
                       "pass replace=true to replace them", "history_exists")  # fmt: skip
    uid = who.user_id
    try:  # best effort, like topics: the history version check rebuilds the list anyway
        await svc.redis.delete(redis_state.hist_key(uid), redis_state.slate_key(uid))
    except Exception as exc:
        log.warning("persona_redis_delete_failed", error=type(exc).__name__)
    return await me_response(request, svc, who)
