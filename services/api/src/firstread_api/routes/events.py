"""POST /v1/events: idempotent batches of reader events (ADR 0016)."""

import structlog
from fastapi import APIRouter, Depends, Request

from firstread_api import redis_state, repo
from firstread_api.auth import Principal
from firstread_api.deps import Services, ensure_profile, limit_ip, services, user
from firstread_api.errors import ERROR_RESPONSES, ErrorResponse, request_id_of
from firstread_api.metrics import EVENTS
from firstread_api.schemas import EventsBatch, EventsResult, RejectedEvent

log = structlog.get_logger("firstread_api.events")
router = APIRouter(tags=["events"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)


@router.post(
    "/v1/events", response_model=EventsResult, operation_id="postEvents",
    responses={413: {"model": ErrorResponse, "description": "Body over 64 KiB (bodylimit.py)"}},
)  # fmt: skip
async def post_events(
    body: EventsBatch,
    request: Request,
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("events")),  # noqa: B008
) -> EventsResult:
    """Record up to 100 events in at most 64 KiB. Retrying a batch is safe: an
    ``event_id`` already stored is reported as ``duplicate`` and has no further effect."""
    async with svc.pool.acquire() as conn:
        q = repo.Q(conn)
        await ensure_profile(svc, q, who)
        out = await repo.process_events(q, who.user_id, body.events, svc.settings.model_version)
    types = {e.event_id: e.type for e in body.events}
    for ids, result in ((out.accepted, "accepted"), (out.duplicate, "duplicate"),
                        (out.rejected, "rejected")):  # fmt: skip
        for i in ids:
            EVENTS.labels(types[i], result).inc()
    # Redis side effects for inserted rows only, after the commit, append-only. If Redis
    # is down, the next feed finds the version gap and rebuilds from Postgres.
    uid = who.user_id
    pipe = svc.redis.pipeline()
    redis_state.append_history(pipe, uid, out.history)
    redis_state.add_members(pipe, redis_state.hidden_key(uid), out.hidden, redis_state.USER_TTL_S)
    redis_state.add_members(pipe, redis_state.seen_key(uid), out.seen, redis_state.SEEN_TTL_S)
    # The slate is kept: it holds the served ids, and its stored history version makes
    # the next page re-rank after a click without repeating anything (P3d).
    try:
        await pipe.execute()
    except Exception as exc:  # Postgres has it; the version check heals Redis
        log.warning("events_redis_failed", error=type(exc).__name__)
    return EventsResult(
        request_id=request_id_of(request),
        model_version=svc.settings.model_version,
        accepted=out.accepted,
        duplicate=out.duplicate,
        rejected=[RejectedEvent(event_id=i, code="unknown_article") for i in out.rejected],
    )
