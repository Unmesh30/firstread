"""GET /v1/feed: the reader's ranked "For You" page (ADR 0017)."""

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response

from firstread_api import repo
from firstread_api.auth import Principal
from firstread_api.deps import Services, ensure_profile, limit_ip, services, user
from firstread_api.errors import ERROR_RESPONSES, ApiError, ErrorResponse, request_id_of
from firstread_api.feed import FeedPipeline, unavailable
from firstread_api.schemas import MAX_PAGE, FeedItem, FeedResponse, ReasonOut
from firstread_api.slate import Cursor

log = structlog.get_logger("firstread_api.feed")
router = APIRouter(
    tags=["feed"],
    dependencies=[Depends(limit_ip)],
    responses={**ERROR_RESPONSES, 400: {"model": ErrorResponse}},
)


@router.get("/v1/feed", response_model=FeedResponse, operation_id="getFeed")
async def get_feed(
    request: Request,
    response: Response,
    limit: Annotated[int | None, Query(ge=1, le=MAX_PAGE)] = None,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("feed")),  # noqa: B008
) -> FeedResponse:
    """A page of the ranked feed. Pass the previous page's ``next_cursor`` for the next
    one: pages never repeat an item, and a click or hide since the last page re-ranks
    the rest. ``degraded`` names any part that failed and was worked around."""
    parsed = Cursor.decode(cursor) if cursor else None  # 400 before any work
    size = limit or svc.ranking.page_size
    request_id = request_id_of(request)
    pipe = FeedPipeline(svc, who.user_id, request_id)
    pipe.t.add("auth", getattr(request.state, "auth_s", 0.0))
    status, n_items = 500, 0  # until it succeeds
    try:
        async with asyncio.timeout(svc.settings.api_feed_timeout_s):
            async with svc.pool.acquire() as conn:
                await ensure_profile(svc, repo.Q(conn), who)
            items, next_cursor = await pipe.run(parsed, size)
        status, n_items = 200, len(items)
    except (TimeoutError, ApiError) as exc:
        if isinstance(exc, TimeoutError):
            pipe.degrade("timeout")
        err = exc if isinstance(exc, ApiError) else unavailable("the feed took too long")
        err.headers["Server-Timing"] = pipe.t.header()
        status = err.status
        raise err from None
    finally:
        pipe.t.observe()
        log.info("feed", status=status, items=n_items, limit=size, cursor=parsed is not None,
                 stages_ms=pipe.t.rounded(), degraded=pipe.degraded)  # fmt: skip
    response.headers["Server-Timing"] = pipe.t.header()
    return FeedResponse(
        request_id=request_id,
        model_version=svc.settings.model_version,
        items=[
            FeedItem(
                id=a.id,
                title=a.title,
                source_name=a.source_name,
                source_domain=a.source_domain,
                url=a.url,
                image_url=a.image_url,
                category=a.category,
                published_at=a.published_at,
                first_seen_at=a.first_seen_at,
                score=i.final,
                reasons=[ReasonOut(code=c, text=t) for c, t in i.reasons],
            )
            for i, a in items
        ],
        next_cursor=next_cursor,
        degraded=pipe.degraded,
    )
