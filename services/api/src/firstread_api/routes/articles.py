"""GET /v1/articles/{id}/why: why the feed would show this item to the caller."""

import uuid

import structlog
from fastapi import APIRouter, Depends, Request

from firstread_api import repo
from firstread_api.auth import Principal
from firstread_api.deps import Services, ensure_profile, limit_ip, services, user
from firstread_api.errors import ERROR_RESPONSES, request_id_of
from firstread_api.schemas import Components, ReasonOut, SimilarRead, WhyResponse
from firstread_api.why import explain

log = structlog.get_logger("firstread_api.why")
router = APIRouter(tags=["feed"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)


@router.get("/v1/articles/{article_id}/why", response_model=WhyResponse, operation_id="getWhy")
async def get_why(
    article_id: uuid.UUID,
    request: Request,
    svc: Services = Depends(services),  # noqa: B008
    who: Principal = Depends(user("feed")),  # noqa: B008
) -> WhyResponse:
    """The blend's components and reasons for this article, recomputed for the caller
    from their own history and topics (no slate needed)."""
    async with svc.pool.acquire() as conn:
        await ensure_profile(svc, repo.Q(conn), who)
    rid = request_id_of(request)
    x = await explain(svc, who.user_id, article_id, rid)
    log.info("why", degraded=x.degraded, reasons=[r.code for r in x.reasons])
    return WhyResponse(
        request_id=rid,
        model_version=svc.settings.model_version,
        article_id=x.article.id,
        title=x.article.title,
        category=x.article.category,
        components=Components(**{k: round(v, 6) for k, v in x.components.items()}),
        age_hours=round(x.age_hours, 3),
        outlets=x.outlets,
        followed_topic=x.followed,
        topic_weight=round(x.topic_weight, 4),
        similar_reads=[SimilarRead(article_id=i, title=t, similarity=s) for i, t, s in x.similar],
        reasons=[ReasonOut(code=r.code, text=r.text) for r in x.reasons],
        degraded=x.degraded,
    )
