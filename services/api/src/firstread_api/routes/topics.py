"""GET /v1/topics: the 13 app topics (public)."""

from fastapi import APIRouter, Depends, Request

from firstread_api.deps import Services, limit_ip, services
from firstread_api.errors import ERROR_RESPONSES, request_id_of
from firstread_api.schemas import Topic, TopicsResponse
from firstread_core.topics import TOPICS

router = APIRouter(tags=["topics"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)


@router.get("/v1/topics", response_model=TopicsResponse, operation_id="listTopics")
async def list_topics(request: Request, svc: Services = Depends(services)) -> TopicsResponse:  # noqa: B008
    """Every topic a reader can follow, in display order."""
    return TopicsResponse(
        request_id=request_id_of(request),
        model_version=svc.settings.model_version,
        topics=[Topic(id=k, name=v) for k, v in TOPICS.items()],
    )
