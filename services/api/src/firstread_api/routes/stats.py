"""GET /v1/stats: public numbers for the System page, from the pool snapshot."""

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from firstread_api.deps import Services, limit_ip, services
from firstread_api.errors import ERROR_RESPONSES, request_id_of
from firstread_api.routes.demo import PUBLIC_CACHE, no_snapshot
from firstread_api.schemas import Credit, IngestionStats, StatsResponse
from firstread_core.personas import parse_all

router = APIRouter(tags=["stats"], dependencies=[Depends(limit_ip)], responses=ERROR_RESPONSES)
# SPEC section 4: credit GDELT with a link wherever its data appears.
GDELT_CREDIT = Credit(
    text="News discovery by The GDELT Project", url="https://www.gdeltproject.org/"
)


def _time(v: Any) -> datetime | None:
    return datetime.fromisoformat(v) if isinstance(v, str) else None


@router.get("/v1/stats", response_model=StatsResponse, operation_id="getStats")
async def get_stats(
    request: Request,
    response: Response,
    svc: Services = Depends(services),  # noqa: B008
) -> StatsResponse:
    """Ingestion health and volumes, freshness and the model version. Served from the
    in-process pool snapshot (published by every ingestion run): no database read."""
    snap = await svc.pools.get()
    if snap is None:
        raise no_snapshot()
    st = snap.stats
    runs, ok = int(st.get("runs_24h", 0)), int(st.get("runs_ok_24h", 0))
    done = ok + int(st.get("runs_partial_24h", 0))
    response.headers["Cache-Control"] = PUBLIC_CACHE
    return StatsResponse(
        request_id=request_id_of(request),
        model_version=svc.settings.model_version,
        ingestion=IngestionStats(
            last_run_finished=_time(st.get("last_run_finished")),
            last_run_status=st.get("last_run_status"),
            runs_24h=runs,
            runs_ok_24h=ok,
            success_rate_24h=round(done / runs, 4) if runs else None,
            snapshot_built_at=snap.built_at.astimezone(UTC),
        ),
        articles_24h=int(st.get("articles_24h", 0)),
        sources_24h=int(st.get("sources_24h", 0)),
        median_freshness_lag_s=st.get("median_lag_s"),
        trending_stories=len(snap.trending),
        personas=len(parse_all(snap.personas)),
        instance_uptime_s=round(time.time() - svc.started, 1),
        credit=GDELT_CREDIT,
    )
