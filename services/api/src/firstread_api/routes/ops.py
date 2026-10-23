"""Liveness, readiness and metrics. Not rate-limited (Cloud Run probes them)."""

import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from firstread_api.deps import Services, services
from firstread_api.errors import ApiError
from firstread_api.health import (
    Check,
    HealthResponse,
    ReadyResponse,
    model_check,
    postgres_check,
    redis_check,
    run_checks,
)
from firstread_api.metrics import render

router = APIRouter(tags=["ops"])


def get_readiness_checks(request: Request) -> dict[str, Check]:
    svc: Services = request.app.state.services
    checks = {"postgres": postgres_check(svc.pool), "redis": redis_check(svc.redis)}
    if svc.settings.api_load_model:  # prod always loads it (config refuses otherwise)
        checks["model"] = model_check(svc.batcher is not None)
    return checks


@router.get("/healthz", response_model=HealthResponse, operation_id="getHealth")
async def healthz() -> HealthResponse:
    """Liveness: the process is up. Never checks dependencies."""
    return HealthResponse()


@router.get(
    "/readyz",
    response_model=ReadyResponse,
    operation_id="getReady",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
async def readyz(
    response: Response,
    checks: Annotated[dict[str, Check], Depends(get_readiness_checks)],
    svc: Services = Depends(services),  # noqa: B008
) -> ReadyResponse:
    """Readiness: Postgres and Redis are reachable and the encoder is loaded."""
    result = await run_checks(checks, svc.settings.readyz_timeout_s)
    if svc.batcher is not None:
        result.model_tag = svc.batcher.encoder.tag
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request, svc: Services = Depends(services)) -> Response:  # noqa: B008
    """Prometheus metrics. Outside local, needs ``Authorization: Bearer $METRICS_TOKEN``."""
    token = svc.settings.metrics_token
    if svc.settings.app_env != "local" and token is not None:
        given = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.get_secret_value().encode()):
            raise ApiError(401, "metrics need a token", headers={"WWW-Authenticate": "Bearer"})
    elif svc.settings.app_env == "prod":
        raise ApiError(401, "metrics need a token")
    return Response(render(), media_type="text/plain; version=0.0.4")
