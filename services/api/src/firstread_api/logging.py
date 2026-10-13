"""Structured JSON logging with a per-request `request_id`."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response

REQUEST_ID_HEADER = "X-Request-ID"

log = structlog.get_logger("firstread_api")


def configure_logging(level: str) -> None:
    """Route structlog (and stdlib logging) to stdout as one JSON object per line."""
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach a request_id to every log line and response, and log request latency."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    return response
