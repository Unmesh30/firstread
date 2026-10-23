"""Structured JSON logging with a per-request ``request_id`` and a hashed user id.

* ``request_id`` is always generated here. It becomes the ``served_slates`` key and
  the id events join on (P3d), so a client must not be able to choose it: a retry
  with the same header would collide. A valid inbound ``X-Request-ID``
  (``[A-Za-z0-9._-]{8,64}``) is only logged as ``client_request_id``, for tracing.
* User ids are logged as ``uid_hash`` = HMAC-SHA256(LOG_UID_SALT, uid)[:16]
  (SPEC section 12), never raw.
* Request metrics use the matched route's *template* as the label.
"""

import hashlib
import hmac
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response

from firstread_api.metrics import HTTP_LATENCY, HTTP_REQUESTS

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

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


def client_request_id(value: str | None) -> str | None:
    """The client's id if it looks like one, else None (never trusted as ours)."""
    return value if value and _VALID_REQUEST_ID.match(value) else None


def uid_hash(uid: uuid.UUID, salt: str) -> str:
    return hmac.new(salt.encode(), uid.bytes, hashlib.sha256).hexdigest()[:16]


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "unmatched"


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach a request_id to every log line and response, log latency, count requests."""
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    client_id = client_request_id(request.headers.get(REQUEST_ID_HEADER))
    if client_id:
        structlog.contextvars.bind_contextvars(client_request_id=client_id)
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers[REQUEST_ID_HEADER] = request_id
    route = route_template(request)
    HTTP_REQUESTS.labels(route, request.method, str(response.status_code)).inc()
    HTTP_LATENCY.labels(route).observe(elapsed)
    log.info(
        "request",
        method=request.method,
        route=route,
        status=response.status_code,
        latency_ms=round(elapsed * 1000, 2),
    )
    return response
