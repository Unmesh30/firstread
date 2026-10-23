"""One error shape for every 4xx and 5xx (ADR 0015).

    {"error": {"code": "rate_limited", "message": "...", "details": [...]},
     "request_id": "..."}

Handlers cover our own :class:`ApiError`, FastAPI/Starlette HTTP errors, request
validation and anything unexpected (a 500 that never leaks the exception text).
"""

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

log = structlog.get_logger("firstread_api.errors")

STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    500: "internal",
    503: "unavailable",
}


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        code: str | None = None,
        headers: dict[str, str] | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code or STATUS_CODES.get(status, "error")
        self.message = message
        self.headers = headers or {}
        self.details = details


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def error_response(
    request: Request,
    status: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details),
        request_id=request_id_of(request),
    )
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, headers=headers)


# Shown in the OpenAPI spec for every route.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    s: {"model": ErrorResponse} for s in (401, 422, 429, 500, 503)
}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(request, exc.status, exc.code, exc.message, exc.headers, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = STATUS_CODES.get(exc.status_code, "error")
        message = exc.detail if isinstance(exc.detail, str) else code.replace("_", " ")
        return error_response(request, exc.status_code, code, message, dict(exc.headers or {}))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))}
            for e in exc.errors()
        ]
        return error_response(request, 422, "invalid_request", "invalid request", None, details)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The type only: driver messages can carry connection strings.
        log.error("unhandled_error", error=type(exc).__name__)
        return error_response(request, 500, "internal", "internal error")
