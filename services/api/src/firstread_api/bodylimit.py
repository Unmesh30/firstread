"""A request-body cap for the routes that take batches (ADR 0015 amendment, P4).

``POST /v1/events`` accepts at most 64 KiB: the browser's limit for a
``fetch(keepalive)`` body, which is how the app flushes its queue when a page closes
(MDN, RequestInit.keepalive). A declared ``Content-Length`` above the cap is refused
before reading; a chunked body is read up to the cap and refused if it runs past it.
Either way the answer is 413 in the shared error shape.
"""

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.requests import Request

from firstread_api.errors import error_response

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

EVENTS_MAX_BYTES = 64 * 1024
LIMITS: dict[tuple[str, str], int] = {("POST", "/v1/events"): EVENTS_MAX_BYTES}


class BodyLimit:
    def __init__(self, app: ASGIApp, limits: dict[tuple[str, str], int] | None = None) -> None:
        self.app = app
        self.limits = LIMITS if limits is None else limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        cap = None
        if scope["type"] == "http":
            cap = self.limits.get((scope["method"], scope["path"]))
        if cap is None:
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > cap:
            await self._refuse(scope, receive, send, cap)
            return
        # Buffer at most cap + 1 bytes, then replay them to the app.
        messages: list[Message] = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break  # a disconnect: let the app see it
            size += len(message.get("body", b""))
            if size > cap:
                await self._refuse(scope, receive, send, cap)
                return
            if not message.get("more_body", False):
                break
        replay = iter(messages)

        async def replayed() -> Message:
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, replayed, send)

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send, cap: int) -> None:
        response = error_response(
            Request(scope), 413, "payload_too_large", f"request body is larger than {cap} bytes"
        )
        await response(scope, receive, send)
