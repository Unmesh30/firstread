"""One HTTP client for every source, with retries, backoff and jitter.

``fetch_with_retry`` retries timeouts, connection errors, 429 and 5xx with
exponential backoff and *full jitter* (sleep a random time in
``[0, base * 2**attempt]``), honouring a numeric ``Retry-After`` header. Sleep
and randomness are injectable so tests are deterministic.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping

import httpx

USER_AGENT = "FirstRead/0.1 (+https://github.com/Unmesh30/firstread)"
TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

Sleep = Callable[[float], Awaitable[None]]


def make_client(**kwargs: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        **kwargs,  # type: ignore[arg-type]
    )


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after", "")
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> httpx.Response:
    """GET ``url``; return the last response (callers check its status).

    Raises the last transport error if every attempt failed without a response.
    """
    rng = rng or random.Random()  # noqa: S311 (jitter, not security)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.get(url, headers=headers, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc, resp = exc, None
        if resp is not None and resp.status_code not in RETRY_STATUSES:
            return resp
        if attempt == attempts - 1:
            if resp is not None:
                return resp
            break
        delay = rng.uniform(0, min(max_delay, base_delay * 2**attempt))
        if resp is not None and (hint := _retry_after(resp)) is not None:
            delay = min(max_delay, max(delay, hint))
        await sleep(delay)
    if last_exc is None:  # unreachable: attempts >= 1 always sets one or returns
        raise RuntimeError("fetch_with_retry made no attempts")
    raise last_exc
