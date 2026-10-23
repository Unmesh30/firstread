"""In-process token buckets, per user and per client IP (ADR 0015).

Why not Redis: a Redis-backed limiter costs at least one command per request, and
Upstash commands are the binding free-tier budget (ADR 0011). The price is that each
Cloud Run instance (at most 3) keeps its own buckets, so a client can get up to 3x
the limit if requests spread across instances. That's acceptable for abuse
protection at demo scale.

Client IPs come from ``request.client.host``. uvicorn rewrites it from
``X-Forwarded-For`` only for proxies in ``FORWARDED_ALLOW_IPS`` (default 127.0.0.1;
the Cloud Run value is set in P5), so a client can't spoof its own IP.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from firstread_api.errors import ApiError
from firstread_api.metrics import RATE_LIMITED


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """``per_minute`` requests per key, refilled continuously; bursts up to the limit."""

    def __init__(
        self, per_minute: int, max_keys: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.max_keys = max_keys
        self.clock = clock
        self.buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def take(self, key: str) -> float:
        """Spend one token; return 0 if allowed, else the seconds until one is free."""
        now = self.clock()
        b = self.buckets.get(key)
        if b is None:
            b = _Bucket(self.capacity, now)
            self.buckets[key] = b
            if len(self.buckets) > self.max_keys:
                self.buckets.popitem(last=False)  # forget the least recently used key
        else:
            self.buckets.move_to_end(key)
            b.tokens = min(self.capacity, b.tokens + (now - b.updated) * self.rate)
            b.updated = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return 0.0
        return (1.0 - b.tokens) / self.rate

    def check(self, key: str, scope: str) -> None:
        wait = self.take(key)
        if wait > 0:
            RATE_LIMITED.labels(scope).inc()
            raise ApiError(
                429,
                "too many requests",
                "rate_limited",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )
