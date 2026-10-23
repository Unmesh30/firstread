"""The pool snapshot, cached in-process and refreshed lazily (ADR 0012/0017).

* At most once per ``check_every_s`` (60 s) per instance, inside a request, the
  cache sends ``GET pools:ver`` (1 command). Only when the version changed does it
  fetch ``GET pools:blob:{ver}`` (~190 KB).
* Single-flight: one refresh at a time. The request that triggers it awaits it (a
  couple of Redis round trips), so no work outlives a response (Cloud Run's
  request-based billing); concurrent requests keep using the current copy.
* A blob whose internal version differs from ``pools:ver``, of an unknown schema, or
  missing keeps the last good copy until the next publish. A cold instance with no
  copy yet serves ANN-only. There's never a per-request SQL fallback (egress).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import structlog

from firstread_api.metrics import POOLS_AGE, POOLS_REFRESH
from firstread_core import pools

log = structlog.get_logger("firstread_api.pools")


class PoolsCache:
    def __init__(
        self, redis: Any, check_every_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.redis = redis
        self.check_every_s = check_every_s
        self.clock = clock
        self.snapshot: pools.PoolSnapshot | None = None
        self.checked_at = -1e18
        self._refreshing: asyncio.Task[None] | None = None

    async def get(self) -> pools.PoolSnapshot | None:
        """The current snapshot. The request that finds the check interval passed runs
        the refresh; a cold instance (no copy yet) waits for an in-flight one."""
        now = self.clock()
        if now - self.checked_at >= self.check_every_s and self._refreshing is None:
            self.checked_at = now
            self._refreshing = asyncio.ensure_future(self._refresh())
            self._refreshing.add_done_callback(self._done)
            await asyncio.shield(self._refreshing)
        elif self.snapshot is None and self._refreshing is not None:
            await asyncio.shield(self._refreshing)
        if self.snapshot is not None:
            POOLS_AGE.set(max(0.0, time.time() - self.snapshot.built_at.timestamp()))
        return self.snapshot

    def _done(self, _: asyncio.Task[None]) -> None:
        self._refreshing = None

    async def _refresh(self) -> None:
        try:
            raw = await self.redis.get(pools.VER_KEY)
            if raw is None:
                POOLS_REFRESH.labels("no_version").inc()
                return
            ver = raw.decode() if isinstance(raw, bytes) else str(raw)
            if self.snapshot is not None and self.snapshot.version == ver:
                POOLS_REFRESH.labels("unchanged").inc()
                return
            blob = await self.redis.get(pools.blob_key(ver))
            if blob is None:
                POOLS_REFRESH.labels("missing_blob").inc()
                return
            self.snapshot = pools.decode(blob, ver)
            POOLS_REFRESH.labels("loaded").inc()
        except pools.SnapshotError as exc:
            POOLS_REFRESH.labels("bad_blob").inc()
            log.warning("pool_snapshot_rejected", error=str(exc))
        except Exception as exc:  # Redis down: keep what we have
            POOLS_REFRESH.labels("error").inc()
            log.warning("pool_snapshot_refresh_failed", error=type(exc).__name__)
