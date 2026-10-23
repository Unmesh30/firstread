"""The in-process pool snapshot cache (ADR 0012/0017), against a fake Redis."""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from firstread_api.pools_cache import PoolsCache
from firstread_core import pools


def blob(version: str) -> bytes:
    a = uuid.uuid4()
    snap = pools.PoolSnapshot(
        version, datetime.now(UTC), [a], {"sports": [a]}, [],
        {a: pools.ArticleInfo(a, "sports", 0, "title")},
    )  # fmt: skip
    return pools.encode(snap)


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.calls: list[str] = []
        self.delay = 0.0
        self.down = False

    async def get(self, key: str) -> Any:
        self.calls.append(key)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.down:
            raise ConnectionError("redis down")
        return self.data.get(key)

    def publish(self, version: str, raw: bytes | None = None) -> None:
        self.data[pools.blob_key(version)] = blob(version) if raw is None else raw
        self.data[pools.VER_KEY] = version.encode()


class Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def cache(r: FakeRedis, clock: Clock) -> PoolsCache:
    return PoolsCache(r, check_every_s=60, clock=clock)


async def test_cold_instance_without_a_snapshot_gets_none() -> None:
    r, clock = FakeRedis(), Clock()
    assert await cache(r, clock).get() is None
    assert r.calls == [pools.VER_KEY]


async def test_version_checked_once_a_minute_and_blob_fetched_only_on_change() -> None:
    r, clock = FakeRedis(), Clock()
    r.publish("v1")
    c = cache(r, clock)
    assert (await c.get()).version == "v1"  # type: ignore[union-attr]
    clock.t = 30
    await c.get()
    assert r.calls == [pools.VER_KEY, pools.blob_key("v1")]
    clock.t = 61
    await c.get()  # unchanged: one GET
    r.publish("v2")
    clock.t = 122
    assert (await c.get()).version == "v2"  # type: ignore[union-attr]
    assert r.calls[2:] == [pools.VER_KEY, pools.VER_KEY, pools.blob_key("v2")]


async def test_single_flight_concurrent_requests_share_one_refresh() -> None:
    r, clock = FakeRedis(), Clock()
    r.publish("v1")
    r.delay = 0.02
    c = cache(r, clock)
    got = await asyncio.gather(*(c.get() for _ in range(10)))
    assert all(s is not None and s.version == "v1" for s in got)  # cold: they all waited
    assert r.calls == [pools.VER_KEY, pools.blob_key("v1")]


async def test_warm_requests_keep_the_current_copy_during_a_refresh() -> None:
    r, clock = FakeRedis(), Clock()
    r.publish("v1")
    c = cache(r, clock)
    await c.get()
    r.publish("v2")
    r.delay = 0.05
    clock.t = 61
    refresher = asyncio.ensure_future(c.get())
    await asyncio.sleep(0.01)
    other = await c.get()  # doesn't wait for the refresh
    assert other is not None
    assert other.version == "v1"
    assert (await refresher).version == "v2"  # type: ignore[union-attr]


async def test_bad_or_mismatched_or_missing_blobs_keep_the_last_good_copy() -> None:
    r, clock = FakeRedis(), Clock()
    r.publish("v1")
    c = cache(r, clock)
    await c.get()
    for k, raw in enumerate([b"garbage", blob("other-version")]):
        r.publish(f"v{k + 2}", raw)
        clock.t += 61
        assert (await c.get()).version == "v1"  # type: ignore[union-attr]
    r.data[pools.VER_KEY] = b"v9"  # version moved, blob missing (expired)
    clock.t += 61
    assert (await c.get()).version == "v1"  # type: ignore[union-attr]
    r.down = True
    clock.t += 61
    assert (await c.get()).version == "v1"  # type: ignore[union-attr]
