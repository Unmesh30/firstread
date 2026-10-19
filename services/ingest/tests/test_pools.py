"""Pools from the window snapshot, and the 3-command publish (real Postgres + Redis db 15)."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from firstread_core import pools as core_pools
from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_ingest import pools, store, window

NOW = datetime(2026, 10, 15, 12, 20, tzinfo=UTC)
MODEL = "m"
pytestmark = pytest.mark.integration


def row(n: int, topic: str, age: timedelta, cluster: uuid.UUID | None = None) -> store.ArticleRow:
    aid = uuid.uuid4()
    v = np.zeros(384, np.float32)
    v[n % 384] = 1.0
    return store.ArticleRow(
        id=aid, url=f"https://e.com/{n}", url_canonical=f"https://e.com/{n}", title=f"t{n}",
        source_name="e", source_domain="e.com", image_url=None, language="en",
        source_country="US", geo="us", category=topic, category_confidence=0.9,
        category_source="model", published_at=None, first_seen_at=NOW - age, embedding=v,
        embedding_model="m", text_source="title", title_fingerprint=f"fp{n}",
        story_cluster_id=cluster or aid, origin="gdelt",
    )  # fmt: skip


async def db(dsn: str) -> Any:
    conn = await connect(dsn)
    await apply_migrations(conn, MIGRATIONS)
    await store.register_codecs(conn)
    return conn


async def snapshot(conn: Any) -> window.WindowSnapshot:
    return (await window.sync(conn, None, NOW, MODEL)).snapshot


async def test_fresh_and_topic_pools_order_caps_and_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await skip_or_fail_if_unreachable_async()
    monkeypatch.setattr(pools, "TOPIC_CAP", 2)
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            rows = [row(i, "sports", timedelta(hours=i)) for i in range(4)]
            rows += [row(10, "health", timedelta(hours=30)), row(11, "health", timedelta(hours=80))]
            await store.insert_articles(conn, rows)
            win = await snapshot(conn)
            fresh = pools.fresh_members(win, NOW)
            assert list(fresh) == [r.id for r in rows[:4]]  # 24 h, newest first
            topics = pools.topic_members(win, NOW)
            assert list(topics["sports"]) == [r.id for r in rows[:2]]  # cap 2
            assert list(topics["health"]) == [rows[4].id]  # 72 h window
            assert topics["weather"] == {}
            # Identical to the pre-P3 SQL builders (the replay checks this every run).
            assert pools.same_pool(fresh, await pools.legacy_fresh(conn, NOW), pools.FRESH_CAP)
            legacy = await pools.legacy_topics(conn, NOW)
            assert all(pools.same_pool(topics[t], legacy[t], 2) for t in topics)
        finally:
            await conn.close()


def test_same_pool_tolerates_only_ties_at_the_cap() -> None:
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert pools.same_pool({a: 2.0, b: 1.0}, {a: 2.0, c: 1.0}, cap=2)  # tie at the edge
    assert not pools.same_pool({a: 2.0, b: 1.0}, {a: 2.0, c: 1.0}, cap=3)  # not at the cap
    assert not pools.same_pool({a: 2.0, b: 1.0}, {c: 2.0, b: 1.0}, cap=2)  # above the edge


async def test_trending_breadth_decay_and_minimum() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            wide, single, old_wide = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            first = row(1, "world", timedelta(hours=3), wide)
            later = row(2, "world", timedelta(hours=1), wide)
            rows = [first, later, row(3, "us", timedelta(0), single)]
            rows.append(row(4, "us", timedelta(hours=20), old_wide))
            await store.insert_articles(conn, rows)
            cov = [(wide, f"w{i}.com", NOW - timedelta(hours=3)) for i in range(5)]
            cov += [(single, "only.com", NOW)]  # 1 outlet: excluded
            cov += [(old_wide, f"o{i}.com", NOW - timedelta(hours=20)) for i in range(5)]
            await store.insert_coverage(conn, cov)
            snap = await pools.build_snapshot(conn, await snapshot(conn), NOW, "v1")
            assert [t.clusters for t in snap.trending] == [(wide,), (old_wide,)]  # 3 h > 20 h
            assert snap.trending[0].id == first.id  # representative = earliest stored article
            assert snap.trending[0].score == pytest.approx(5 * 2 ** (-3 / 6), rel=1e-6)
            assert snap.trending[0].outlets == 5
            assert snap.member_ids() <= set(snap.articles)
            assert set(snap.centroids) == {"world", "us"}
            assert snap.stats["articles_24h"] == 4
        finally:
            await conn.close()


class Recorder:
    """A fake Redis pipeline target that records commands in order."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.log: list[tuple[str, ...]] = []
        self.data: dict[str, Any] = {}
        self.fail_on = fail_on

    def pipeline(self, transaction: bool = False) -> "Recorder.Pipe":
        assert transaction is False
        return Recorder.Pipe(self)

    class Pipe:
        def __init__(self, r: "Recorder") -> None:
            self.r, self.cmds = r, []  # type: ignore[var-annotated]

        def set(self, k: str, v: Any) -> None:
            self.cmds.append(("set", k, v))

        def expire(self, k: str, s: int) -> None:
            self.cmds.append(("expire", k, s))

        async def execute(self) -> None:
            for c in self.cmds:
                if self.r.fail_on and c[0] == "set" and c[1] == self.r.fail_on:
                    raise ConnectionError("lost")
                self.r.log.append(tuple(str(x) for x in c[:2]))
                if c[0] == "set":
                    self.r.data[c[1]] = c[2]


async def test_publish_order_three_commands_and_grace_period() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(conn, [row(1, "sports", timedelta(hours=1))])
            win = await snapshot(conn)
            redis = Recorder()
            first = await pools.publish(
                redis, conn, await pools.build_snapshot(conn, win, NOW, "a")
            )
            assert first.commands == 2  # nothing to expire yet
            second = await pools.publish(
                redis, conn, await pools.build_snapshot(conn, win, NOW, "b")
            )
            assert second.commands == 3
            assert redis.log[-3:] == [
                ("set", "pools:blob:b"),
                ("set", "pools:ver"),
                ("expire", "pools:blob:a"),
            ]
            decoded = core_pools.decode(redis.data["pools:blob:b"], redis.data["pools:ver"])
            assert decoded.version == "b"
        finally:
            await conn.close()


async def test_a_failed_publish_keeps_the_old_version_and_the_next_run_expires_the_orphan() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(conn, [row(1, "sports", timedelta(hours=1))])
            win = await snapshot(conn)
            redis = Recorder()
            await pools.publish(redis, conn, await pools.build_snapshot(conn, win, NOW, "a"))
            redis.fail_on = "pools:ver"  # blob b is written, the version never moves
            with pytest.raises(ConnectionError):
                await pools.publish(redis, conn, await pools.build_snapshot(conn, win, NOW, "b"))
            assert redis.data["pools:ver"] == "a"  # readers keep the old snapshot
            redis.fail_on = None
            c = await pools.publish(redis, conn, await pools.build_snapshot(conn, win, NOW, "c"))
            assert c.orphan_expired
            assert c.commands == 4
            assert ("expire", "pools:blob:b") in redis.log  # orphan
            assert redis.log[-1] == ("expire", "pools:blob:a")  # previous, after the switch
        finally:
            await conn.close()


async def test_drop_legacy_is_one_delete(redis_test: Any) -> None:
    await redis_test.zadd("pool:fresh", {"x": 1.0})
    await redis_test.zadd("pool:topic:sports", {"x": 1.0})
    assert await pools.drop_legacy(redis_test) == 2
    assert not await redis_test.exists("pool:fresh", "pool:topic:sports")


async def test_publish_against_real_redis(redis_test: Any) -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(conn, [row(1, "sports", timedelta(hours=1))])
            win = await snapshot(conn)
            await pools.publish(redis_test, conn, await pools.build_snapshot(conn, win, NOW, "a"))
            await pools.publish(redis_test, conn, await pools.build_snapshot(conn, win, NOW, "b"))
            assert (await redis_test.get("pools:ver")) == b"b"
            assert 0 < await redis_test.ttl("pools:blob:a") <= 120  # grace period
            assert await redis_test.ttl("pools:blob:b") == -1  # current blob: no TTL
        finally:
            await conn.close()
