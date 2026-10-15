"""Redis pools rebuilt from Postgres (real Postgres + Redis db 15)."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_core.topics import TOPIC_IDS
from firstread_ingest import pools, store

NOW = datetime(2026, 10, 15, 12, 20, tzinfo=UTC)  # minute 20: not the hourly full rewrite
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


def test_plan_writes_fresh_and_trending_always_topics_when_changed() -> None:
    topics: dict[str, dict[str, float]] = {t: {} for t in TOPIC_IDS}
    w = pools.plan_writes({}, topics, {}, ["sports", "not-a-topic"], full=False)
    assert set(w) == {pools.FRESH, pools.TRENDING, "pool:topic:sports"}
    full = pools.plan_writes({}, topics, {}, [], full=True)
    assert len(full) == 2 + 13


async def test_pools_contents_order_caps_and_empty_pools(
    redis_test: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await skip_or_fail_if_unreachable_async()
    monkeypatch.setattr(pools, "TOPIC_CAP", 2)
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            rows = [row(i, "sports", timedelta(hours=i)) for i in range(4)]
            rows += [row(10, "health", timedelta(hours=30)), row(11, "health", timedelta(hours=80))]
            await store.insert_articles(conn, rows)
            # A stale pool for a topic that is now empty must disappear, not error.
            await redis_test.zadd("pool:topic:weather", {"old": 1.0})
            result = await pools.rebuild(redis_test, conn, NOW, ["sports", "health", "weather"])
            fresh = await redis_test.zrevrange(pools.FRESH, 0, -1)
            assert [x.decode() for x in fresh] == [
                str(r.id) for r in rows[:4]
            ]  # 24 h, newest first
            sports = await redis_test.zrevrange("pool:topic:sports", 0, -1)
            assert [x.decode() for x in sports] == [str(r.id) for r in rows[:2]]  # cap 2
            health = await redis_test.zrange("pool:topic:health", 0, -1)
            assert [x.decode() for x in health] == [str(rows[4].id)]  # 72 h window
            assert not await redis_test.exists("pool:topic:weather")  # empty -> DEL
            assert not await redis_test.exists("pool:fresh:tmp")
            # fresh (2) + empty trending (1 DEL) + sports, health (2 each) + weather DEL (1)
            assert result.commands == 8
        finally:
            await conn.close()


async def test_trending_breadth_decay_and_minimum(redis_test: Any) -> None:
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
            top = await pools.trending(conn, NOW)
            assert [t["story_cluster_id"] for t in top] == [wide, old_wide]  # decay: 3 h > 20 h
            assert top[0]["id"] == str(first.id)  # representative = earliest stored article
            assert top[0]["score"] == pytest.approx(5 * 2 ** (-3 / 6), rel=1e-6)
            assert top[0]["domains"] == 5
            await pools.rebuild(redis_test, conn, NOW, [])
            assert await redis_test.zcard(pools.TRENDING) == 2
        finally:
            await conn.close()


async def test_full_hourly_rewrite_stays_within_budget(redis_test: Any) -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(
                conn, [row(i, t, timedelta(hours=1)) for i, t in enumerate(TOPIC_IDS)]
            )
            top_of_hour = NOW.replace(minute=5)
            result = await pools.rebuild(redis_test, conn, top_of_hour, [])
            assert result.commands <= pools.MAX_COMMANDS
            for t in TOPIC_IDS:
                assert await redis_test.zcard(f"pool:topic:{t}") == 1
        finally:
            await conn.close()


async def test_budget_is_enforced(redis_test: Any) -> None:
    writes = {f"pool:x{i}": {"a": 1.0} for i in range(16)}  # 32 commands
    with pytest.raises(RuntimeError, match="budget"):
        await pools.write_pools(redis_test, writes)
