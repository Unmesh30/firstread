"""The feed against real Postgres + Redis and the real pinned encoder (ADR 0017).

Articles are seeded around three topic directions; the pool snapshot is published the
way ingestion does it. Deadlines are generous here except where a test is about one.
"""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from redis.asyncio import Redis

from firstread_api import feed as feed_mod
from firstread_api import redis_state as rs
from firstread_api import repo
from firstread_api.candidates import Exclusions, ann
from firstread_api.deps import Services
from firstread_api.encoder import MicroBatcher
from firstread_api.redis_state import CountingRedis
from firstread_api.slate import Cursor, Slate
from firstread_api.testing import (
    LiveApi,
    Seeded,
    insert_article,
    live_api,
    publish_pools,
    topic_vectors,
)
from firstread_core.dbtest import skip_or_fail_if_unreachable_async

pytestmark = pytest.mark.integration

TOPICS = ["sports", "health", "world"]
GENEROUS = {
    "api_load_model": True,
    "api_feed_timeout_s": 10.0,
    "api_state_timeout_s": 2.0,
    "api_encode_timeout_s": 2.0,
    "api_candidates_timeout_s": 2.0,
    "api_hydrate_timeout_s": 2.0,
    "api_persist_timeout_s": 2.0,
}


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[LiveApi]:
    await skip_or_fail_if_unreachable_async()
    async with live_api(tmp_path, **GENEROUS) as a:
        yield a


def svc(api: LiveApi) -> Services:
    s: Services = api.app.state.services
    return s


async def seed(api: LiveApi, per_topic: int = 40, publish: bool = True) -> list[Seeded]:
    vecs = topic_vectors(TOPICS, per_topic)
    conn = await api.db()
    try:
        out = []
        for t in TOPICS:
            for k, v in enumerate(vecs[t]):
                out.append(await insert_article(
                    conn, v, category=t, domain=f"{t}{k % 4}.example", age_hours=1 + k * 0.5
                ))  # fmt: skip
    finally:
        await conn.close()
    if publish:
        await publish_pools(api.redis, out)
    return out


async def add(api: LiveApi, vector: np.ndarray, **kw: Any) -> Seeded:
    conn = await api.db()
    try:
        return await insert_article(conn, vector, **kw)
    finally:
        await conn.close()


async def onboard(api: LiveApi, uid: uuid.UUID, topics: Sequence[str] = TOPICS) -> None:
    r = await api.client.put("/v1/me/topics", json={"topics": list(topics)}, headers=api.auth(uid))
    assert r.status_code == 200, r.text


async def page(
    api: LiveApi, uid: uuid.UUID, cursor: str | None = None, limit: int = 20, status: int = 200
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    r = await api.client.get("/v1/feed", params=params, headers=api.auth(uid))
    assert r.status_code == status, r.text
    body: dict[str, Any] = r.json()
    body["_headers"] = dict(r.headers)
    return body


def ids(body: dict[str, Any]) -> list[str]:
    return [i["id"] for i in body["items"]]


async def events(api: LiveApi, uid: uuid.UUID, *evs: dict[str, Any]) -> None:
    r = await api.client.post("/v1/events", json={"events": list(evs)}, headers=api.auth(uid))
    assert r.status_code == 200, r.text
    assert len(r.json()["accepted"]) == len(evs), r.text


def ev(kind: str, article: Any, **kw: Any) -> dict[str, Any]:
    return {"event_id": str(uuid.uuid4()), "type": kind, "article_id": str(article), **kw}


async def stored_slate(api: LiveApi, uid: uuid.UUID) -> Slate:
    raw = await api.redis.get(rs.slate_key(uid))
    assert raw is not None
    return Slate.loads(raw)


# ---- the end-to-end flow ----


async def test_anonymous_reader_topics_feed_click_next_page_delete(api: LiveApi) -> None:
    arts = await seed(api)
    by_id = {str(a.id): a for a in arts}
    uid = uuid.uuid4()
    await onboard(api, uid)

    first = await page(api, uid)
    assert first["degraded"] == []
    assert len(first["items"]) == 20
    assert first["model_version"] == "v0.2.0+rank-2"
    assert all(i["reasons"] for i in first["items"])
    assert {by_id[i].category for i in ids(first)} == set(TOPICS)  # topics, cold start
    timing = first["_headers"]["server-timing"]
    for stage in ("auth", "state", "encode", "cand", "hydrate", "rank", "persist"):
        assert f"{stage};dur=" in timing

    # What page 2 would have been without the click.
    before = await stored_slate(api, uid)
    unclicked_next = [i.id for i in before.remaining()[:20]]
    clicked = by_id[ids(first)[0]]
    await events(api, uid, ev("click", clicked.id, position=0))

    second = await page(api, uid, first["next_cursor"])
    assert second["degraded"] == []
    assert not set(ids(second)) & set(ids(first))  # never repeats
    assert ids(second) != unclicked_next  # the click re-ranked the rest
    after = await stored_slate(api, uid)
    assert after.id == before.id
    assert after.hist_version == 1
    vec = await api.redis.hget(rs.user_key(uid), "vec")  # the encoder ran and was cached
    assert vec is not None
    assert len(vec) == 384 * 4

    conn = await api.db()
    try:
        pages = await conn.fetch(
            "select page, article_ids, model_version from public.served_slates "
            "where user_id = $1 order by page", uid,
        )  # fmt: skip
        assert [p["page"] for p in pages] == [0, 1]
        assert [str(x) for x in pages[0]["article_ids"]] == ids(first)
        r = await api.client.delete("/v1/me", headers=api.auth(uid))
        assert r.status_code == 204
        n = await conn.fetchval("select count(*) from public.served_slates where user_id = $1", uid)
        assert n == 0
    finally:
        await conn.close()
    assert await api.redis.exists(*rs.all_keys(uid)) == 0


# ---- pagination ----


async def test_five_pages_never_repeat_and_a_repeat_request_returns_the_same_page(
    api: LiveApi,
) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    seen: list[str] = []
    cursor = None
    cursors = []
    for _ in range(5):
        body = await page(api, uid, cursor, limit=10)
        assert len(body["items"]) == 10
        seen += ids(body)
        cursors.append(body["next_cursor"])
        cursor = body["next_cursor"]
    assert len(seen) == len(set(seen)) == 50
    # Page 3 asked again within 60 s: the same items, not the ones after page 5.
    again = await page(api, uid, cursors[1], limit=10)
    assert ids(again) == seen[20:30]


async def test_cursors_are_opaque_and_validated(api: LiveApi) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    body = await page(api, uid)
    token = body["next_cursor"]
    assert "slate" not in token
    assert "{" not in token
    for bad in ("abc", token[:-3] + "xyz", "e30", base64.urlsafe_b64encode(
        json.dumps({"s": "x", "o": 1, "v": 1}).encode()).decode()):  # fmt: skip
        r = await api.client.get("/v1/feed", params={"cursor": bad}, headers=api.auth(uid))
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_cursor"
    unknown = Cursor(str(uuid.uuid4()), 20).encode()
    r = await api.client.get("/v1/feed", params={"cursor": unknown}, headers=api.auth(uid))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "cursor_expired"
    # Another reader can't page through this reader's slate.
    other = uuid.uuid4()
    await onboard(api, other)
    r = await api.client.get("/v1/feed", params={"cursor": token}, headers=api.auth(other))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "cursor_expired"


# ---- filters ----


async def test_clicked_hidden_hidden_source_seen_and_read_stories_are_excluded(
    api: LiveApi,
) -> None:
    arts = await seed(api)
    by_id = {str(a.id): a for a in arts}
    uid = uuid.uuid4()
    await onboard(api, uid)
    top = ids(await page(api, uid, limit=50))
    clicked, hidden, from_source, viewed = (by_id[top[k]] for k in range(4))
    # Same story as the clicked / hidden articles (their cluster), and a near-identical
    # copy of the clicked one under another cluster id (cos ≥ τ_story).
    mate_clicked = await add(
        api, clicked.vector, category=clicked.category, cluster=clicked.cluster
    )
    mate_hidden = await add(api, hidden.vector, category=hidden.category, cluster=hidden.cluster)
    noise = np.random.default_rng(7).normal(size=384).astype(np.float32)
    twin = clicked.vector + 0.05 * noise / np.linalg.norm(noise)
    twin_read = await add(api, twin / np.linalg.norm(twin), category=clicked.category,
                          domain="twin.example")  # fmt: skip
    await publish_pools(api.redis, [*arts, mate_clicked, mate_hidden, twin_read], version="t-2")
    svc(api).pools.checked_at = -1e18  # pick up the new snapshot now
    await events(
        api, uid,
        ev("click", clicked.id), ev("hide", hidden.id),
        ev("hide", from_source.id, scope="source"), ev("impression", viewed.id),
    )  # fmt: skip
    got: set[str] = set()
    for _ in range(3):  # several fresh first pages: all of the feed's top
        body = await page(api, uid, limit=50)
        got |= set(ids(body))
    banned = {clicked.id, hidden.id, from_source.id, viewed.id, mate_clicked.id, mate_hidden.id,
              twin_read.id}  # fmt: skip
    assert not {str(b) for b in banned} & got
    assert from_source.domain not in {by_id[i].domain for i in got if i in by_id}


async def story_with_copies(api: LiveApi, arts: list[Seeded], n: int = 4) -> set[str]:
    """``n`` more articles of the feed's top story (other outlets, ≈ the same vector)."""
    uid = uuid.uuid4()
    await onboard(api, uid)
    lead = ids(await page(api, uid, limit=1))[0]
    star = next(a for a in arts if str(a.id) == lead)
    rng = np.random.default_rng(3)
    copies = []
    for k in range(n):
        e = rng.normal(size=384).astype(np.float32)
        v = star.vector + 0.02 * e / np.linalg.norm(e)
        copies.append(await add(api, v / np.linalg.norm(v), category=star.category,
                                domain=f"copy{k}.example", age_hours=0.5))  # fmt: skip
    await publish_pools(api.redis, [*arts, *copies], version="t-3")
    svc(api).pools.checked_at = -1e18
    return {str(star.id), *(str(c.id) for c in copies)}


async def test_at_most_two_items_per_story(api: LiveApi) -> None:
    story = await story_with_copies(api, await seed(api))
    uid = uuid.uuid4()
    await onboard(api, uid)
    await page(api, uid, limit=50)
    slate = await stored_slate(api, uid)
    assert len([i for i in slate.items if i.id in story]) == 2


async def test_the_story_cap_holds_across_pages_and_re_ranks(api: LiveApi) -> None:
    story = await story_with_copies(api, await seed(api))
    uid = uuid.uuid4()
    await onboard(api, uid)
    shown: list[str] = []
    cursor = None
    for _ in range(4):  # each page follows a click, so each one re-ranks
        body = await page(api, uid, cursor, limit=5)
        shown += ids(body)
        other = next(i for i in ids(body) if i not in story)
        await events(api, uid, ev("click", other))
        cursor = body["next_cursor"]
    assert len([i for i in shown if i in story]) == 2


async def test_ann_rows_carry_first_seen_cluster_and_domain_outside_the_pools(
    api: LiveApi,
) -> None:
    arts = await seed(api, publish=False)
    a = arts[5]
    async with svc(api).pool.acquire() as conn:
        rows = await ann(repo.Q(conn), a.vector, a.first_seen.replace(year=2000), Exclusions(),
                         limit=3, mode="exact")  # fmt: skip
        hnsw = await ann(repo.Q(conn), a.vector, a.first_seen.replace(year=2000), Exclusions(),
                         limit=3, mode="hnsw")  # fmt: skip
    assert rows[0].id == a.id
    assert hnsw[0].id == a.id
    assert rows[0].cluster == a.cluster
    assert rows[0].source_domain == a.domain
    assert rows[0].first_seen_at == a.first_seen
    assert rows[0].score == pytest.approx(1.0, abs=2e-3)  # halfvec rounding


# ---- degraded modes: each row of the stage table ----


async def test_redis_down_serves_from_postgres_and_says_so(api: LiveApi) -> None:
    arts = await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    first = await page(api, uid)
    await events(api, uid, ev("click", ids(first)[0]), ev("hide", ids(first)[1]))
    dead = CountingRedis(Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2))
    live = svc(api).redis
    svc(api).redis = dead
    try:
        second = await page(api, uid, first["next_cursor"])
        assert second["degraded"] == ["redis"]
        assert len(second["items"]) == 20
        assert not set(ids(second)) & set(ids(first))  # served ids from served_slates
        third = await page(api, uid, second["next_cursor"])
        assert not set(ids(third)) & (set(ids(first)) | set(ids(second)))
        fresh = await page(api, uid, limit=50)  # a first page: history + hides from Postgres
        assert fresh["degraded"] == ["redis"]
        assert not {ids(first)[0], ids(first)[1]} & set(ids(fresh))
    finally:
        svc(api).redis = live
        await dead.aclose()
    assert len(arts) == 120


class FailingEncoder:
    def __init__(self, population: np.ndarray, delay: float = 0.0) -> None:
        self.population = population
        self.tag = "v0.2.0"
        self.delay = delay

    def run(self, hist: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.delay:
            import time

            time.sleep(self.delay)
            return np.zeros((len(hist), 384), np.float32)
        raise RuntimeError("onnx exploded")


@pytest.mark.parametrize("delay", [0.0, 0.3], ids=["error", "timeout"])
async def test_encoder_failure_falls_back_to_the_mean_of_history(
    api: LiveApi, delay: float
) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    first = await page(api, uid)
    await events(api, uid, ev("click", ids(first)[0]))
    real = svc(api).batcher
    assert real is not None
    broken = MicroBatcher(FailingEncoder(real.encoder.population, delay))  # type: ignore[arg-type]
    svc(api).batcher = broken
    svc(api).settings.api_encode_timeout_s = 0.05
    try:
        body = await page(api, uid)
    finally:
        svc(api).batcher = real
        await broken.close()
    assert body["degraded"] == ["encoder"]
    assert len(body["items"]) == 20


async def test_ann_failure_keeps_the_pools(api: LiveApi, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)

    async def broken(*a: Any, **kw: Any) -> Any:
        raise ConnectionError("ann down")

    monkeypatch.setattr(feed_mod, "ann", broken)
    body = await page(api, uid)
    assert body["degraded"] == ["ann"]
    assert len(body["items"]) == 20


async def test_a_cold_instance_without_a_snapshot_serves_ann_only(api: LiveApi) -> None:
    await seed(api, publish=False)
    uid = uuid.uuid4()
    await onboard(api, uid)
    body = await page(api, uid)
    assert body["degraded"] == ["pools"]
    assert len(body["items"]) == 20


async def test_no_candidate_source_is_a_503(api: LiveApi, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(api, publish=False)
    uid = uuid.uuid4()
    await onboard(api, uid)

    async def broken(*a: Any, **kw: Any) -> Any:
        raise ConnectionError("ann down")

    monkeypatch.setattr(feed_mod, "ann", broken)
    body = await page(api, uid, status=503)
    assert body["error"]["code"] == "unavailable"
    assert body["_headers"]["retry-after"] == "5"
    assert "cand;dur=" in body["_headers"]["server-timing"]


async def test_hydrate_failure_is_a_503(api: LiveApi, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)

    async def broken(*a: Any, **kw: Any) -> Any:
        raise ConnectionError("db down")

    monkeypatch.setattr(svc(api).articles, "get_many", broken)
    await page(api, uid, status=503)


async def test_persist_failure_is_a_503(api: LiveApi, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    real = repo.Q.execute

    async def execute(self: repo.Q, name: str, sql: str, *args: Any) -> str:
        if name == "feed_served_insert":
            raise ConnectionError("db down")
        return await real(self, name, sql, *args)

    monkeypatch.setattr(repo.Q, "execute", execute)
    await page(api, uid, status=503)
    assert await api.redis.exists(rs.slate_key(uid)) == 0  # nothing half-saved for paging


async def test_the_whole_request_has_a_deadline(
    api: LiveApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    real = feed_mod.ann  # type: ignore[attr-defined]

    async def slow(*a: Any, **kw: Any) -> Any:
        await asyncio.sleep(0.5)
        return await real(*a, **kw)

    monkeypatch.setattr(feed_mod, "ann", slow)
    svc(api).settings.api_feed_timeout_s = 0.2
    body = await page(api, uid, status=503)
    assert body["error"]["code"] == "unavailable"


async def test_stage_deadlines_degrade_instead_of_waiting(
    api: LiveApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    real = feed_mod.ann  # type: ignore[attr-defined]

    async def slow(*a: Any, **kw: Any) -> Any:
        await asyncio.sleep(0.5)
        return await real(*a, **kw)

    monkeypatch.setattr(feed_mod, "ann", slow)
    svc(api).settings.api_candidates_timeout_s = 0.1
    body = await page(api, uid)
    assert body["degraded"] == ["ann"]


# ---- metrics ----


async def test_metrics_use_route_templates_and_record_stages(api: LiveApi) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    await page(api, uid)
    text = (await api.client.get("/metrics")).text
    assert 'http_requests_total{method="GET",route="/v1/feed",status="200"}' in text
    for stage in ("state", "encode", "cand", "hydrate", "rank", "persist"):
        assert f'feed_stage_seconds_count{{stage="{stage}"}}' in text
    assert "encoder_batch_size_count" in text
    assert 'pools_refresh_total{result="loaded"}' in text


# ---- budgets (ADR 0011): Upstash bills every command, pipelined or not ----


def redis_commands() -> float:
    from firstread_api.metrics import REDIS_COMMANDS

    return sum(s.value for m in REDIS_COMMANDS.collect() for s in m.samples
               if s.name.endswith("_total"))  # fmt: skip


async def test_redis_commands_per_page(api: LiveApi) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    await page(api, uid)  # warms the pool snapshot (GET ver + GET blob, once a minute)
    c0 = redis_commands()
    first = await page(api, uid)  # 4 reads + SET slate (no clicks: no vector to cache)
    c1 = redis_commands()
    second = await page(api, uid, first["next_cursor"])  # from the slate: GET, SMEMBERS, SET
    c2 = redis_commands()
    await page(api, uid, first["next_cursor"])  # the page cache: GET, SMEMBERS
    c3 = redis_commands()
    await events(api, uid, ev("click", ids(second)[0]))
    c4 = redis_commands()
    await page(api, uid, second["next_cursor"])  # re-rank: 2 + 3 reads, SET, HSET + EXPIRE
    c5 = redis_commands()
    assert (c1 - c0, c2 - c1, c3 - c2, c5 - c4) == (5, 3, 2, 8)


# ---- PR #20 review: lost slate writes, hanging Redis ----


class Flaky:
    """Wraps the API's Redis: the first pipeline whose queued commands match ``when``
    fails on execute; everything else goes through."""

    def __init__(self, inner: CountingRedis, when: Any) -> None:
        self.inner, self.when, self.fired = inner, when, False

    def pipeline(self, transaction: bool = False) -> Any:
        pipe = self.inner.pipeline(transaction)
        run = pipe.execute

        async def execute() -> list[Any]:
            if not self.fired and self.when(list(pipe._names)):
                self.fired = True
                pipe._names = []
                raise ConnectionError("flaky redis")
            return await run()

        pipe.execute = execute  # type: ignore[method-assign]
        return pipe

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.mark.parametrize(
    "when",
    [
        pytest.param(lambda names: names == ["get", "smembers"], id="light-read-fails"),
        pytest.param(lambda names: "set" in names, id="slate-write-fails"),
    ],
)
async def test_a_lost_slate_write_never_repeats_a_page(api: LiveApi, when: Any) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    first = await page(api, uid, limit=10)
    live = svc(api).redis
    flaky = Flaky(live, when)
    svc(api).redis = flaky  # type: ignore[assignment]
    try:
        second = await page(api, uid, first["next_cursor"], limit=10)
        assert flaky.fired
        assert second["degraded"] == ["redis"]
        third = await page(api, uid, second["next_cursor"], limit=10)
        fourth = await page(api, uid, third["next_cursor"], limit=10)
    finally:
        svc(api).redis = live
    shown = [*ids(first), *ids(second), *ids(third), *ids(fourth)]
    assert len(shown) == len(set(shown)) == 40


class HangingRedis:
    async def get(self, key: str) -> Any:
        await asyncio.sleep(30)


async def test_a_hanging_redis_leaves_a_cold_instance_ann_only_inside_the_deadline(
    api: LiveApi,
) -> None:
    import time

    from firstread_api.pools_cache import PoolsCache

    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    svc(api).pools = PoolsCache(HangingRedis(), 60, refresh_timeout_s=0.3)  # cold
    svc(api).settings.api_feed_timeout_s = 1.0
    t0 = time.perf_counter()
    body = await page(api, uid)
    elapsed = time.perf_counter() - t0
    assert body["degraded"] == ["pools"]
    assert len(body["items"]) == 20
    assert elapsed < 0.8
    assert svc(api).pools._refreshing is None  # the in-flight refresh was cleared


async def test_topics_save_with_redis_down(api: LiveApi) -> None:
    from firstread_api.main import redis_client
    from firstread_api.testing import make_settings

    uid = uuid.uuid4()
    live = svc(api).redis
    dead = CountingRedis(redis_client(make_settings(redis_url="redis://127.0.0.1:1/0")))
    svc(api).redis = dead
    try:
        await onboard(api, uid, ["sports", "health", "travel"])
    finally:
        svc(api).redis = live
        await dead.aclose()
    me = (await api.client.get("/v1/me", headers=api.auth(uid))).json()
    assert me["topics"] == ["sports", "health", "travel"]


async def test_delete_me_with_redis_down_is_a_503_and_a_retry_finishes_it(api: LiveApi) -> None:
    from firstread_api.main import redis_client
    from firstread_api.testing import make_settings

    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    first = await page(api, uid)
    await events(api, uid, ev("click", ids(first)[0]))
    await page(api, uid, first["next_cursor"])  # slate, vector cache, history, hidden set
    await api.redis.sadd(rs.seen_key(uid), ids(first)[1].encode())  # an impression
    assert await api.redis.exists(*rs.all_keys(uid)) == 5

    live = svc(api).redis
    dead = CountingRedis(redis_client(make_settings(redis_url="redis://127.0.0.1:1/0")))
    svc(api).redis = dead
    try:
        r = await api.client.delete("/v1/me", headers=api.auth(uid))
        assert r.status_code == 503
        assert r.headers["retry-after"] == "5"
        assert r.json()["error"]["code"] == "unavailable"
        assert r.json()["request_id"]
    finally:
        svc(api).redis = live
        await dead.aclose()
    conn = await api.db()
    try:
        for table in ("profiles", "user_history", "events", "served_slates"):
            n = await conn.fetchval(
                f"select count(*) from public.{table} where user_id = $1",  # noqa: S608
                uid,
            )
            assert n == 0, table
    finally:
        await conn.close()
    # The instance forgot the profile: the feed recreates it instead of 503ing.
    assert (await page(api, uid))["items"]
    r = await api.client.delete("/v1/me", headers=api.auth(uid))
    assert r.status_code == 204
    assert await api.redis.exists(*rs.all_keys(uid)) == 0
