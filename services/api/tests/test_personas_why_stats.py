"""Demo personas, /why and /v1/stats against real Postgres + Redis (ADR 0018)."""

import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from firstread_api import redis_state as rs
from firstread_api.metrics import DB_QUERIES, REDIS_COMMANDS
from firstread_api.testing import (
    LiveApi,
    Seeded,
    insert_article,
    insert_persona,
    live_api,
    persona_summary,
    publish_pools,
    topic_vectors,
)
from firstread_core.dbtest import skip_or_fail_if_unreachable_async
from firstread_core.ranking import ReasonInputs, reasons

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
STATS = {
    "articles_24h": 4855, "sources_24h": 1200, "median_lag_s": 912.0, "runs_24h": 96,
    "runs_ok_24h": 94, "runs_partial_24h": 1, "last_run_finished": "2026-10-23T10:15:00+00:00",
    "last_run_status": "ok",
}  # fmt: skip


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[LiveApi]:
    await skip_or_fail_if_unreachable_async()
    async with live_api(tmp_path, **GENEROUS) as a:
        yield a


@pytest.fixture
async def bare(tmp_path: Path) -> AsyncIterator[LiveApi]:
    """No model, no snapshot published."""
    await skip_or_fail_if_unreachable_async()
    async with live_api(tmp_path) as a:
        yield a


async def seed(api: LiveApi, per_topic: int = 30) -> list[Seeded]:
    vecs = topic_vectors(TOPICS, per_topic)
    conn = await api.db()
    try:
        out = []
        for t in TOPICS:
            for k, v in enumerate(vecs[t]):
                out.append(await insert_article(
                    conn, v, category=t, domain=f"{t}{k % 4}.example", age_hours=1 + k * 0.5
                ))  # fmt: skip
        sports = [a.id for a in out if a.category == "sports"][:10]
        health = [a.id for a in out if a.category == "health"][:10]
        await insert_persona(conn, "sports-fan", ["sports"], sports)
        await insert_persona(conn, "health-nut", ["health", "world"], health)
        await conn.execute(
            "insert into public.demo_personas (id, name, description, topics, refreshed_at) "
            "values ('empty', 'Empty', 'none yet', '{world}', now())"
        )
    finally:
        await conn.close()
    personas = [persona_summary("sports-fan", ["sports"], 10),
                persona_summary("health-nut", ["health", "world"], 10)]  # fmt: skip
    await publish_pools(api.redis, out, personas=personas, stats=STATS)
    return out


def total(counter: Any) -> float:
    return float(
        sum(s.value for m in counter.collect() for s in m.samples if s.name.endswith("_total"))
    )


async def activate(
    api: LiveApi, uid: uuid.UUID, persona: str, replace: bool | None = None, status: int = 200
) -> dict[str, Any]:
    params = {} if replace is None else {"replace": str(replace).lower()}
    r = await api.client.post(
        f"/v1/demo/personas/{persona}/activate", params=params, headers=api.auth(uid)
    )
    assert r.status_code == status, r.text
    body: dict[str, Any] = r.json()
    return body


async def history(api: LiveApi, uid: uuid.UUID) -> list[Any]:
    conn = await api.db()
    try:
        return list(await conn.fetch(
            "select article_id, version, kind from public.user_history where user_id = $1 "
            "order by version", uid,
        ))  # fmt: skip
    finally:
        await conn.close()


async def post_events(api: LiveApi, uid: uuid.UUID, *evs: dict[str, Any]) -> None:
    r = await api.client.post("/v1/events", json={"events": list(evs)}, headers=api.auth(uid))
    assert r.status_code == 200, r.text


def ev(kind: str, article: Any, **kw: Any) -> dict[str, Any]:
    return {"event_id": str(uuid.uuid4()), "type": kind, "article_id": str(article), **kw}


def ids_of(body: dict[str, Any]) -> list[str]:
    return [i["id"] for i in body["items"]]


# ---- public, from the snapshot ----


async def test_personas_and_stats_come_from_the_snapshot_without_postgres(api: LiveApi) -> None:
    await seed(api)
    queries = total(DB_QUERIES)
    r = await api.client.get("/v1/demo/personas")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=60"
    body = r.json()
    assert [p["id"] for p in body["personas"]] == ["sports-fan", "health-nut"]
    assert body["personas"][0]["samples"][0]["title"] == "A headline for sports-fan"
    assert body["model_version"] == "v0.2.0+rank-3"
    r = await api.client.get("/v1/stats")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=60"
    stats = r.json()
    assert stats["credit"]["url"] == "https://www.gdeltproject.org/"
    assert "GDELT" in stats["credit"]["text"]
    assert stats["articles_24h"] == 4855
    assert stats["ingestion"]["last_run_status"] == "ok"
    assert stats["ingestion"]["success_rate_24h"] == round(95 / 96, 4)
    assert stats["personas"] == 2
    assert total(DB_QUERIES) == queries  # zero Postgres reads, zero egress


async def test_without_a_snapshot_personas_and_stats_are_503(bare: LiveApi) -> None:
    for path in ("/v1/demo/personas", "/v1/stats"):
        r = await bare.client.get(path)
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "unavailable"


# ---- activation ----


async def test_activation_copies_the_history_and_the_feed_uses_it(api: LiveApi) -> None:
    arts = await seed(api)
    by_id = {str(a.id): a for a in arts}
    uid = uuid.uuid4()
    await api.redis.set(rs.slate_key(uid), b"old")  # stale state from before
    await api.redis.rpush(rs.hist_key(uid), b"old")
    me = await activate(api, uid, "sports-fan")
    assert me["active_persona"] == "sports-fan"
    assert me["topics"] == ["sports"]
    assert me["history_count"] == 10
    rows = await history(api, uid)
    assert [r["version"] for r in rows] == list(range(1, 11))  # contiguous
    assert {r["kind"] for r in rows} == {"persona"}
    persona_ids = {str(r["article_id"]) for r in rows}
    assert await api.redis.exists(rs.slate_key(uid), rs.hist_key(uid)) == 0
    r = await api.client.get("/v1/feed", params={"limit": 20}, headers=api.auth(uid))
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] == []
    assert not set(ids_of(body)) & persona_ids  # history is never recommended back
    cats = [by_id[i].category for i in ids_of(body)]
    assert cats.count("sports") > len(cats) / 2
    # The history was rebuilt into Redis from Postgres: 10 entries and a marker.
    assert await api.redis.llen(rs.hist_key(uid)) == 11


async def test_a_reader_with_their_own_history_or_topics_needs_replace(api: LiveApi) -> None:
    arts = await seed(api)
    with_topics, with_click, fresh = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    r = await api.client.put(
        "/v1/me/topics", json={"topics": TOPICS}, headers=api.auth(with_topics)
    )
    assert r.status_code == 200
    await post_events(api, with_click, ev("click", arts[40].id))
    for uid in (with_topics, with_click):
        err = await activate(api, uid, "sports-fan", status=409)
        assert err["error"]["code"] == "history_exists"
        err = await activate(api, uid, "sports-fan", replace=False, status=409)
        me = await activate(api, uid, "sports-fan", replace=True)
        assert me["active_persona"] == "sports-fan"
        assert me["topics"] == ["sports"]
        assert me["history_count"] == 10
    await activate(api, fresh, "sports-fan")  # a brand-new reader needs no flag


async def test_switching_personas_needs_no_flag_and_keeps_hides(api: LiveApi) -> None:
    arts = await seed(api)
    uid = uuid.uuid4()
    await activate(api, uid, "sports-fan")
    before = await history(api, uid)
    world = next(a for a in arts if a.category == "world")
    await post_events(
        api, uid, ev("hide", world.id, scope="source"), ev("hide", arts[1].id),
        ev("click", arts[41].id),
    )  # fmt: skip
    me = await activate(api, uid, "health-nut")  # own click + a persona: still no flag
    assert me["active_persona"] == "health-nut"
    assert me["topics"] == ["health", "world"]
    assert me["hidden_sources"] == [world.domain]
    after = await history(api, uid)
    assert len(after) == 10
    assert min(r["version"] for r in after) > max(r["version"] for r in before) + 1
    conn = await api.db()
    try:
        hides = await conn.fetchval(
            "select count(*) from public.events where user_id = $1 and type = 'hide'", uid
        )
    finally:
        await conn.close()
    assert hides == 2


async def test_unknown_empty_and_malformed_personas(api: LiveApi) -> None:
    await seed(api)
    uid = uuid.uuid4()
    assert (await activate(api, uid, "nobody", status=404))["error"]["code"] == "not_found"
    err = await activate(api, uid, "empty", status=503)
    assert err["error"]["code"] == "unavailable"
    await activate(api, uid, "Bad_Id", status=422)


# ---- /why ----


async def onboard(api: LiveApi, uid: uuid.UUID, topics: Sequence[str] = TOPICS) -> None:
    r = await api.client.put("/v1/me/topics", json={"topics": list(topics)}, headers=api.auth(uid))
    assert r.status_code == 200


async def why(api: LiveApi, uid: uuid.UUID, article: Any, status: int = 200) -> dict[str, Any]:
    r = await api.client.get(f"/v1/articles/{article}/why", headers=api.auth(uid))
    assert r.status_code == status, r.text
    body: dict[str, Any] = r.json()
    return body


async def test_why_is_computed_for_the_caller_only(api: LiveApi) -> None:
    arts = await seed(api)
    sports = [a for a in arts if a.category == "sports"]
    a_reader, b_reader = uuid.uuid4(), uuid.uuid4()
    await activate(api, a_reader, "sports-fan")
    await activate(api, b_reader, "health-nut")
    target = sports[20]
    a_why = await why(api, a_reader, target.id)
    b_why = await why(api, b_reader, target.id)
    a_hist = {str(r["article_id"]) for r in await history(api, a_reader)}
    b_hist = {str(r["article_id"]) for r in await history(api, b_reader)}
    assert {s["article_id"] for s in a_why["similar_reads"]} <= a_hist
    assert {s["article_id"] for s in b_why["similar_reads"]} <= b_hist
    assert len(a_why["similar_reads"]) == 2
    assert a_why["followed_topic"] is True
    assert b_why["followed_topic"] is False
    assert a_why["components"]["model"] > b_why["components"]["model"]  # sports reader
    await why(api, a_reader, uuid.uuid4(), status=404)


async def test_why_matches_the_feed_and_reasons_come_from_the_components(api: LiveApi) -> None:
    await seed(api)
    uid = uuid.uuid4()
    await onboard(api, uid)
    r = await api.client.get("/v1/feed", params={"limit": 5}, headers=api.auth(uid))
    items = r.json()["items"]
    commands = total(REDIS_COMMANDS)
    for item in items:
        w = await why(api, uid, item["id"])
        assert w["degraded"] == []
        assert w["components"]["final"] == pytest.approx(item["score"], abs=2e-3)
        c = w["components"]
        expected = reasons(ReasonInputs(
            model=c["model"], fresh=c["fresh"], trend=c["trend"], age_hours=w["age_hours"],
            outlets=w["outlets"], topic=w["category"], followed=TOPICS,
            topic_weight=w["topic_weight"],
            nearest_read=w["similar_reads"][0]["title"] if w["similar_reads"] else None,
        ))  # fmt: skip
        assert [[x["code"], x["text"]] for x in w["reasons"]] == [
            [x.code, x.text] for x in expected
        ]
        feed_codes = {x["code"] for x in item["reasons"]} - {"diversity"}
        assert feed_codes <= {x["code"] for x in w["reasons"]} | {"diversity"}
    # 2 Redis commands per /why (the user vector cache + the history), nothing else.
    assert total(REDIS_COMMANDS) - commands == 2 * len(items)
