"""Profiles, events and the Redis history against real Postgres + Redis (ADR 0016)."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from firstread_api import redis_state as rs
from firstread_api import repo
from firstread_api.testing import LiveApi, insert_articles, live_api
from firstread_core.dbtest import skip_or_fail_if_unreachable_async

pytestmark = pytest.mark.integration


@pytest.fixture
async def api(tmp_path: Path) -> Any:
    await skip_or_fail_if_unreachable_async()
    async with live_api(tmp_path) as a:
        yield a


def ev(kind: str, article: uuid.UUID | None, **kw: Any) -> dict[str, Any]:
    return {"event_id": str(uuid.uuid4()), "type": kind,
            "article_id": str(article) if article else None, **kw}  # fmt: skip


async def articles(api: LiveApi, n: int, **kw: Any) -> list[uuid.UUID]:
    conn = await api.db()
    try:
        return await insert_articles(conn, n, **kw)
    finally:
        await conn.close()


async def redis_history(api: LiveApi, uid: uuid.UUID) -> list[rs.HistoryEntry]:
    return [rs.unpack(x) for x in await api.redis.lrange(rs.hist_key(uid), 0, -1)]


async def hist_version(api: LiveApi, uid: uuid.UUID) -> int:
    conn = await api.db()
    try:
        return int(await conn.fetchval(
            "select hist_version from public.profiles where user_id = $1", uid
        ))  # fmt: skip
    finally:
        await conn.close()


async def test_topics_me_and_delete(api: LiveApi) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    r = await api.client.put("/v1/me/topics", json={"topics": ["sports", "health", "world"]},
                             headers=h)  # fmt: skip
    assert r.status_code == 200, r.text
    me = (await api.client.get("/v1/me", headers=h)).json()
    assert me["topics"] == ["sports", "health", "world"]
    assert me["is_anonymous"] is True
    assert me["history_count"] == 0
    [a] = await articles(api, 1)
    await api.client.post("/v1/events", json={"events": [ev("click", a)]}, headers=h)
    assert await api.redis.exists(rs.hist_key(uid))
    assert (await api.client.delete("/v1/me", headers=h)).status_code == 204
    conn = await api.db()
    try:
        for table in ("profiles", "user_history", "events", "served_slates"):
            n = await conn.fetchval(f"select count(*) from public.{table} where user_id = $1", uid)  # noqa: S608
            assert n == 0, table
    finally:
        await conn.close()
    assert not await api.redis.exists(*rs.all_keys(uid))


async def test_events_are_idempotent_and_side_effects_follow_inserts_only(api: LiveApi) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    a, b, c = await articles(api, 3)
    ghost = uuid.uuid4()  # not stored (expired, or bogus)
    batch = [ev("impression", a, position=0), ev("click", a), ev("like", b),
             ev("hide", c), ev("dwell", a, dwell_ms=4000), ev("click", ghost)]  # fmt: skip
    r = (await api.client.post("/v1/events", json={"events": batch}, headers=h)).json()
    assert len(r["accepted"]) == 5
    assert r["rejected"] == [{"event_id": batch[-1]["event_id"], "code": "unknown_article"}]
    hist = await redis_history(api, uid)
    assert [e.article_id for e in hist] == [a, b]
    assert rs.is_consistent(hist, await hist_version(api, uid))
    assert await api.redis.smembers(rs.seen_key(uid)) == {str(a).encode()}
    assert await api.redis.smembers(rs.hidden_key(uid)) == {str(c).encode()}
    # A retried batch: nothing new in Postgres, so nothing new in Redis.
    again = (await api.client.post("/v1/events", json={"events": batch}, headers=h)).json()
    assert len(again["duplicate"]) == 5
    assert again["accepted"] == []
    assert len(await redis_history(api, uid)) == 2
    assert await hist_version(api, uid) == 2


async def test_history_is_capped_at_fifty_and_stays_consistent(api: LiveApi) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    ids = await articles(api, 70)
    for start in range(0, 70, 35):
        batch = [ev("click", x) for x in ids[start : start + 35]]
        assert (
            await api.client.post("/v1/events", json={"events": batch}, headers=h)
        ).status_code == 200
    conn = await api.db()
    try:
        assert await conn.fetchval(
            "select count(*) from public.user_history where user_id = $1", uid) == 50  # fmt: skip
    finally:
        await conn.close()
    hist = await redis_history(api, uid)
    assert len(hist) == rs.HIST_KEEP_RAW
    assert rs.is_consistent(hist, await hist_version(api, uid))
    assert len(rs.latest(hist)) == 50


async def test_overlapping_batches_both_reach_redis(api: LiveApi) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    await api.client.get("/v1/me", headers=h)  # profile exists before the race
    ids = await articles(api, 8)
    first = {"events": [ev("click", x) for x in ids[:4]]}
    second = {"events": [ev("click", x) for x in ids[4:]]}
    r1, r2 = await asyncio.gather(
        api.client.post("/v1/events", json=first, headers=h),
        api.client.post("/v1/events", json=second, headers=h),
    )
    assert r1.status_code == r2.status_code == 200
    hist = await redis_history(api, uid)
    assert {e.article_id for e in hist} == set(ids)
    assert sorted(e.version for e in hist) == list(range(1, 9))
    assert rs.is_consistent(sorted(hist, key=lambda e: e.version), 8)


async def test_a_redis_failure_after_commit_is_healed_by_the_version_check(
    api: LiveApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    a, b = await articles(api, 2)
    await api.client.post("/v1/events", json={"events": [ev("click", a)]}, headers=h)
    svc = api.app.state.services

    class Broken:
        def __getattr__(self, name: str) -> Any:
            if name == "execute":

                async def fail() -> None:
                    raise ConnectionError("redis down")

                return fail
            return lambda *a, **k: self

    monkeypatch.setattr(svc.redis, "pipeline", lambda transaction=False: Broken())
    r = await api.client.post("/v1/events", json={"events": [ev("click", b)]}, headers=h)
    assert r.status_code == 200  # Postgres has it
    monkeypatch.undo()
    version = await hist_version(api, uid)
    stale = await redis_history(api, uid)
    assert not rs.is_consistent(stale, version)  # the gap is visible
    conn = await api.db()
    try:
        rows, v = await repo.history_for_rebuild(repo.Q(conn), uid)
    finally:
        await conn.close()
    await rs.rebuild_history(svc.redis, uid, rows, v)
    healed = await redis_history(api, uid)
    assert rs.is_consistent(healed, version)
    assert [e.article_id for e in rs.latest(healed)] == [a, b]


async def test_hides_survive_losing_redis(api: LiveApi) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    a, b = await articles(api, 2)
    batch = [ev("hide", a), ev("hide", b, scope="source")]
    await api.client.post("/v1/events", json={"events": batch}, headers=h)
    await api.redis.flushdb()
    me = (await api.client.get("/v1/me", headers=h)).json()
    conn = await api.db()
    try:
        domain = await conn.fetchval("select source_domain from public.articles where id = $1", b)
        hidden = await repo.hidden_for_rebuild(repo.Q(conn), uid)
    finally:
        await conn.close()
    assert me["hidden_sources"] == [domain]  # a preference in Postgres
    assert hidden == [a]  # article hides come back from events
    assert rs.hidden_from(await api.redis.smembers(rs.hidden_key(uid))) is None  # lost: rebuild
    await rs.rebuild_hidden(api.app.state.services.redis, uid, hidden)
    assert rs.hidden_from(await api.redis.smembers(rs.hidden_key(uid))) == {a}


async def test_unhide_a_source_keeps_article_hides_and_is_owner_scoped(api: LiveApi) -> None:
    uid, other = uuid.uuid4(), uuid.uuid4()
    a, b = await articles(api, 2)
    await api.client.post("/v1/events", json={"events": [ev("hide", a, scope="source"),
                                                         ev("hide", b)]},
                          headers=api.auth(uid))  # fmt: skip
    await api.client.post("/v1/events", json={"events": [ev("hide", a, scope="source")]},
                          headers=api.auth(other))  # fmt: skip
    conn = await api.db()
    try:
        domain = await conn.fetchval("select source_domain from public.articles where id = $1", a)
    finally:
        await conn.close()
    r = await api.client.delete(f"/v1/me/hidden-sources/{domain.upper()}", headers=api.auth(uid))
    assert r.status_code == 200, r.text
    assert r.json()["hidden_sources"] == []
    # Not hidden (any more): a no-op. A bad domain: 422.
    r = await api.client.delete(f"/v1/me/hidden-sources/{domain}", headers=api.auth(uid))
    assert r.status_code == 200
    r = await api.client.delete("/v1/me/hidden-sources/bad%20domain", headers=api.auth(uid))
    assert r.status_code == 422
    # The other reader's preference is untouched; the article hide stays.
    me = (await api.client.get("/v1/me", headers=api.auth(other))).json()
    assert me["hidden_sources"] == [domain]
    conn = await api.db()
    try:
        assert await repo.hidden_for_rebuild(repo.Q(conn), uid) == [b]
    finally:
        await conn.close()


async def test_last_seen_is_touched_at_most_once_a_day(api: LiveApi) -> None:
    uid = uuid.uuid4()
    h = api.auth(uid)
    await api.client.get("/v1/me", headers=h)
    conn = await api.db()
    try:
        first = await conn.fetchval(
            "select last_seen_at from public.profiles where user_id = $1", uid
        )
        await conn.execute(
            "update public.profiles set last_seen_at = now() - interval '2 days' "
            "where user_id = $1",
            uid,
        )
        api.app.state.services.ensured.clear()  # a new day for this instance
        await api.client.get("/v1/me", headers=h)
        touched = await conn.fetchval(
            "select last_seen_at from public.profiles where user_id = $1", uid
        )
        await api.client.get("/v1/me", headers=h)  # same day: no write at all
        again = await conn.fetchval(
            "select last_seen_at from public.profiles where user_id = $1", uid
        )
    finally:
        await conn.close()
    assert touched >= first
    assert again == touched


async def test_a_click_between_the_rebuild_read_and_write_is_never_lost(
    api: LiveApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race from the PR #19 review: a click commits (and reaches Redis) after the
    rebuild read its snapshot but before the rebuild's DEL. With rows and version read
    in one statement, the marker carries the old version, so the click shows up as a
    gap and the next check rebuilds again, instead of vanishing."""
    uid = uuid.uuid4()
    h = api.auth(uid)
    a, b = await articles(api, 2)
    await api.client.post("/v1/events", json={"events": [ev("click", a)]}, headers=h)
    original = repo.Q.fetch
    fired = False

    async def fetch_then_click(self: repo.Q, name: str, sql: str, *args: Any) -> list[Any]:
        nonlocal fired
        rows = await original(self, name, sql, *args)
        if name == "history_rebuild" and not fired:
            fired = True  # interleave: a full click (commit + RPUSH) right after the read
            r = await api.client.post("/v1/events", json={"events": [ev("click", b)]}, headers=h)
            assert r.status_code == 200
        return rows

    monkeypatch.setattr(repo.Q, "fetch", fetch_then_click)
    svc = api.app.state.services
    conn = await api.db()
    try:
        rows, version = await repo.history_for_rebuild(repo.Q(conn), uid)
        assert fired
        assert version == max(e.version for e in rows)  # never a version without its row
        await rs.rebuild_history(svc.redis, uid, rows, version)  # DEL wipes b's RPUSH
        now = await hist_version(api, uid)
        assert now == version + 1
        after = await redis_history(api, uid)
        assert not rs.is_consistent(after, now)  # the lost click is visible as a gap
        monkeypatch.undo()
        rows2, v2 = await repo.history_for_rebuild(repo.Q(conn), uid)
    finally:
        await conn.close()
    await rs.rebuild_history(svc.redis, uid, rows2, v2)
    healed = await redis_history(api, uid)
    assert rs.is_consistent(healed, now)
    assert [e.article_id for e in rs.latest(healed)] == [a, b]
