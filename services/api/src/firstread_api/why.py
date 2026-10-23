"""GET /v1/articles/{id}/why: one item's blend, recomputed for the caller (ADR 0018).

It doesn't read a slate: it recomputes the components the feed would give this
article for **this reader now**, with the feed's own pieces:

* the reader's state: ``HMGET u:{uid} vec vec_key`` + ``LRANGE`` history (2 Redis
  commands) ∥ the profile row; a history that fails the version check, or Redis being
  down, is read from Postgres instead (the next feed request rebuilds Redis);
* the user vector: :meth:`FeedPipeline.user_vector` (cached vector, else the batched
  encoder, then the cold-start blend);
* the article from the in-process LRU (or 1 row); trend score and outlets from the
  pool snapshot;
* the blend (``firstread_core.ranking``) and the same ``reasons()`` the feed uses.

The trend term is a percentile *within a request* (ADR 0014). The feed ranks it among
that request's candidates; here it is ranked among the reader's pool candidates
(fresh, trending and followed topics, :func:`feed.pool_picks`), the bulk of the same
set, so the value can differ slightly. The reason texts don't depend on it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

import numpy as np

from firstread_api import redis_state as rs
from firstread_api import repo
from firstread_api.articles_cache import Article
from firstread_api.deps import Services
from firstread_api.errors import ApiError
from firstread_api.feed import FeedPipeline, State, parse_vec, pool_picks, trend_map
from firstread_core.ranking import ReasonInputs, blend, freshness, reasons, trend_feature
from firstread_core.ranking.reasons import Reason

SIMILAR = 2  # nearest history items returned


@dataclass
class Explanation:
    article: Article
    components: dict[str, float]
    age_hours: float
    outlets: int
    followed: bool
    topic_weight: float
    similar: list[tuple[uuid.UUID, str, float]]
    reasons: list[Reason]
    degraded: list[str] = field(default_factory=list)


async def _state(pipe: FeedPipeline) -> State:
    """History + cached vector (2 commands) ∥ the profile; Postgres when Redis can't
    be trusted. Hidden and seen sets aren't needed to explain one item."""
    uid = pipe.uid
    reads = pipe._redis(("hmget", (rs.user_key(uid), "vec", "vec_key")),
                        ("lrange", (rs.hist_key(uid), 0, -1)))  # fmt: skip
    got, profile = await asyncio.gather(reads, pipe._profile(), return_exceptions=True)
    if isinstance(profile, BaseException):
        raise profile
    vec = None
    entries: list[rs.HistoryEntry] | None = None
    if isinstance(got, BaseException):
        pipe.redis_ok = False
        pipe.degrade("redis", got)
    else:
        raw_vec, raw_hist = got
        parsed = [rs.unpack(x) for x in raw_hist]
        if rs.is_consistent(parsed, profile.hist_version):
            entries = parsed
            vec = parse_vec(list(raw_vec), profile.hist_version, pipe.s.api_vec_ttl_s,
                            pipe.svc.clock().timestamp())  # fmt: skip
    if entries is None:
        entries, profile.hist_version = await pipe._history_from_postgres()
    return State(profile, set(), rs.latest(entries), set(), vec, full=True)


async def explain(svc: Services, uid: uuid.UUID, article_id: uuid.UUID, rid: str) -> Explanation:
    pipe = FeedPipeline(svc, uid, rid)
    async with svc.pool.acquire() as conn:
        rows = await svc.articles.get_many(repo.Q(conn), [article_id])
    article = rows.get(article_id)
    if article is None:
        raise ApiError(404, "article not found")
    st = await _state(pipe)
    snap = await svc.pools.get()
    u, w_topic = await pipe.user_vector(st, snap)

    # The model term, as SQL computes it: halfvec(user) · halfvec(article).
    u16 = np.asarray(u, np.float16).astype(np.float32)
    cos = float(u16 @ np.asarray(article.vector, np.float32))
    trends = trend_map(snap)
    raw_trend, outlets = trends.get(article.story_cluster_id, (0.0, 0))
    reference: list[float] = []
    if snap is not None:
        ids = {i for _, picked in pool_picks(snap, st.profile.topics) for i in picked}
        ids.discard(article_id)
        for i in ids:
            info = snap.articles.get(i)
            reference.append(trends.get(info.cluster, (0.0, 0))[0] if info else 0.0)
    trend = float(trend_feature(np.array([*reference, raw_trend]))[-1])
    age_h = (svc.clock() - article.first_seen_at).total_seconds() / 3600
    fresh = float(freshness(age_h, svc.ranking.half_life_hours))
    w_fresh, w_trend = svc.ranking.weights_for(len(st.history))
    final = float(
        blend(np.array([cos]), np.array([fresh]), np.array([trend]), svc.ranking.model_norm,
              w_fresh, w_trend)[0]
    )  # fmt: skip
    comp = {
        "model": final - w_fresh * fresh - w_trend * trend,
        "fresh": w_fresh * fresh,
        "trend": w_trend * trend,
        "final": final,
    }

    similar: list[tuple[uuid.UUID, str, float]] = []
    if st.history:
        hist = np.stack([np.asarray(e.vector, np.float32) for e in st.history])
        sims = hist @ np.asarray(article.vector, np.float32)
        top = [int(k) for k in np.argsort(-sims, kind="stable")[:SIMILAR]]
        wanted = [st.history[k].article_id for k in top]
        titles = await pipe._titles(set(wanted))
        similar = [
            (st.history[k].article_id, titles[st.history[k].article_id], round(float(sims[k]), 4))
            for k in top
            if st.history[k].article_id in titles
        ]
    nearest = similar[0][1] if similar else None
    why = reasons(ReasonInputs(
        model=comp["model"], fresh=comp["fresh"], trend=comp["trend"], age_hours=age_h,
        outlets=outlets, topic=article.category, followed=st.profile.topics,
        topic_weight=w_topic, nearest_read=nearest,
    ))  # fmt: skip
    return Explanation(
        article, comp, age_h, outlets, article.category in st.profile.topics, w_topic, similar,
        why, pipe.degraded,
    )  # fmt: skip
