"""Candidate pools, built from the runner-side window and published as one blob (ADR 0012).

Each run builds a :class:`firstread_core.pools.PoolSnapshot`:

* ``fresh``: first seen in the last 24 h, newest first, cap 500;
* ``topics``: per app topic, last 72 h, newest first, cap 300 each;
* ``trending``: one representative per story, top 200 by
  ``trend(c) = Σ_{distinct outlets d covering c} 2^(-age_d / 6 h)`` over
  ``story_coverage`` in the last 24 h, for stories carried by at least 2 outlets;
* topic centroids and public stats.

Fresh, topic pools, centroids and stats come from the window snapshot (no egress).
Trending stays **one SQL aggregate over story_coverage**, so every outlet that ran a
story counts, dropped duplicates included (ADR 0010).

**Publishing costs 3 Redis commands a run** (Upstash bills every pipelined command):
``SET pools:blob:{ver}`` (no TTL), then ``SET pools:ver``, then
``EXPIRE pools:blob:{previous} 120``. The published and pending versions are kept in
``source_cursor`` (Postgres, free), so finding the previous blob needs no Redis read,
and a blob orphaned by a failed run is expired by the next one (one extra command,
only after a failure). The legacy ``pool:*`` sorted sets are removed once, by hand,
with ``firstread-ingest drop-legacy-pools``.

The legacy SQL builders (:func:`legacy_fresh`, :func:`legacy_topics`) are kept only
for the replay's equivalence check.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from firstread_core import pools as core_pools
from firstread_core.pools import ArticleInfo, PoolSnapshot, TrendingEntry
from firstread_core.topics import TOPIC_IDS
from firstread_ingest import store
from firstread_ingest.window import CATEGORY_SOURCES, TEXT_SOURCES, WindowSnapshot

FRESH_WINDOW, TOPIC_WINDOW, TRENDING_WINDOW = (
    timedelta(hours=24),
    timedelta(hours=72),
    timedelta(hours=24),
)
FRESH_CAP, TOPIC_CAP, TRENDING_CAP = (
    core_pools.FRESH_CAP,
    core_pools.TOPIC_CAP,
    core_pools.TRENDING_CAP,
)
HALF_LIFE_HOURS = 6.0
MIN_DOMAINS = 2
MAX_COMMANDS = 3  # per run; +1 only when expiring a blob orphaned by a failed run
CENTROID_MIN_CONFIDENCE = 0.50  # the release's low_confidence_threshold
CURSOR_PUBLISHED, CURSOR_PENDING = "pools", "pools_pending"
LEGACY_KEYS = ("pool:fresh", "pool:trending", *(f"pool:topic:{t}" for t in TOPIC_IDS))


def _epoch(us: int) -> float:
    return us / 1_000_000


# ---- fresh and topic pools from the window ----


def _in(win: WindowSnapshot, now: datetime, window: timedelta) -> np.ndarray:
    lo, hi = int((now - window).timestamp() * 1e6), int(now.timestamp() * 1e6)
    return np.flatnonzero((win.first_seen_us > lo) & (win.first_seen_us <= hi))


def _newest(win: WindowSnapshot, idx: np.ndarray, cap: int) -> list[int]:
    """Indices newest first; ties broken by id so every run orders the same way."""
    order = sorted(idx.tolist(), key=lambda i: (-win.first_seen_us[i], win.ids[i].bytes))
    return order[:cap]


def fresh_members(win: WindowSnapshot, now: datetime) -> dict[uuid.UUID, float]:
    idx = _newest(win, _in(win, now, FRESH_WINDOW), FRESH_CAP)
    return {win.ids[i]: _epoch(int(win.first_seen_us[i])) for i in idx}


def topic_members(win: WindowSnapshot, now: datetime) -> dict[str, dict[uuid.UUID, float]]:
    idx = _in(win, now, TOPIC_WINDOW)
    by_topic: dict[str, list[int]] = defaultdict(list)
    for i in idx.tolist():
        by_topic[TOPIC_IDS[win.category[i]]].append(i)
    out: dict[str, dict[uuid.UUID, float]] = {t: {} for t in TOPIC_IDS}
    for t, rows in by_topic.items():
        for i in _newest(win, np.asarray(rows, np.int64), TOPIC_CAP):
            out[t][win.ids[i]] = _epoch(int(win.first_seen_us[i]))
    return out


# ---- trending: one aggregate over story_coverage ----

TRENDING_SQL = """
with c as (
  select story_cluster_id,
         sum(power(2.0, -extract(epoch from ($1 - first_seen_at)) / 3600.0 / $3)) as score,
         count(*) as domains
  from public.story_coverage
  where first_seen_at > $1 - $2::interval and first_seen_at <= $1
  group by story_cluster_id
  having count(*) >= $4
), rep as (
  select distinct on (story_cluster_id) story_cluster_id, id, category, text_source,
         first_seen_at
  from public.articles
  where story_cluster_id in (select story_cluster_id from c)
  order by story_cluster_id, first_seen_at, id
)
select rep.id, rep.story_cluster_id, rep.category, rep.text_source,
       extract(epoch from rep.first_seen_at)::bigint as first_seen,
       c.score::float8 as score, c.domains
from c join rep using (story_cluster_id)
order by c.score desc, rep.id
limit $5
"""


async def trending(conn: Any, now: datetime, limit: int = TRENDING_CAP) -> list[dict[str, Any]]:
    """Top stories by breadth-and-recency score, one representative article each."""
    rows = await conn.fetch(TRENDING_SQL, now, TRENDING_WINDOW, HALF_LIFE_HOURS, MIN_DOMAINS, limit)
    return [dict(r) for r in rows]


# ---- the snapshot ----


def centroids(win: WindowSnapshot, now: datetime) -> dict[str, np.ndarray]:
    """Cold-start topic vectors from the last 72 h (confident model labels preferred)."""
    idx = _in(win, now, TOPIC_WINDOW)
    if not len(idx):
        return {}
    confident = (win.category_source[idx] == CATEGORY_SOURCES.index("model")) & (
        win.confidence[idx] >= CENTROID_MIN_CONFIDENCE
    )
    return core_pools.topic_centroids(
        win.embedding[idx].astype(np.float32),
        [TOPIC_IDS[c] for c in win.category[idx]],
        confident,
    )


async def stats(conn: Any, win: WindowSnapshot, now: datetime) -> dict[str, Any]:
    """Public numbers for /v1/stats: from the window, plus one tiny runs aggregate."""
    idx = _in(win, now, timedelta(hours=24))
    lag_s = (win.created_us[idx] - win.first_seen_us[idx]) / 1e6 if len(idx) else np.zeros(0)
    runs = await conn.fetchrow(
        "select count(*) as runs, count(*) filter (where status = 'ok') as ok, "
        "count(*) filter (where status = 'partial') as partial, "
        "max(finished_at) as last_finished from public.ingestion_runs "
        "where started_at > $1 and finished_at is not null",
        now - timedelta(hours=24),
    )
    return {
        "articles_24h": len(idx),
        "sources_24h": len({win.source_domain[i] for i in idx.tolist()}),
        "median_lag_s": round(float(np.median(lag_s)), 1) if len(lag_s) else None,
        "runs_24h": int(runs["runs"]),
        "runs_ok_24h": int(runs["ok"]),
        "runs_partial_24h": int(runs["partial"]),
        "last_run_finished": runs["last_finished"].isoformat() if runs["last_finished"] else None,
    }


async def build_snapshot(
    conn: Any, win: WindowSnapshot, now: datetime, version: str
) -> PoolSnapshot:
    fresh = fresh_members(win, now)
    topics = topic_members(win, now)
    trend = await trending(conn, now)
    entries = [
        TrendingEntry(r["id"], float(r["score"]), int(r["domains"]), (r["story_cluster_id"],))
        for r in trend
    ]
    members = set(fresh) | {i for pool in topics.values() for i in pool}
    pos = {aid: k for k, aid in enumerate(win.ids)}
    articles: dict[uuid.UUID, ArticleInfo] = {}
    for aid in members:
        k = pos[aid]
        articles[aid] = ArticleInfo(
            win.clusters[k],
            TOPIC_IDS[win.category[k]],
            int(win.first_seen_us[k] // 1_000_000),
            TEXT_SOURCES[win.text_source[k]],
        )
    for r in trend:  # the representative may be older than the 72 h window
        articles.setdefault(
            r["id"],
            ArticleInfo(
                r["story_cluster_id"], r["category"], int(r["first_seen"]), r["text_source"]
            ),
        )
    return PoolSnapshot(
        version=version,
        built_at=now,
        fresh=list(fresh),
        topics={t: list(v) for t, v in topics.items() if v},
        trending=entries,
        articles=articles,
        centroids=centroids(win, now),
        stats=await stats(conn, win, now),
    )


@dataclass
class PoolResult:
    commands: int = 0
    version: str = ""
    blob_bytes: int = 0
    sizes: dict[str, int] = field(default_factory=dict)
    orphan_expired: bool = False


async def publish(redis: Any, conn: Any, snap: PoolSnapshot) -> PoolResult:
    """Write the blob, move ``pools:ver``, expire the previous blob (3 commands)."""
    blob = core_pools.encode(snap)
    if len(blob) > core_pools.MAX_BLOB_BYTES:
        raise RuntimeError(f"pool snapshot is {len(blob)} bytes, over {core_pools.MAX_BLOB_BYTES}")
    previous = await store.load_cursor(conn, CURSOR_PUBLISHED)
    pending = await store.load_cursor(conn, CURSOR_PENDING)
    pipe = redis.pipeline(transaction=False)
    commands, orphan = 0, False
    if pending and pending not in (previous, snap.version):
        pipe.expire(core_pools.blob_key(pending), core_pools.PREVIOUS_GRACE_S)
        commands += 1
        orphan = True
    await store.save_cursor(conn, CURSOR_PENDING, snap.version, snap.built_at)
    pipe.set(core_pools.blob_key(snap.version), blob)
    pipe.set(core_pools.VER_KEY, snap.version)
    commands += 2
    if previous and previous != snap.version:
        pipe.expire(core_pools.blob_key(previous), core_pools.PREVIOUS_GRACE_S)
        commands += 1
    if commands > MAX_COMMANDS + int(orphan):
        raise RuntimeError(f"{commands} Redis commands planned, budget is {MAX_COMMANDS}")
    await pipe.execute()
    await store.save_cursor(conn, CURSOR_PUBLISHED, snap.version, snap.built_at)
    return PoolResult(
        commands=commands,
        version=snap.version,
        blob_bytes=len(blob),
        sizes=dict(core_pools.summarize(snap)),
        orphan_expired=orphan,
    )


async def drop_legacy(redis: Any) -> int:
    """One-off: delete ADR 0010's sorted-set pools (one DEL). Returns keys removed."""
    removed: int = await redis.delete(*LEGACY_KEYS)
    return removed


# ---- legacy SQL builders, for the replay's equivalence check only ----


async def legacy_fresh(conn: Any, now: datetime) -> dict[uuid.UUID, float]:
    rows = await conn.fetch(
        "select id, extract(epoch from first_seen_at)::float8 as score "
        "from public.articles where first_seen_at > $1 and first_seen_at <= $2 "
        "order by first_seen_at desc limit $3",
        now - FRESH_WINDOW,
        now,
        FRESH_CAP,
    )
    return {r["id"]: r["score"] for r in rows}


async def legacy_topics(conn: Any, now: datetime) -> dict[str, dict[uuid.UUID, float]]:
    rows = await conn.fetch(
        "select category, id, score from ("
        "  select category, id, extract(epoch from first_seen_at)::float8 as score,"
        "    row_number() over (partition by category order by first_seen_at desc) as rn"
        "  from public.articles where first_seen_at > $1 and first_seen_at <= $2"
        ") t where rn <= $3",
        now - TOPIC_WINDOW,
        now,
        TOPIC_CAP,
    )
    out: dict[str, dict[uuid.UUID, float]] = {t: {} for t in TOPIC_IDS}
    for r in rows:
        out[r["category"]][r["id"]] = r["score"]
    return out


def same_pool(a: dict[uuid.UUID, float], b: dict[uuid.UUID, float], cap: int) -> bool:
    """Equal, allowing only a different pick among rows tied at the cap boundary.

    SQL's ``order by first_seen_at desc limit n`` breaks ties arbitrarily; the
    snapshot breaks them by id. Any other difference is a real mismatch.
    """
    if a == b:
        return True
    if len(a) != len(b) or len(a) < cap:
        return False
    edge = min(a.values())
    if edge != min(b.values()):
        return False

    def strip(d: dict[uuid.UUID, float]) -> dict[uuid.UUID, float]:
        return {k: v for k, v in d.items() if v != edge}

    return strip(a) == strip(b)
