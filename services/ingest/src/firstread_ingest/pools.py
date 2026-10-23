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
from firstread_core import stories
from firstread_core.pools import ArticleInfo, PoolSnapshot, TrendingEntry
from firstread_core.topics import TOPIC_IDS
from firstread_ingest import store
from firstread_ingest.config import load_outlet_networks
from firstread_ingest.window import CATEGORY_SOURCES, ORIGINS, TEXT_SOURCES, WindowSnapshot

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
TRENDING_CANDIDATES = 400  # clusters considered for merging before the top 200 are kept
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


# ---- trending v2 (ADR 0013): every outlet, networks collapsed, split stories merged ----

# Per (cluster, outlet) the earliest sighting; an outlet is a syndication network when
# the domain is in one (applied networks only), otherwise the domain itself. With no
# networks this is exactly the ADR 0010 score.
_OUTLETS = """
  select {key}, coalesce(net.network, sc.source_domain) as outlet,
         min(sc.first_seen_at) as first_seen_at
  from public.story_coverage sc
  {join}
  left join unnest($4::text[], $5::text[]) as net(domain, network)
    on net.domain = sc.source_domain
  where sc.first_seen_at > $1::timestamptz - $2::interval and sc.first_seen_at <= $1::timestamptz
  group by 1, 2
"""
_SCORE = (
    "sum(power(2.0, -extract(epoch from ($1::timestamptz - first_seen_at)) / 3600.0"
    " / $3::float8))::float8"
)

TRENDING_CANDIDATES_SQL = f"""
with cov as ({_OUTLETS.format(key="sc.story_cluster_id", join="")})
select story_cluster_id, {_SCORE} as score, count(*) as outlets
from cov group by 1 having count(*) >= $6::int
order by score desc, story_cluster_id limit $7::int
"""  # noqa: S608 (constant fragments)

MERGED_SQL = f"""
with m as (select * from unnest($6::uuid[], $7::int[]) as m(cluster, g)),
cov as ({_OUTLETS.format(key="m.g", join="join m on m.cluster = sc.story_cluster_id")})
select g, {_SCORE} as score, count(*) as outlets from cov group by g
"""  # noqa: S608 (constant fragments)


def _network_arrays(networks: dict[str, str]) -> tuple[list[str], list[str]]:
    domains = sorted(networks)
    return domains, [networks[d] for d in domains]


def quality_key(win: WindowSnapshot, k: int, networks: dict[str, str]) -> tuple[Any, ...]:
    """Representative order (ADR 0013), best first: RSS (a permitted feed with a
    description and usually an image), has an image, not a syndicated copy, a confident
    model topic, earliest seen, then id."""
    confident = (
        win.category_source[k] == CATEGORY_SOURCES.index("model")
        and win.confidence[k] >= CENTROID_MIN_CONFIDENCE
    )
    return (
        win.origin[k] != ORIGINS.index("rss"),
        not win.has_image[k],
        win.source_domain[k] in networks,
        not confident,
        int(win.first_seen_us[k]),
        win.ids[k].bytes,
    )


@dataclass
class TrendingInfo:
    candidates: int = 0
    merged_groups: int = 0  # entries made of more than one story cluster
    clusters_merged: int = 0  # clusters folded into another entry
    largest_group: int = 0
    no_window_article: int = 0  # candidate stories with nothing stored in the last 72 h
    samples: list[list[uuid.UUID]] = field(default_factory=list)  # merged rep ids


async def trending_entries(
    conn: Any,
    win: WindowSnapshot,
    now: datetime,
    networks: dict[str, str] | None = None,
    merge: bool = True,
    limit: int = TRENDING_CAP,
) -> tuple[list[TrendingEntry], TrendingInfo]:
    """Trending stories, best first, with near-identical stories merged (no chaining)."""
    nets = load_outlet_networks().mapping() if networks is None else networks
    domains, names = _network_arrays(nets)
    window_args = (now, TRENDING_WINDOW, HALF_LIFE_HOURS, domains, names)
    rows = await conn.fetch(TRENDING_CANDIDATES_SQL, *window_args, MIN_DOMAINS, TRENDING_CANDIDATES)
    info = TrendingInfo(candidates=len(rows))
    by_cluster: dict[uuid.UUID, list[int]] = defaultdict(list)
    for k, c in enumerate(win.clusters):
        by_cluster[c].append(k)
    cands = []  # (cluster, score, outlets, best window row)
    for r in rows:
        ks = by_cluster.get(r["story_cluster_id"])
        if not ks:
            info.no_window_article += 1  # the feed's window is 72 h: nothing to show
            continue
        best = min(ks, key=lambda k: quality_key(win, k, nets))
        cands.append((r["story_cluster_id"], float(r["score"]), int(r["outlets"]), best))
    if merge and cands:
        groups = stories.group(
            win.embedding[[c[3] for c in cands]].astype(np.float32),
            [TEXT_SOURCES[win.text_source[c[3]]] for c in cands],  # type: ignore[misc]
            [c[0] for c in cands],
        )
    else:
        groups = list(range(len(cands)))
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    multi = {g: m for g, m in members.items() if len(m) > 1}
    merged_scores: dict[int, tuple[float, int]] = {}
    if multi:
        flat = [(cands[i][0], g) for g, m in multi.items() for i in m]
        out = await conn.fetch(MERGED_SQL, *window_args, [c for c, _ in flat], [g for _, g in flat])
        merged_scores = {r["g"]: (float(r["score"]), int(r["outlets"])) for r in out}
    entries = []
    for g, m in members.items():
        rep = min((cands[i][3] for i in m), key=lambda k: quality_key(win, k, nets))
        score, outlets = merged_scores.get(g, (cands[m[0]][1], cands[m[0]][2]))
        entries.append(TrendingEntry(win.ids[rep], score, outlets, tuple(cands[i][0] for i in m)))
        if len(m) > 1:
            info.merged_groups += 1
            info.clusters_merged += len(m) - 1
            info.largest_group = max(info.largest_group, len(m))
            if len(info.samples) < 10:
                info.samples.append([win.ids[cands[i][3]] for i in m])
    entries.sort(key=lambda e: (-e.score, e.id.bytes))
    return entries[:limit], info


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
        "max(finished_at) as last_finished, (select status from public.ingestion_runs "
        "where finished_at is not null order by finished_at desc limit 1) as last_status "
        "from public.ingestion_runs where started_at > $1 and finished_at is not null",
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
        "last_run_status": runs["last_status"],
    }


async def build_snapshot(
    conn: Any, win: WindowSnapshot, now: datetime, version: str
) -> PoolSnapshot:
    fresh = fresh_members(win, now)
    topics = topic_members(win, now)
    entries, info = await trending_entries(conn, win, now)
    members = set(fresh) | {i for pool in topics.values() for i in pool} | {e.id for e in entries}
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
    snapshot = PoolSnapshot(
        version=version,
        built_at=now,
        fresh=list(fresh),
        topics={t: list(v) for t, v in topics.items() if v},
        trending=entries,
        articles=articles,
        centroids=centroids(win, now),
        stats=await stats(conn, win, now),
    )
    snapshot.stats["trending"] = {
        "candidates": info.candidates,
        "merged_groups": info.merged_groups,
        "clusters_merged": info.clusters_merged,
        "largest_group": info.largest_group,
    }
    return snapshot


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
