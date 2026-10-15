"""Redis candidate pools, rebuilt from Postgres at the end of each run (ADR 0010).

Pools are sorted sets of article ids the API reads at request time:

* ``pool:fresh``: first seen in the last 24 h, newest first (score = epoch), cap 500;
* ``pool:topic:{topic}``: per app topic, last 72 h, newest first, cap 300;
* ``pool:trending``: one article per story, top 200 by
  ``trend(c) = Σ_{distinct outlets d covering c} 2^(-age_d / 6 h)`` over
  ``story_coverage`` in the last 24 h, for stories carried by at least 2 outlets.
  The representative is the story's earliest stored article.

Postgres stays the source of truth: every write replaces a whole pool, so a lost or
stale Redis heals on the next run. Each pool is written as ``ZADD <key>:tmp`` +
``RENAME`` onto the live key (readers never see a half-built or empty pool), and an
empty pool is a ``DEL`` (``ZADD`` with no members and ``RENAME`` of a missing key both
error). All writes go in one non-transactional pipeline.

Command budget (Upstash bills every pipelined command): 2 per pool written, 1 per
empty pool; at most 2 x (2 + 13) = 30 per run. Topic pools are only rewritten when
they got new articles this run, plus all of them in the first run of each hour
(trims expired members).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from firstread_core.topics import TOPIC_IDS

FRESH, TRENDING = "pool:fresh", "pool:trending"
TOPIC_PREFIX = "pool:topic:"
FRESH_WINDOW, TOPIC_WINDOW, TRENDING_WINDOW = (
    timedelta(hours=24),
    timedelta(hours=72),
    timedelta(hours=24),
)
FRESH_CAP, TOPIC_CAP, TRENDING_CAP = 500, 300, 200
HALF_LIFE_HOURS = 6.0
MIN_DOMAINS = 2
MAX_COMMANDS = 30


def topic_key(topic: str) -> str:
    return f"{TOPIC_PREFIX}{topic}"


@dataclass
class PoolResult:
    commands: int = 0
    sizes: dict[str, int] = field(default_factory=dict)  # pools written this run


async def fresh_members(conn: Any, now: datetime) -> dict[str, float]:
    rows = await conn.fetch(
        "select id::text as id, extract(epoch from first_seen_at)::float8 as score "
        "from public.articles where first_seen_at > $1 and first_seen_at <= $2 "
        "order by first_seen_at desc limit $3",
        now - FRESH_WINDOW,
        now,
        FRESH_CAP,
    )
    return {r["id"]: r["score"] for r in rows}


async def topic_members(conn: Any, now: datetime) -> dict[str, dict[str, float]]:
    rows = await conn.fetch(
        "select category, id, score from ("
        "  select category, id::text as id, extract(epoch from first_seen_at)::float8 as score,"
        "    row_number() over (partition by category order by first_seen_at desc) as rn"
        "  from public.articles where first_seen_at > $1 and first_seen_at <= $2"
        ") t where rn <= $3",
        now - TOPIC_WINDOW,
        now,
        TOPIC_CAP,
    )
    out: dict[str, dict[str, float]] = {t: {} for t in TOPIC_IDS}
    for r in rows:
        out[r["category"]][r["id"]] = r["score"]
    return out


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
  select distinct on (story_cluster_id) story_cluster_id, id
  from public.articles
  where story_cluster_id in (select story_cluster_id from c)
  order by story_cluster_id, first_seen_at, id
)
select rep.id::text as id, rep.story_cluster_id, c.score::float8 as score, c.domains
from c join rep using (story_cluster_id)
order by c.score desc, rep.id
limit $5
"""


async def trending(conn: Any, now: datetime, limit: int = TRENDING_CAP) -> list[dict[str, Any]]:
    """Top stories by breadth-and-recency score, one representative article each."""
    rows = await conn.fetch(TRENDING_SQL, now, TRENDING_WINDOW, HALF_LIFE_HOURS, MIN_DOMAINS, limit)
    return [dict(r) for r in rows]


def plan_writes(
    fresh: dict[str, float],
    topics: dict[str, dict[str, float]],
    trend: dict[str, float],
    changed_topics: Iterable[str],
    full: bool,
) -> dict[str, dict[str, float]]:
    """The pools to (re)write this run: fresh and trending always, topics when needed."""
    writes = {FRESH: fresh, TRENDING: trend}
    wanted = set(TOPIC_IDS) if full else set(changed_topics) & set(TOPIC_IDS)
    for t in sorted(wanted):
        writes[topic_key(t)] = topics[t]
    return writes


async def write_pools(redis: Any, writes: dict[str, dict[str, float]]) -> int:
    """Replace each pool atomically in one pipeline; return the commands sent."""
    pipe = redis.pipeline(transaction=False)
    commands = 0
    for key, members in writes.items():
        if members:
            pipe.zadd(f"{key}:tmp", members)
            pipe.rename(f"{key}:tmp", key)
            commands += 2
        else:
            pipe.delete(key)
            commands += 1
    if commands > MAX_COMMANDS:
        raise RuntimeError(f"{commands} Redis commands planned, budget is {MAX_COMMANDS}")
    if commands:
        await pipe.execute()
    return commands


async def rebuild(
    redis: Any, conn: Any, now: datetime, changed_topics: Iterable[str]
) -> PoolResult:
    full = now.minute < 15  # first run of each hour rewrites every topic pool
    trend = {r["id"]: r["score"] for r in await trending(conn, now)}
    writes = plan_writes(
        await fresh_members(conn, now), await topic_members(conn, now), trend, changed_topics, full
    )
    commands = await write_pools(redis, writes)
    return PoolResult(commands=commands, sizes={k: len(v) for k, v in writes.items()})
