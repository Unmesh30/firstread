# ADR 0010: Redis candidate pools and trending, rebuilt from Postgres each run

- Status: Accepted
- Date: 2026-10-15

## Context

At request time the API (P3) needs candidate articles fast: the freshest stories, the freshest per topic the reader follows, and what's trending (SPEC §8: candidate generation ≤ 60 ms). Postgres can answer these, but the free Supabase instance is shared with everything else, and the API wants a few-millisecond read.

The budget is tight:
- Upstash Redis's free tier is **500K commands per month** (256 MB, 10 GB bandwidth, 10K commands/s), checked on [upstash.com/pricing/redis](https://upstash.com/pricing/redis) on 2026-10-15.
- Upstash's billing docs state that **"each command inside a pipeline or transaction is still billed individually"** ([billing](https://upstash.com/docs/redis/overall/billing)). Pipelining saves round trips, not commands.
- SPEC §9 caps our own use at 300K/month, and the API needs most of that.

"Trending" needs a definition that works without clicks (P3 adds engagement later) and isn't fooled by press releases.

## Decision

**Pools are sorted sets of article ids, rebuilt from Postgres at the end of every ingestion run** (`services/ingest/src/firstread_ingest/pools.py`):

| Key | Members | Score | Cap |
|---|---|---|---|
| `pool:fresh` | Articles first seen in the last 24 h | `first_seen_at` epoch | 500 |
| `pool:topic:{topic}` (13) | Per app topic, last 72 h | `first_seen_at` epoch | 300 |
| `pool:trending` | One representative article per story, last 24 h | trend score | 200 |

**Trending score** for story `c`, from `story_coverage` over the last 24 h:

`trend(c) = Σ_{distinct outlets d covering c} 2^(−age_d / 6 h)`

- `age_d` is the time since outlet `d` first carried the story.
- A story needs **≥ 2 outlets**. The representative is the story's earliest stored article.
- The sum rewards **breadth**: many independent outlets.
- The 6-hour half-life rewards **recency**: a story five outlets covered 3 h ago (5 × 0.71 = 3.5) beats one five outlets covered 20 h ago (5 × 0.10 = 0.5).
- `story_coverage` counts outlets that ran a dropped duplicate too, so a wire story carried by 30 outlets counts 30 (ADR 0009).
- **Press releases never reach coverage:** press-wire domains and PR-shaped announcement titles are dropped before dedupe (ADR 0007). Otherwise a syndicated earnings release would look like the biggest story of the day.

**Writes:**
- Each pool is replaced whole: `ZADD <key>:tmp` with all members, then `RENAME` onto the live key, so a reader sees either the old pool or the new one, never a half-built or empty one.
- **An empty pool is `DEL`'d** instead, because `ZADD` with no members and `RENAME` of a missing key both error.
- Every write goes in **one non-transactional pipeline**: one round trip, no `MULTI`/`EXEC` (those would be billed too).

**Command budget:**
- **2 commands per pool written, 1 per empty pool.** `pool:fresh` and `pool:trending` are written every run.
- A topic pool is written when it got new articles this run, plus all 13 in the first run of each hour (`minute < 15`), which trims expired members.
- Maximum **2 × (2 + 13) = 30 per run**, enforced in code (a write that plans more raises) and recorded in `ingestion_runs.redis_commands`.
- At 96 runs a day that is at most ~86K a month (typically fewer; measured in the replay, learn chapter 02), well under the 300K SPEC cap and Upstash's 500K.

**Postgres is the source of truth.** If Redis is unreachable, the run finishes as `partial`, the articles are still stored, and the next run rebuilds every pool. A flushed or replaced Redis heals within one run (the first run of the next hour for the topic pools).

## Alternatives considered

- **Incremental updates** (`ZADD` new members, `ZREMRANGEBYSCORE` old ones). This uses similar command counts, but drifts after any failure or manual change, needs separate trimming, and can't express a re-ranked trending list. Whole-pool rebuilds are idempotent and self-healing.
- **Query Postgres from the API instead of Redis.** Simpler, but every feed request would hit the shared free Postgres with window queries. Redis reads are one `ZREVRANGE` per pool.
- **`MULTI`/`EXEC` around the swap.** Not needed, since `RENAME` is atomic on its own, and Upstash bills `MULTI` and `EXEC` as commands.
- **Trending by raw article count.** Counts one prolific outlet's rewrites as breadth and has no recency. Distinct outlets with decay is the simplest score that captures "many outlets are covering this right now".
- **Trending from clicks.** There are no clicks before P3. The API can blend an engagement term into the score later without changing these pools.

## Consequences

- The API's candidate generation is a handful of Redis reads.
- Pools lag ingestion by at most one run (15 min).
- Article ids in pools may briefly point at articles that retention just deleted (14 days, far older than any pool window). The API must tolerate a missing id.
- The trending formula's constants (24 h window, 6 h half-life, ≥ 2 outlets) are guesses validated only by eye on the replay's daily top 10 (learn chapter 02). P3/P6 can tune them with real engagement data.
