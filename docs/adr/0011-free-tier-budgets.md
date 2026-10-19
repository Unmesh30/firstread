# ADR 0011: Free-tier budgets as hard constraints

- Status: Accepted (ingestion measured on the replay; sessions estimated, re-measured in P3e and against the real usage pages in P5)
- Date: 2026-10-18

## Context

FirstRead must cost $0 a month (SPEC §4). Phase 3 adds an API in front of the same free services ingestion already uses, so each one's quota is now shared. A design that is cheap per request can still exhaust a monthly quota at modest traffic, and some quotas fail hard: **Supabase Free goes read-only above 500 MB of database size**, which blocks ingestion too.

Quotas checked 2026-10-18:

| Service | Quota | Source |
|---|---|---|
| Supabase Free egress | 5 GB uncached + 5 GB cached a month; with the shared pooler, the pooler's egress is what counts | [egress](https://supabase.com/docs/guides/platform/manage-your-usage/egress) |
| Supabase Free database | read-only above 500 MB (database size, not disk); deleted rows free space only after vacuum | [database size](https://supabase.com/docs/guides/platform/database-size) |
| Upstash Redis Free | 500K commands, 250 MB, 10 GB bandwidth a month; every command in a pipeline or transaction is billed | [pricing](https://upstash.com/pricing/redis), [billing](https://upstash.com/docs/redis/overall/billing) |
| Cloud Run (request-based billing) | 2M requests, 180K vCPU-s, 360K GiB-s a month; CPU only during requests | [billing settings](https://docs.cloud.google.com/run/docs/configuring/billing-settings), [pricing](https://cloud.google.com/run/pricing) |
| GitHub Actions cache | 10 GB per repository, LRU eviction, 7-day idle eviction | [caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching) |

SPEC §9 also caps our own Upstash use at **300K commands a month**, below the 500K quota.

## Decision

Every design choice in Phase 3 is costed per ingestion run and per feed session against these quotas, and the totals are kept in the table below. Counters make the costs observable:
- `ingestion_runs.redis_commands` and `details.egress_bytes_est`;
- in the API (P3c/d), `redis_commands_total` and `db_bytes_estimated_total`.

**A feed session** is:
- 3 feed pages, 2 computed and 1 served from the cached slate;
- 2 clicks and ~40 viewed impressions, in 3 event batches;
- 1 "Why this?" and 1 profile or topics call;
- about 10 HTTP requests.

**Demo traffic level:** 100 feed sessions a day (3,000 a month).

| Resource | Quota | Per ingestion run | Per feed session | At 100 sessions/day | Sessions/day the budget allows |
|---|---|---|---|---|---|
| Supabase egress | 5 GB | **84 KB measured** (replay median; p95 86 KB): window delta, check, trending aggregates, bookkeeping → 0.24 GB/month, plus ~4 full window reloads × 14 MB | ~200 KB (2 computed pages × ~96 KB, plus events and profile) | ~0.9 GB (18%) | ~790 |
| Supabase DB size | 500 MB (target 400; guard from 350) | articles ~227 MB and coverage ~13 MB at steady state | events (impressions kept 7 days), history ≤ 50 rows/user, slates | ~280 MB | ~400 before the 350 MB guard |
| Upstash commands | 300K (SPEC) / 500K | **3** (blob, version, expire previous) → 8.6K/month | ~28 | ~108K (36%) | **~330 (binding)**; ~570 against 500K |
| Upstash bandwidth | 10 GB | one blob write, **192 KB measured** → 0.55 GB/month | ~160 KB | 0.5 GB + blob writes 0.55 GB + blob reads 0.55–1.7 GB (1–3 instances) = ≤ 2.7 GB | ~1,500 |
| Upstash memory | 250 MB | 1 live blob (2 briefly) | ~60 KB per active reader (7-day TTL) | ~30 MB | ~4,000 weekly-active readers |
| Cloud Run requests | 2M | — | ~10 | 30K (1.5%) | ~6,600 |
| Cloud Run vCPU-s | 180K | — | ~1 (100 ms rounding, 1 vCPU), plus cold starts | ~6K (3%) | ~5,600 |

**Binding constraint: Upstash commands**, at ~330 sessions a day under SPEC's 300K cap.

**Design rules that follow:**
- Ingestion keeps the 72 h window on the runner and publishes pools as one blob ([ADR 0012](0012-runner-window-and-pool-snapshot.md)).
- The API never pulls vectors in bulk per request:
  - dot products run in SQL;
  - vectors are fetched only for the re-rank set;
  - article rows are cached in-process (they never change after insert);
  - user history vectors live in Redis, with Postgres as the durable copy.
- No timers and no work after a response on Cloud Run: refreshes happen lazily inside requests, and events are written before responding.
- Rate limits are in-process, because Redis limits would cost a command per request.

## Alternatives considered

- **Assume demo traffic stays tiny and skip the accounting.** One recruiter link shared widely could exhaust the command quota in days; the counters make that visible before it happens.
- **A paid tier for headroom.** Ruled out by SPEC §4.

## Consequences

- Each Phase 3 PR states its cost against this table, and P3e replaces the session estimates with numbers measured in the load test.
- Above ~330 sessions a day the first thing to give is Redis commands: the next levers are fewer impression batches per session and a longer user-vector cache.
- P5 compares the estimates with Supabase's and Upstash's usage pages.
