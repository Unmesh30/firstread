# ADR 0011: Free-tier budgets as hard constraints

- Status: Accepted (ingestion measured on the replay; sessions measured on the local stack in P3e, 2026-10-23; P5 checks both against the real usage pages)
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

### Measured per session (P3e, 2026-10-23)

`loadtest/session_cost.py` runs ADR 0011's session against the API image at Cloud Run's shape (1 vCPU, 512 MiB) on the local stack and reads the API's own counters (`redis_commands_total`, `db_bytes_estimated_total`) and Redis' `INFO` network bytes around it. The session: `GET /v1/me`; page 1; 20 impressions + a click; page 2 (re-ranked); 20 impressions; page 3 (from the slate); a click; one `/why`. That is 8 requests (estimated: ~10). Results in `loadtest/results/session-*.json`.

| Instance and reader | Redis commands | Postgres egress (est.) | Redis bytes (both ways) | Billable time |
|---|---:|---:|---:|---:|
| **Cold instance** (just restarted: no snapshot, empty article cache), reader with no Redis state | 37 | 373 KB | 289 KB (incl. the 193 KB blob) | 0.8 s |
| Warm instance, reader's first session (Redis state rebuilt: +5 commands) | 35 | 178 KB (median of 20) | 96 KB | 0.8 s |
| **Warm instance, returning reader** | **30** | **153 KB** (median of 20) | **106 KB** | 0.8 s |

- **Commands, returning reader:** page 1: 4 reads + slate `SET` + vector cache `HSET`/`EXPIRE` = 7; click batch 5 (impressions 2, click 3); page 2 re-rank 8; impression batch 2; page 3 from the slate 3; click 3; `/why` 2. The estimate was 28; ADR 0017's per-page counts were right, the vector-cache writes were not in the estimate.
- **Egress:** per computed page, the pool cosines (~600 rows × ~70 B) are the largest part, then ANN (200 × 70 B). The estimates were ~115 KB warm and ~290 KB cold; the measured 153 KB and 373 KB are higher because `/why`, the history titles and the first page's history rebuild were not in them.
- The egress column is the API's estimate from result sizes (`rows_bytes`), not wire bytes; P5 compares it with Supabase's usage page.

### The table, with measured sessions

| Resource | Quota | Per ingestion run | Per feed session (measured) | At 100 sessions/day | Sessions/day the budget allows |
|---|---|---|---|---|---|
| Supabase egress | 5 GB | **84 KB measured** (replay median; p95 86 KB): window delta, check, trending aggregates, persona summaries (~2 KB), bookkeeping → 0.25 GB/month, plus ~4 full window reloads × 14 MB | **153 KB warm, 373 KB cold** | 0.3 GB + 0.46–1.1 GB = 0.8–1.4 GB (15–28%) | **~1,000 warm, ~420 cold** |
| Supabase DB size | 500 MB (target 400; guard from 350) | articles ~227 MB and coverage ~13 MB at steady state; personas ~0.1 MB | events (impressions kept 7 days), history ≤ 50 rows/user, slates | ~280 MB | ~400 before the 350 MB guard |
| Upstash commands | 300K (SPEC) / 500K | **3** (blob, version, expire previous) → 8.6K/month | **30 returning, 35 first visit, 37 cold** | 8.6K + pool checks ~15K + 90–111K = 114–135K (38–45%) | **~250–310 (binding)**; ~430–530 against 500K |
| Upstash bandwidth | 10 GB | one blob write, **193 KB measured** (with personas) → 0.56 GB/month | **106 KB warm, 289 KB cold** | 0.32–0.87 GB + blob writes 0.56 GB + blob reads 0.56–1.7 GB (1–3 instances) ≤ 3.1 GB | ~2,400–2,800 warm, ~900 cold |
| Upstash memory | 250 MB | 1 live blob (2 briefly) | ~60 KB per active reader (7-day TTL) | ~30 MB | ~4,000 weekly-active readers |
| Cloud Run requests | 2M | — | 8 | 24K (1.2%) | ~8,300 |
| Cloud Run vCPU-s | 180K | — | ≤ 0.8 (8 requests, 100 ms rounding, sequential), plus cold starts (~0.8 s each) | ~2.4K (1.3%) | ~7,500 |
| Cloud Run GiB-s | 360K | — | ≤ 0.4 (512 MiB) | ~1.2K | ~30,000 |

**Binding constraint: still Upstash commands,** at about 250–310 sessions a day under SPEC's 300K cap (estimated: ~330). Supabase egress comes next on cold instances (~420 a day if every session hit a cold instance).

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

- Each Phase 3 PR states its cost against this table; P3e replaced the session estimates with measured numbers (above).
- Above ~250–300 sessions a day the first thing to give is Redis commands. The levers, in order of commands saved per session: skip the user-vector cache write when the encoder is cheap (it costs 2 per computed page and saves at most an encode, 0.7 ms), batch impressions less often (2 per batch), and a longer page cache.
- P5 compares the estimates with Supabase's and Upstash's usage pages.
