# ADR 0017: The feed serving pipeline: stages, deadlines, degraded modes, slates, ANN

- Status: Accepted
- Date: 2026-10-23

## Context

`GET /v1/feed` turns stored, embedded live articles into a ranked page for one reader (SPEC §8). Five constraints shape it:

1. **Cloud Run bills CPU only during requests** (request-based billing). Nothing may run after the response, and there is no background loop.
2. **Commands and egress are budgets** (ADR 0011). Upstash bills every command, pipelined ones included. Supabase bills pooler egress, so vectors must not cross the wire in bulk.
3. **Train/serve consistency.** The blend, MMR, cold-start vector, story thresholds and reasons exist once, in `packages/core` (ADR 0013/0014). The API only wires them together.
4. **Every dependency fails sometimes:** Redis, the encoder, the ANN query, a missing pool snapshot on a cold instance. Each failure needs a defined, tested answer.
5. **Pages must not repeat,** and a click must show on the next page.

## Decision

### Model loading

- The verifying loader lives in `firstread_core.release`, with `huggingface_hub` behind the `core[release]` extra. Ingest imports it too.
- **The Docker `models` stage** downloads v0.2.0 at its pinned Hub commit and verifies the manifest SHA256, each file's hash and the manifest's `embedding`/`features` blocks against core. The runtime and dev images get only `user_encoder.onnx`, `population.npy` and `manifest.json` (3.3 MB), with `API_MODEL_DIR` pointing at them.
- At startup the API re-verifies the same pins, with no network and no `huggingface_hub`. `/readyz` adds `model: ok` and `model_tag`.
- Prod refuses to start without `API_MODEL_DIR`, or with `API_LOAD_MODEL=false`.

### Micro-batching (`encoder.py`)

- Concurrent requests that need a fresh user vector share one `session.run`. The batch closes at 32 requests or after 5 ms. The call runs on a dedicated single worker thread, with ONNX intra-op threads = 1.
- There is no background loop: the first request of a batch starts a short-lived driver task.
- **Failures:** a cancelled waiter is skipped and the batch still runs for the others; an exception reaches every waiter of the batch; shutdown rejects new work and drains the in-flight batch.
- **`API_ENCODER_BATCHING`** turns batching off (P6 compares the two).

### Pipeline (`feed.py`)

The whole request has a 1 s deadline (→ 503). Each stage has its own deadline. A stage's deadline covers its queries, not the wait for a pooled connection, which only the request deadline bounds.

| # | Stage | Work | Deadline | On failure |
|---|---|---|---|---|
| 1 | auth | JWT against the cached JWKS (the route dependency) | — | 401, or 503 `auth_unavailable` |
| 2 | state | Redis pipeline ∥ the profile row; version check; rebuilds | 30 ms | **Redis down:** history and hides from Postgres, no seen set, served ids from `served_slates` → `degraded: ["redis"]` |
| 3 | encode | the cached vector if computed for this `hist_version` < 10 min ago; else the batcher; then `cold_start_vector` (population vector when there are no centroids) | 30 ms | **Error or timeout:** mean of history → `encoder` |
| 4 | cand | ANN top-200 ∥ cosines for pool members (fresh 200, trending 200, 100 per followed topic), both in SQL | 80 ms each | **ANN down:** pools only (`ann`). **No snapshot yet:** ANN only (`pools`). **Both:** 503 |
| 5 | hydrate | top 150 by the pre-hydrate blend, from the 20K-row LRU; misses in one query | 60 ms | 503 |
| 6 | rank | read-story filter; story groups; MMR; reasons | CPU | — |
| 7 | persist | `served_slates` insert, **then** the Redis slate (and the vector cache) | 50 ms | Postgres: 503 `Retry-After: 5`. Redis: `redis` |

- **Postgres comes before Redis in persist.** A page that wasn't recorded must not be marked as served. This was found by a test: written in parallel, a failed insert still saved the slate.
- **Every candidate row carries `(id, score, first_seen_at, story_cluster_id, source_domain)`,** about 70 B. So the blend has age, story and source for ANN hits outside the pools too.
- **Exclusions run in SQL.** Ids and clusters go in as parameters, and ingress is free. Excluded: seen, clicked, hidden and served ids; hidden sources; the clusters of read stories; the clusters of hidden articles, looked up server-side.
- **The rank stage drops any item near-identical to something read,** at τ_story for its pair type (0.925 / 0.80 / 0.875, ADR 0013).
- **Story groups use complete linkage at τ_feed** (`stories.group`, no chaining), and MMR applies λ = 0.7 with ≤ 2 items per group.
- **The cap also counts what the reader was already shown.** The newest 100 served items lead the grouping and count toward the cap, so a re-rank can't bring back a third copy of a story. This was found in the local curl walk: page 2 repeated a page-1 story.
- **Trend features come from the snapshot,** by story cluster: an article in a trending story gets that story's score and outlet count.
- **Reasons** use the blend's actual components. "Because you read …" needs history titles: one small query, only when topic weight < 0.5.
- **Observability:**
  - `Server-Timing: auth;dur=…, state;…, encode;…, cand;…, hydrate;…, rank;…, persist;…`;
  - the same breakdown in the `feed` log line;
  - `feed_stage_seconds{stage}`, `degraded_total{reason}`, `encoder_batch_size`, `encoder_seconds`, `pools_snapshot_age_seconds`, `pools_refresh_total{result}`, `history_rebuilds_total{kind}`.

### Pages (`slate.py`)

- **The first page** (no cursor) ranks a slate of up to 100 items and stores it in `u:{uid}:slate` (zlib JSON, 30 min).
- **The cursor** is opaque base64url `{"s": slate_id, "o": offset, "v": 1}`, strictly validated: malformed → 400 `invalid_cursor`. A slate this reader doesn't have (expired with no `served_slates` rows, or another reader's) → 400 `cursor_expired`.
- **A cursor page reads only `GET` slate + `SMEMBERS` hidden** (2 commands) ∥ the profile row. It re-ranks the rest, always excluding everything served, when:
  - `hist_version` moved (a click);
  - the hidden signature changed (a hide);
  - fewer than `limit` items remain.
  Only then does it read the other three keys.
- **A Redis slate behind the cursor is treated as missing.** If a page's slate write was lost (Redis failed during that request's read or write), `u:{uid}:slate` still holds the previous version, which isn't stale by `hist_version` or the hidden signature. A slate whose served count is below `min(offset, 1000)` is therefore ignored and the served ids come from `served_slates`, so the next page can't repeat the lost page (PR #20 review).
- **The same page asked again within 60 s returns the same items.** It still writes a `served_slates` row, so events from that response join.

**Measured per page** (`test_redis_commands_per_page`, plus the local stack's `/metrics` after a real ingestion run):

| Page | Redis commands | Estimated Postgres egress |
|---|---|---|
| first page | 5 (+2 for a brand-new reader: the hidden set's sentinel is written once; +1–2 when the instance checks the pool snapshot) | 37 KB with a warm article cache, 209 KB on a cold instance (150 rows with vectors) |
| cursor page served from the slate | 3 | 0.1 KB |
| a repeat within 60 s | 2 | ~0.2 KB |
| re-rank after a click | 8 | 77 KB |

- **Commands:** these replace ADR 0011's estimate of 6 per computed page. A 3-page session with one click costs about 5 + 8 + 3 = 16 feed commands (estimated before: 14).
- **Egress:** a session costs about 115 KB on a warm instance and about 290 KB on a cold one (estimated before: ~200 KB). With min instances 0, demo traffic will often hit cold instances, so P3e's k6 run re-measures the session and updates ADR 0011's table.

### Pools in-process (`pools_cache.py`)

- **The refresh has its own 0.3 s deadline** (`API_POOLS_REFRESH_TIMEOUT_S`, `pools_refresh_total{result="timeout"}`). Without it, a hanging Redis held every request on a cold instance until the 1 s feed deadline (503) instead of serving ANN-only (PR #20 review).

- At most one `GET pools:ver` a minute per instance, inside a request; the blob is fetched only when the version changed. One refresh runs at a time (single-flight).
- **Warm requests** keep using the current copy during a refresh. **A cold instance** waits for the in-flight refresh.
- **A bad, mismatched or missing blob** keeps the last good copy. With no copy at all, the feed is ANN-only. There's never a per-request SQL fallback.

### The Redis client fails fast

redis-py 8.1 defaults to a 5 s socket (and connect) timeout and up to 10 retries on connection errors and timeouts, with backoff capped at 1 s (`redis/_defaults.py`), so an un-wrapped call could hang for about a minute. The API's client uses `REDIS_SOCKET_TIMEOUT_S` and `REDIS_CONNECT_TIMEOUT_S` of 0.5 s and `REDIS_RETRIES` = 1 with no backoff; the config refuses larger values. Calls outside the feed's deadlines (events, `/me/topics`) therefore fail within about a second, and their Redis writes are best effort: Postgres holds the truth and the version check heals Redis (PR #20 review).

### ANN (pgvector 0.8): HNSW iterative scans at ef_search 400

- **Query shape:** `SET LOCAL hnsw.ef_search` and `SET LOCAL hnsw.iterative_scan = relaxed_order` inside the query's transaction, because session settings don't survive the transaction pooler. Then a materialized CTE re-sorted with `ORDER BY d + 0`.
- **Benchmark** (`services/api/scripts/ann_bench.py`, local Docker): a 14-day-sized table of 72,000 rows, 14,400 of them in the 72 h window, and 200 synthetic readers.
- **Pre-registered rule:** HNSW if recall@200 ≥ 0.95 **and** p95 ≤ exact's p95; otherwise exact.

| Query shape | Exact p95 | HNSW ef400 p95 | HNSW recall@200 | Decision |
|---|---|---|---|---|
| Shipped (with the story filters), 3 runs (`repeats`) | 5.63–5.83 ms | 5.35–5.52 ms | 0.9834–0.984 (lowest single query 0.94, in the run in `results`) | **HNSW ef400**, all 3 runs |
| First draft (no story filters), 1 run (`previous_query_shape`) | 5.28 ms | 5.29 ms | 0.9831 (in the file as committed in 51fe0cf) | exact, by 0.01 ms |

- The rule is applied to the query that ships. Both shapes are kept in `docs/learn/data/03-ann.json`. An earlier run of the shipped shape crashed at the script's EXPLAIN step before writing its file; it isn't counted.
- **The cost:** about 1.7% of the exact top 200 is missed, on a candidate set that the pools also feed.
- **The CI `supabase` job** asserts pgvector ≥ 0.8 and runs both shapes once, as the non-superuser `postgres`. It also checks that `SET LOCAL` doesn't leak into the session.

### Encoder latency inside the image (`python -m firstread_api.bench_encoder`, `docker run --cpus=1 --memory=512m`)

| ONNX threads | Batch | p50 | p95 | per user (p50) |
|---|---|---|---|---|
| 1 | 1 | 0.74 ms | 0.76 ms | 0.74 ms |
| 1 | 32 | 22.4 ms | 25.2 ms | 0.70 ms |
| 2 | 1 | 0.44 ms | 0.50 ms | 0.44 ms |
| 2 | 32 | 12.8 ms | **66.5 ms** | 0.40 ms |

- **One thread stays the default.** With 2 threads under a 1-CPU quota, CFS throttling blows up the p95. The container also *sees* 10 CPUs, which is why the thread count is pinned instead of left to ONNX Runtime.
- **On one core the cost per user is nearly flat** (0.74 → 0.70 ms), so batching saves little CPU. Its value is fewer thread hand-offs under concurrency, which P6 measures with batching on and off.
- **A full batch of 32 (≈ 25 ms) plus the 5 ms window would reach the 30 ms encode deadline.** But it needs about 32 encodes inside 5 ms, around 6,000 encodes a second, far beyond one vCPU's feed throughput. If P6 ever sees `encoder` degradations under load, the fix is a smaller `API_BATCH_MAX`.
- **Image:** 406 → 503 MB uncompressed (onnxruntime and its dependencies, plus the 3.3 MB of model files). Healthy 1.05 s after start at 1 vCPU / 512 MB, 78 MiB resident.

## Alternatives considered

- **A background batching loop or a background pool refresher.** These need CPU outside requests: unbilled on request-based Cloud Run, so they would stall, or cost money with instance-based billing.
- **Persisting Postgres and Redis in parallel.** About 1 ms faster, but a failed insert left Redis marking a page as served that the client never received.
- **Filtering read and hidden stories after hydration only.** It wastes ANN slots and hydrate egress on items that are dropped anyway.
- **A per-slate story cap only** (the first draft). It repeats a story across pages after a re-rank.
- **Re-reading the full state for every cursor page.** That is 4 commands instead of 2, for a version check the profile row already answers.
- **Exact search everywhere.** Exact would also have been acceptable at 14,400 rows, but the pre-registered rule picks HNSW on the shipped query, and HNSW scales better past the 72 h window size.

## Consequences

- Every degraded row of the stage table has a test (`services/api/tests/test_feed.py`). So do the E2E flow (anonymous reader → topics → feed → click → re-ranked page 2 → delete), cursor validation, no repeats over 5 pages, the 60 s page cache, every filter, the story cap within and across pages, and command counts. They run against CI's Postgres and Redis with the real encoder; the release files are cached in CI by the hash of `release.py`.
- Tests can't fake `X-Request-ID` into `served_slates`: the id is always server-generated (ADR 0015).
- **Local only so far:** Supabase Nano is shared CPU, so HNSW, exact and pooler latencies there may be 2–3× local. P5's smoke tests re-measure, and the rule can be re-run against the live database.
- P6 compares batching on and off, and caching on and off, under k6. P3e reports the local p95 for the whole feed.
