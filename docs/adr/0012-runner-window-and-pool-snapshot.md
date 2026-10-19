# ADR 0012: A runner-side window for ingestion, and pools as one versioned snapshot

- Status: Accepted
- Date: 2026-10-18
- Supersedes: the pool format of [ADR 0010](0010-redis-pools-and-trending.md) (sorted sets rebuilt with ZADD + RENAME). ADR 0010's pool definitions and trending formula still apply, as amended by [ADR 0013](0013-trending-v2.md).

## Context

Two free-tier limits made Phase 2's design unaffordable once Phase 3 adds an API:

1. **Supabase egress.** Supabase Free includes 5 GB a month of uncached egress, and with the shared pooler (Supavisor) the pooler's egress is what counts ([egress docs](https://supabase.com/docs/guides/platform/manage-your-usage/egress), checked 2026-10-18). Every ingestion run began with `store.load_recent`, which selected every article of the last 72 h with its vector: about 14,500 rows × ~1 KB ≈ **14 MB a run**. At 96 runs a day that is **~40 GB a month, 8× the quota**, before the API reads anything.
2. **Upstash commands.** Upstash bills every command inside a pipeline ([billing](https://upstash.com/docs/redis/overall/billing)). Phase 2 rewrote pools with up to 30 commands a run (~86K a month). The API would have needed 15 more per refresh per instance to read them, and SPEC §9 caps our total at 300K.

The API also needs more than ids from ingestion: topic centroids for cold-start readers, public stats, and (P3e) persona summaries. Readers must never pair a new version pointer with an old blob, and Cloud Run instances with min-instances 0 start cold often, so the last good data has to survive ingestion gaps of any length.

## Decision

### The window snapshot (`firstread_ingest/window.py`)

- Each run keeps the last 72 h of articles in a `WindowSnapshot`: id, cluster, text source, the **exact float16 vector** Postgres stores (`halfvec`), canonical URL, fingerprint, first-seen and created times, topic, topic source and confidence, origin, domain, has-image. Saved as an `.npz` with a JSON header (format version, schema SHA, embedding model id, watermark).
- **It is only a cache.** `window.sync` trusts it only when:
  - the file exists, has the current format version, the applied schema (SHA of `schema_migrations`) and the embedding model id; and
  - after fetching rows created since its watermark, **one check query** over the window agrees on `count(*)`, `max(created_at)` and an MD5 of the ids sorted by uuid (byte order on both sides, so no text collation is involved).
  - Anything else (a missing, corrupt or old-format file, a schema or model change, a checksum mismatch) means a **full reload** from Postgres, with the reason recorded in the run summary.
- **Why a delta is exact:** rows are immutable after insert (ADR 0009: clusters are never re-merged), and there is one writer (the `ingest` concurrency group). The checksum catches anything else, such as a local run against the production database.
- **Each run syncs twice:** at the start (normally an empty delta plus the check) and after its insert (the delta is exactly this run's rows, read back as stored). The second snapshot is what the pools are built from and what is saved.
- **GitHub Actions** restores the newest entry with prefix `ingest-window-`, runs, saves a new entry `ingest-window-<run_id>-<attempt>` (cache keys are immutable), and **only after that save** deletes all but the newest 2 entries. GitHub evicts least-recently-used caches above 10 GB per repository ([caching reference](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)); without deletion, ~1.3 GB a day of snapshots would evict the model and CI caches.
- `permissions: actions: write` is granted **on the ingest job only**, for that deletion. The workflow default stays `contents: read`. Both are tested.

### The pool snapshot (`firstread_core/pools.py`, shared with the API)

- One `PoolSnapshot` per run: fresh (24 h, cap 500), 13 topic pools (72 h, cap 300), trending (cap 200, with outlet counts and merged clusters), per-member info (cluster, topic, first-seen, text source), 13 topic centroids (float16) and public stats.
- **One code path** builds it in cron, replay and local runs: fresh and topic pools, centroids and stats come from the window snapshot (zero egress). Trending stays **one SQL aggregate over `story_coverage`**, so every outlet that carried a story counts, dropped duplicates included.
- **Format:** zlib-compressed JSON (no pickle); ids as base64url of their 16 bytes; times as epoch seconds. `SCHEMA_VERSION` and the snapshot's own version are inside the blob. Encoding refuses a pool member without info. A test caps the blob at 256 KB with every pool full of distinct ids (229 KB).
- **Publishing, 3 commands a run:**
  1. `SET pools:blob:{ver} <blob>` with **no TTL**;
  2. `SET pools:ver {ver}`;
  3. `EXPIRE pools:blob:{previous} 120`: a grace period for an instance that read the old version just before the switch.
- The published and pending versions live in `source_cursor` (Postgres, free), so finding the previous blob needs no Redis read. If a run dies after writing its blob but before moving the version, the next run expires that orphan: one extra command, only after a failure. No SCAN.
- The code refuses to plan more than 3 commands (4 with an orphan).
- The pre-P3 `pool:*` sorted sets are removed once, by hand: `firstread-ingest drop-legacy-pools` (one `DEL`). It is outside the per-run path, so the first run can't trip the budget guard.

### Readers (the API, P3d)

- Read `pools:ver` lazily, at most once a minute per instance and inside a request (Cloud Run's request-based billing gives no CPU outside requests). Fetch the blob only when the version changed.
- Accept a blob only if its internal version equals `pools:ver` and its schema version is known. Otherwise keep the last good copy; a cold instance with no copy serves ANN-only until the next publish. Never fall back to per-request SQL.
- **Missing centroids** (a cold instance, or a topic with no recent articles) fall back to the release's population vector.

## Alternatives considered

- **Nearest-neighbour search in Postgres** (send each candidate's vector, get back only its nearest stored neighbour). Ingress is free, so egress would drop as much. But approximate HNSW results change dedupe decisions; up to 2,000 queries a run would land on Supabase Free's shared Nano CPU, which the API also needs; and the pools would still need SQL reads. The runner copy keeps dedupe exact and makes the pools free.
- **Keep the sorted sets and add a version key.** A refresh would cost 15 `ZRANGE`s per instance per version (~43K a month per warm instance) on top of ingestion's 86K, and the API would still have no centroids or stats.
- **A TTL on the blob.** Any ingestion gap longer than the TTL would leave cold instances without personas, stats, centroids or topic pools, so the demo-reader list would be empty. Expiring the *previous* blob only after `pools:ver` moves costs the same number of commands.
- **Delete the previous blob immediately** instead of a 120 s `EXPIRE`. Same cost, but an instance that read the old version a moment before the switch would find nothing.

## Consequences

- **Measured on the 3-day replay (288 runs, ADR 0011's budget):**
  - estimated egress fell from ~14 MB to **84 KB a run** (median; p95 86 KB, max 87 KB), about 0.24 GB a month;
  - Redis fell from ≤ 30 to **3 commands a run**;
  - the blob is **192 KB**;
  - the only full reload was the first run (no file yet).
- **Equivalence** (`make replay FRESH=1 COMPARE=1`; `docs/learn/data/03-replay-equivalence.json`), comparing every run with the pre-P3 SQL paths:
  - dedupe's inputs were identical in all 288 runs, and **0 of 267,877 decisions changed**;
  - the fresh and topic pools were identical in every run.
- A snapshot restored from an older cache entry just costs a bigger delta. A lost cache costs one full reload (~14 MB).
- Changing the snapshot's columns means bumping `FORMAT_VERSION`; changing the blob means bumping `SCHEMA_VERSION`, and the API must understand the new schema before ingestion publishes it.
- Tests: snapshot round trip, delta, every reload trigger, bit-identical inputs to the old `load_recent`, publish order and command count, the failure/orphan path, the TTLs on real Redis, the blob cap, and the workflow's permissions and step order.
