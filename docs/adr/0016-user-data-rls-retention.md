# ADR 0016: User data: tables, owner-only access, versioned Redis state, retention and the size guard

- Status: Accepted
- Date: 2026-10-23

## Context

Phase 3 stores readers' data for the first time: followed topics, clicks, hides, events and what each feed page showed. Five constraints shape it:

1. **Supabase owns `auth`.** Since 2025-04-21 no tables or functions may be created in it; foreign keys to `auth.users` are fine ([changelog](https://supabase.com/changelog/34270-restricting-access-on-auth-storage-and-realtime-schemas-on-april-21-2025)). The local pgvector image has no `auth` schema at all.
2. **Data API exposure changed.** New public tables are no longer exposed to Supabase's Data API by default (new projects from 2026-05-30, all projects from 2026-10-30, [changelog](https://supabase.com/changelog/45329-breaking-change-tables-not-exposed-to-data-and-graphql-api-automatically)). Access must be explicit either way.
3. **Articles expire after 14 days** (ADR 0008), but a reader's history must keep the vectors of what they clicked.
4. **Redis loses data:** a TTL, a flush, a replaced instance. Two overlapping event batches must not drop a click.
5. **Supabase Free goes read-only above 500 MB of database size**, ingestion included, and deleted rows free space only after vacuum ([database size](https://supabase.com/docs/guides/platform/database-size)).

## Decision

### Tables (migrations 0007–0011)

| Table | Key points |
|---|---|
| `0007` auth shim | A `DO` block that creates `auth`, `auth.users(id, is_anonymous, created_at)` and `auth.uid()` **only if `auth` doesn't exist**. A no-op on Supabase and the CI image; the CI test asserts every `auth` object is unchanged. Locally the API inserts the shim row on first sight (local/test only). |
| `profiles` | `topics` (⊆ the 13 ids), `hidden_sources` (a preference), `is_anonymous`, `active_persona`, `hist_version`, `last_seen_at` (touched at most once a day per instance) |
| `user_history` | One row per (reader, article): the click-time copy of the article's vector, title, source, topic, cluster and text source, plus its `version`. ≤ 50 per reader, trimmed on write. Copied with `INSERT … SELECT`, so the vector never crosses the wire for the copy. |
| `events` | `id` = the client's `event_id` (idempotency); **no FK to articles** (they expire first); type, scope (hides), position, dwell, request id, model version, client time |
| `served_slates` | Per feed page: request id, slate id, page, model version, article ids and 4 components per item (model, fresh, trend, final), so events can be joined to what was shown (written in P3d) |

- **Every user table cascades from `auth.users`.**
- **Every table has RLS** with an owner-only `select` policy (`user_id = auth.uid()`).
- **On Supabase,** a guarded `DO` block revokes everything from `anon` and `authenticated`, then grants `select` back to `authenticated` only. No role can write through the Data API.
- **The API and ingestion use a role that bypasses RLS** (the owner locally; a dedicated role in P5), and the API scopes every query by the verified `sub`.

### Events (`POST /v1/events`)

- **Validation:**
  - ≤ 100 events per batch;
  - unique ids;
  - per-type rules (dwell needs `dwell_ms` ≤ 1 h, `scope` only on hides, position 0–999, `client_ts` within ±24 h).
  - Events on unknown or expired articles are rejected per item (`unknown_article`).
- **One transaction:**
  1. `insert … on conflict (id) do nothing returning id`;
  2. side effects only for the returned ids;
  3. clicks and likes bump `profiles.hist_version` once per distinct article (the row lock serializes concurrent batches, so versions are unique and consecutive);
  4. the history row is upserted with its version and the history is trimmed to 50;
  5. `hide` with `scope: source` adds the domain to `profiles.hidden_sources`.
- **A retried batch** is all `duplicate` and changes nothing, in Postgres or in Redis.
- **Meanings:**
  - like = an explicit click (joins the history);
  - hide = exclude that article (and its story, P3d), or its source with `scope: source`;
  - impressions feed the 3-day **seen** set (seen means *viewed*);
  - dwell is stored only, for a future re-ranker.

### Redis state: append-only, versioned, rebuildable (`firstread_api/redis_state.py`)

| Key | Content | Writes |
|---|---|---|
| `u:{uid}:hist` (list, 7 d) | entries `(version, article, cluster, topic, text_source, fp16 vector)` | `RPUSH` + `LTRIM -60 -1` + `EXPIRE` |
| `u:{uid}:hidden` (set, 7 d) | hidden article ids, plus sentinel `_` after a rebuild | `SADD` + `EXPIRE` |
| `u:{uid}:seen` (set, 3 d) | viewed article ids | `SADD` + `EXPIRE` |
| `u:{uid}`, `u:{uid}:slate` | user-vector cache, feed slate (P3d) | — |

- **No read-modify-write anywhere,** so two overlapping batches can't lose an update (tested with concurrent requests).
- **History read:** de-duplicated by article (latest version wins), ≤ 50.
- **Consistency:**
  - A rebuild writes the rows from `user_history`, then a **marker** holding `profiles.hist_version`.
  - The list is trusted only if the entries after the last marker run exactly from marker + 1 to the current `hist_version` (or, once the marker is trimmed, form 60 consecutive versions ending there).
  - **A failed push, a lost key or a persona switch shows up as a gap,** and the next feed (P3d) rebuilds. A click that lands mid-rebuild only causes another rebuild.
  - A missing hidden set (no sentinel) is rebuilt from hide events; hidden sources live in `profiles`. **Hides survive Redis loss and the TTL** (tested by flushing Redis).
- **Redis failures after the Postgres commit are logged, not raised:** Postgres has the data, and the version check heals Redis.

### Deleting a reader (`DELETE /v1/me`)

- **Deleted now:** the reader's `served_slates`, `events`, `user_history` and `profiles` rows, plus their 5 Redis keys (one `DEL`, no SCAN). Returns 204.
- **Deleting the Supabase Auth user** needs the service key or the service role, so it waits for **P5**. After that, the cascade makes deletion complete even if the app rows were missed.

### Retention and the size guard (ingestion, every run; the API has no timers)

| Rows | Kept |
|---|---|
| impressions | 7 days |
| other events, served slates | 30 days |
| inactive anonymous readers' app rows (not seen for 30 days) | deleted (Supabase's own anonymous-user cleanup is P5) |
| `user_history` | never expired by time; ≤ 50 per reader, deleted with the reader |

- **The size guard** (`store.guard`, measured before retention) can only shorten these:
  - **> 350 MB:** impressions 2 days;
  - **> 375 MB:** also slates 7 days and other events 14 days;
  - **> 400 MB:** also articles 10 days.
- Each step is recorded in the run summary (`retention.guard_step`, and a `_size_guard` note in the job summary).
- **Vacuum:** after big (≥ 10,000 rows) or tightened deletes, a plain `VACUUM (ANALYZE)` runs on the touched tables outside the transaction. That makes the space reusable so files stop growing; it doesn't shrink them (VACUUM FULL, with a lock, goes in the runbook). **That's why the guard acts at 350 MB, not near 500.**

### Size estimate (ADR 0011's model)

At 100 sessions a day, about 280 MB:
- articles 227 MB + coverage 13 MB;
- events ~22 MB;
- history ~11 MB;
- slates ~5 MB.

The 350 MB guard is reached around 400 sessions a day.

## Alternatives considered

- **Creating `auth` objects through migrations on Supabase.** Forbidden since 2025-04-21. The shim only ever runs where `auth` is absent.
- **Exempting clicked articles from retention** (instead of `user_history`). It couples a reader's history to the article table's size budget, and a 14-day-old article row is bigger than the copy we need.
- **A Redis hash for history with HGET/HSET.** Read-modify-write, so overlapping batches can lose a click; the version check can't catch the batch with the newer version writing last.
- **Deleting the slate on every click.** It loses the served list, so pages could repeat. P3d keeps the slate and re-ranks when the history version moved.
- **Retention in the API.** No CPU outside requests on Cloud Run; ingestion already runs every 15 minutes with database access.

## Consequences

- One click costs 3 Redis commands; a batch of impressions 2; a hide 2 (ADR 0011).
- A Redis loss costs one rebuild per active reader (≤ ~40 KB of egress each), not lost data.
- The CI `supabase` job proves the cascade, owner-only reads, no anon access, no writes, an untouched `auth`, and pgvector ≥ 0.8.
