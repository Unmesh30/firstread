# ADR 0018: Demo personas, "Why this?" and the public stats

- Status: Accepted
- Date: 2026-10-23

## Context

SPEC §2 promises recruiters a one-tap demo ("Try a demo reader"), a "Why am I seeing this?" for every item, and a System page with live numbers. Four constraints shape them:

1. **Personas must be built from live news, never MIND rows** (MIND's research license; SPEC §8).
2. **The public endpoints get the most anonymous traffic.** A recruiter link shared widely hits the persona list and the System page first, so they must cost no Postgres egress (ADR 0011).
3. **A real reader's history and topics must never be replaced silently** by a demo.
4. **"Why this?" must say what the ranking actually did,** with the same code (ADR 0014's `reasons()`), not a separate story.

## Decision

### Personas (`config/personas.yaml`, `firstread_ingest/personas.py`, migration 0012)

- **Five personas:** Markets & tech (business, tech_science), Sports & health, World affairs (world, politics), Culture & food (entertainment, food_drink, lifestyle), U.S. news (us, politics, weather).
- **Refresh:** by the ingest run, in its first run after 06:00 UTC, whenever the last refresh is more than 24 h old (a skipped run can't leave them stale), or when the config changes.
- **Selection, deterministic, from the runner-side window** (ADR 0012), so no vectors are read from Postgres:
  - the last 48 h;
  - confidently placed topics only: `category_source = model` with confidence ≥ 0.50, or an RSS feed's hint;
  - titles of ≥ 6 words (checked in SQL for the best 300 candidates per topic: ids in, ids out);
  - one article per story: not the same `story_cluster_id`, and not ≥ τ_feed to any article already picked;
  - ≤ 3 per domain;
  - best first by the story's trending score, then recency, then id;
  - 20 per persona, split across its topics and interleaved; a topic short of articles leaves its places to the others.
- **Storage:** `demo_personas` and `persona_articles`. The articles are copied with `INSERT … SELECT` from `articles`, vector included, so a persona outlives the 14-day article retention. Both tables are service-only: RLS on, no policy, and no grants at all on Supabase.
- **In the snapshot:** each run reads the 5 summaries (~2 KB: name, topics, article count, refresh time, 3 sample headlines) and puts them in the pool blob. The blob with five summaries at every cap is 229,875 bytes, under the 256 KB test cap.

### Activation (`POST /v1/demo/personas/{id}/activate?replace=false`)

- **Guard:** a reader with no active persona **and** their own history or topics gets **409 `history_exists`** unless `replace=true`. The P4 app asks first.
- **No flag needed** for a brand-new reader, or to switch from one persona to another (even after clicking a few things as that persona).
- **In one transaction** (the profile row locked): `user_history` is replaced by the persona's articles, copied server-side with `kind = 'persona'` and fresh contiguous versions above the old `hist_version` (the best article gets the newest); `topics` and `active_persona` are set.
- **Kept:** hidden sources and hidden articles. They are the reader's own choices.
- **Redis:** `DEL u:{uid}:hist u:{uid}:slate`, 1 command, best effort like the topics route. If it fails, the version gap makes the next feed rebuild the history, and the stale slate is re-ranked (ADR 0016).

### `GET /v1/demo/personas` and `GET /v1/stats` (public)

- **Served from the in-process pool snapshot:** zero Postgres queries, tested by counting them. `Cache-Control: public, max-age=60`.
- **`/v1/stats`:** last run time and status, runs and success rate over 24 h, articles and sources in 24 h, median freshness lag (stored − first seen), the snapshot's build time, the trending and persona counts, the model version, the instance's uptime, and the GDELT credit `{text, url: https://www.gdeltproject.org/}` (SPEC §4).
- **No snapshot** on a cold instance with Redis down: 503 with `Retry-After`, never a SQL fallback.

### `GET /v1/articles/{id}/why`

- **Recomputes, for the caller only,** what the feed would give this article now. It doesn't read a slate:
  - the reader's state: `HMGET` user-vector cache + `LRANGE` history (**2 Redis commands**) ∥ the profile row. A history that fails the version check, or Redis being down, is read from Postgres instead; `/why` never writes Redis.
  - the user vector through `FeedPipeline.user_vector` (cache, batched encoder, cold-start blend);
  - the article from the LRU (or 1 row); trend score and outlet count from the snapshot;
  - `blend()` and the same `reasons()` as the feed;
  - the two nearest history items, with their titles (one small query).
- **One difference from the feed, documented:** the trend term is a percentile within a request (ADR 0014). The feed ranks it among that request's candidates; `/why` ranks it among the reader's pool candidates (fresh, trending, followed topics), the bulk of the same set. The test checks `final` against the feed's score within 0.002 and that the reasons follow from the returned components.
- `diversity` can't be recomputed outside a slate, so `/why` never shows it.

## Alternatives considered

- **Personas from MIND users.** Not allowed (license), and MIND articles can't be shown.
- **Hand-picked persona articles.** Stale within a day, and someone has to pick them.
- **Personas as a SQL query per activation.** Each activation would scan 48 h of articles on Supabase Nano. Selecting once a day on the runner costs nothing per visitor.
- **Persona summaries or stats read from Postgres per request.** The most anonymous traffic would carry Postgres egress; the blob already reaches every instance.
- **`/why` from the served slate** (`served_slates.components`). Cheaper, but it can only explain items already shown, and a stale slate explains an older ranking. The plan (and the owner) chose a recomputation for the caller.
- **Merging a persona into the reader's own history.** Mixing a demo with real clicks makes neither meaningful; replacing behind an explicit flag is clearer.

## Consequences

- A recruiter can open the persona list and the System page without the API touching Postgres.
- Activation costs one Redis command plus, on the next feed, one history rebuild (20 rows with vectors, about 20 KB of egress, estimated).
- The persona replay report (`services/api/scripts/persona_replay.py`, `docs/learn/data/03-personas.json`) exercises personas, story grouping, freshness and the new-reader check with the real pipeline on three replayed days.
