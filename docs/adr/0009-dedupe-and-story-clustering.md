# ADR 0009: Exact and near-duplicate detection and story clustering

- Status: Accepted
- Date: 2026-10-14

## Context

The same story reaches us many times: wire copies on dozens of sites, the same article on bbc.co.uk and bbc.com, and different write-ups of the same event. The feed (P3) shows at most 2 items per story (SPEC §7.8), and trending ranks stories by how many outlets cover them (ADR 0010). Ingestion therefore has to drop copies and group the rest into stories, without merging different stories: a false merge hides news, while a missed merge only shows two versions.

GDELT items are embedded from the **title only**; RSS items from title + description. Vectors of the same story are less similar across those two kinds of text.

## Decision

For each new item, in order (`firstread_ingest/dedupe.py`, pure numpy):

1. **Exact duplicate**, which is not inserted:
   - the same canonical URL (`firstread_core.urls`: https, no `www.`/`m.`, no tracking params including BBC's `at_*`, sorted query, AMP removed, `bbc.com` aliased to `bbc.co.uk`); or
   - the same **title fingerprint** (SHA1 of the NFKC, casefolded alphanumeric tokens) within 72 h.
2. **Near-duplicate**, which is not inserted: cosine to the nearest article of the last 48 h (stored, or earlier in the batch) ≥ **τ_dup** for that pair type.
3. **Same story:** τ_story ≤ cosine < τ_dup. The item is inserted into the neighbour's `story_cluster_id`.
4. Otherwise a **new story** (cluster id = its own id).

Every item carries its story cluster, including dropped duplicates, so `story_coverage` can count every outlet.

**Thresholds by pair type, validated on 291 hand-labelled live pairs (2026-10-14).** Pairs were sampled by `scripts/sample_pairs.py` (nearest neighbours, stratified by pair type and 0.025 cosine bin), labelled duplicate / same story / different, and scored by `scripts/score_pairs.py`. A threshold is accepted when **every** bin at or above it has ≥ 95% precision. Aggregates are in `docs/learn/data/02-thresholds.json`.

Labelled pairs by type and label:

| Pair type | Pairs | Duplicate | Same story | Different |
|---|---:|---:|---:|---:|
| GDELT↔GDELT (title↔title) | 144 | 28 | 85 | 31 |
| GDELT↔RSS (title↔title+description) | 96 | 2 | 91 | 3 |
| RSS↔RSS (title+description both) | 51 | 0 | 25 | 26 |
| **Total** | **291** | **30** | **201** | **60** |

Thresholds (τ_dup / τ_story):

| Pair type | Data supports | **Used** | Status |
|---|---|---|---|
| GDELT↔GDELT | 0.975 / 0.875 | **0.975 / 0.875** | Validated |
| GDELT↔RSS | 0.925 / 0.725 | **0.95 / 0.80** | **Provisional** (re-check in 2b) |
| RSS↔RSS | none / 0.775 | **0.97 / 0.875** | **Provisional** (re-check in 2b) |

**Why the two provisional types are set above what the data supported:**

- **GDELT↔RSS, τ_story 0.80 instead of 0.725.**
  - The 96 pairs come from very few stories. The cross-source neighbours that day were dominated by about six big stories (the Pope's surgery, Netflix layoffs, Ukraine's Starlink access, Liverpool v LASK, Man City v PSG, Sally Rooney's books).
  - Pairs inside one big, well-covered story are "same story" at low cosines, so precision at 0.725–0.80 is flattered and wouldn't hold for ordinary days with many small stories.
  - Only 3 "different" pairs were seen, so the 95% bar was never really tested below 0.80.
- **GDELT↔RSS, τ_dup 0.95 instead of 0.925.** Only **2** duplicates were labelled, too few to validate any duplicate threshold. 0.95 sits between the GDELT↔GDELT value and what the data hinted.
- **RSS↔RSS, τ_story 0.875 instead of 0.775.**
  - Only 51 pairs, from 9 feeds and mostly one publisher (BBC); the bins above 0.775 hold 1–5 pairs each, and the 0.85 bin has one "different" pair out of two.
  - 0.875 is the lowest value with every non-empty bin above it fully "same story".
- **RSS↔RSS, τ_dup 0.97.** **No** duplicates were labelled (identical BBC items are caught earlier by URL and fingerprint). 0.97 keeps near-duplicate drops close to "the same text".

In every case the values were **raised, never lowered**: a missed merge shows two versions of a story, while a false merge hides one.

**Re-check in 2b**, after several days of ingestion with more varied news:
- re-run `sample_pairs.py` with more files and a per-story cap, so a few big stories can't dominate;
- target at least 100 pairs and 10 duplicates per provisional type;
- re-score, and update `VALIDATED_BY_PAIR` and this ADR.

Values are only ever raised from what the data allowed, never lowered. They live in `dedupe.VALIDATED_BY_PAIR`; `INGEST_TAU_DUP` / `INGEST_TAU_STORY` set the fallback.

## Alternatives considered

- **One global threshold.** The data shows title-only vs title+description pairs sit lower for the same story (cross-type story pairs were already 100% precise from 0.725, title↔title only from 0.875), so one number either over-merges GDELT copies or under-clusters across sources.
- **MinHash / shingle Jaccard on titles.** It catches reworded copies poorly (paraphrased headlines share few shingles) and needs separate tuning. We already have a semantic vector for every item.
- **Full transitive clustering (union-find, HDBSCAN).** Chains drift: A~B and B~C merges A and C even when they're far apart. Nearest-neighbour assignment within 48 h bounds the drift.
- **pgvector ANN in SQL for the neighbour search.** The 48 h window is about 10K vectors, so an exact numpy matmul is faster than a round trip and easy to test.

## Consequences

- About 1,000 of 6,000 items in a typical first run are exact duplicates, mostly the same GDELT URL across files and wire copies.
- Publisher suffixes that don't match the outlet's domain ("| Namoi Valley Independent", "– NBC4 Washington") defeat the fingerprint, and such pairs fall to cosine (≥ 0.97 in the labels, so they are still caught as near-duplicates).
- The labels are **LLM-labeled**, on one day's news. They were **spot-checked by a second, independent LLM session, approved by the owner**:
  - all 30 sampled pair labels held;
  - pairs #16 (Pakistan–Oman) and #71 (Holiday Train MN vs Ontario) are ambiguous, but both sit below every threshold;
  - every sampled pair at or above its type's τ_story was same story or duplicate (14/14), and both pairs above the GDELT↔GDELT τ_dup were true duplicates.
- The thresholds should be re-validated when sources change; the scripts make that a one-hour job.
- Only nearest-neighbour clustering: a story's cluster is set by its first close neighbour and never re-merged later.
