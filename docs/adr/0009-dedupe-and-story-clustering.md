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

| Pair type | n | Data says (τ_dup / τ_story) | **Used** | Why it differs |
|---|---:|---|---|---|
| GDELT↔GDELT (title↔title) | 144 | 0.975 / 0.875 | **0.975 / 0.875** | — |
| GDELT↔RSS (title↔title+desc) | 96 | 0.925 / 0.725 | **0.95 / 0.80** | Only 2 duplicates labelled; the cross pairs were dominated by about 6 big stories, which flatters low-cosine precision |
| RSS↔RSS (title+desc both) | 51 | — / 0.775 | **0.97 / 0.875** | No duplicates labelled and thin bins; kept strict |

Values are only ever raised from what the data allowed, never lowered. They live in `dedupe.VALIDATED_BY_PAIR`; `INGEST_TAU_DUP` / `INGEST_TAU_STORY` set the fallback.

## Alternatives considered

- **One global threshold.** The data shows title-only vs title+description pairs sit lower for the same story (cross-type story pairs were already 100% precise from 0.725, title↔title only from 0.875), so one number either over-merges GDELT copies or under-clusters across sources.
- **MinHash / shingle Jaccard on titles.** It catches reworded copies poorly (paraphrased headlines share few shingles) and needs separate tuning. We already have a semantic vector for every item.
- **Full transitive clustering (union-find, HDBSCAN).** Chains drift: A~B and B~C merges A and C even when they're far apart. Nearest-neighbour assignment within 48 h bounds the drift.
- **pgvector ANN in SQL for the neighbour search.** The 48 h window is about 10K vectors, so an exact numpy matmul is faster than a round trip and easy to test.

## Consequences

- About 1,000 of 6,000 items in a typical first run are exact duplicates, mostly the same GDELT URL across files and wire copies.
- Publisher suffixes that don't match the outlet's domain ("| Namoi Valley Independent", "– NBC4 Washington") defeat the fingerprint, and such pairs fall to cosine (≥ 0.97 in the labels, so they are still caught as near-duplicates).
- The labels are one person's judgement on one day's news. The owner spot-checks 30, and the thresholds should be re-validated when sources change. The scripts make that a one-hour job.
- Only nearest-neighbour clustering: a story's cluster is set by its first close neighbour and never re-merged later.
