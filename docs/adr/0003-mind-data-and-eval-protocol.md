# ADR 0003: MIND data handling and the offline evaluation protocol

- Status: Accepted
- Date: 2026-10-13

## Context

Phase 1 trains and evaluates on MIND (Microsoft News Dataset). Three things make this decision worth writing down:

1. **Access.** The official Azure blob links have returned `PublicAccessNotPermitted` since July 2024. The Recommenders project hosts the official zips on Hugging Face (`Recommenders/MIND`).
2. **License.** MIND is under the Microsoft Research License Terms. It is for non-commercial research only (public demonstrations count), it must not be redistributed, and results may be published only if no material portion of the data is included.
3. **Honest numbers.** Offline metrics are only meaningful if the protocol is fixed before anyone looks at results. That means fixing the splits, the slices, tie handling, and when dev is used.

## Decision

### Data
- Download only `MINDsmall_{train,dev}` and `MINDlarge_{train,dev}` from `Recommenders/MIND` at the pinned revision `2115068f457b577f93a268fffe22373e45727d89`. `MINDlarge_test` has no labels and is not used.
- Pin each zip's **SHA256 and size** in `ml/src/firstread_ml/mind/manifest.py`, verify them after download, and delete files that fail.
- The download requires an explicit `--accept-license` flag. Without it, the command prints the license URL and a summary of the terms, then exits 2.
- Everything stays in gitignored `data/`. That covers raw TSVs, parquet, embedding caches and model checkpoints. **Never commit or upload** MIND files, MIND text or MIND-derived embedding caches.
- Committed reports (`ml/reports/`) contain aggregates only: counts, quantiles and metrics. They never contain text or ids.

### Splits (no test-set tuning)
- **holdout** is used for tuning and model selection. We fit on MIND train minus its last calendar day and test on that last day. For MIND-small that is 2019-11-14: fit = 126,695 impressions, holdout = 30,270.
- **dev** is used for final numbers. We fit on all of train with configs chosen on holdout and report dev **once**. Only `make eval` / `make reproduce` evaluate on dev.
- On holdout, everything learned from "the past" is recomputed from the fit days only. That includes popularity, the cold-start article set and the population vector.

### Metrics (per impression, then averaged)
- **AUC**: Mann–Whitney. Tied scores share their average rank, which matches `sklearn.roc_auc_score`.
- **MRR**: `Σ(label/rank) / Σlabel`, as in MIND's evaluation script.
- **nDCG@5, nDCG@10**: gain `2^label − 1`, discount `log2(rank+1)`.
- **Ties** in MRR/nDCG are broken by a seeded random key, not by input order. MIND's script uses argsort, so ties fall in logged order. That can flatter or punish models with many ties, such as popularity on brand-new articles.
- Impressions without at least one click and one non-click are excluded and counted. MIND-small has none.

### Slices
Each slice keeps a subset of every impression's candidates. An impression stays in the slice only if it still has at least one click and one non-click.
- **overall**: all candidates.
- **cold_candidates**: articles never clicked in the training period. That means they are not a positive in any train impression and do not appear in any train user history.
- **cold_users**: impressions whose user has ≤ 2 history clicks, including 0. We report how many have an empty history.
- **category:\<c\>**: candidates in MIND category c. Slices under 100 impressions are flagged as low-n.

### Baselines
- **random**: seeded.
- **popularity**: clicks in train impressions.
- **recency**: first-seen time. MIND's `news.tsv` has no publish time, so first-seen = the earliest impression timestamp in which the article appears as a candidate, across train ∪ test split. This doesn't leak: if an article is a candidate at time *t*, its first-seen is ≤ *t*.
- **mean_history**: cosine between the candidate and the normalized mean of the user's last 50 clicked-article embeddings. An empty history uses the **population vector**, the normalized mean embedding of all training clicks.
- The ID-based matrix-factorization baseline is skipped; SPEC lists it as optional.

### Eval-regression gate (SPEC §13)
- **CI** runs `firstread-ml eval-gate` (also a pytest test) on a **seeded synthetic MIND-format dataset** with a hash embedder. It fails if any model's nDCG@10 drops more than 1% (relative) on overall, cold-candidate or cold-user slices versus `ml/tests/data/eval_gate_expected.json`. This catches regressions in the metrics, slices, features and pipeline code without MIND.
- **Real data:** a local `make eval-gate` on a fixed MIND-small dev subset is added with the model (PR 1b/1c), and the release script requires it to pass. A real-data gate in CI is deferred because it would mean storing MIND where CI can reach it, which is redistribution.

## Alternatives considered

- **The Azure links or a third-party mirror:** the official links are dead, and unofficial mirrors can't be verified. The Recommenders mirror is maintained by the same Microsoft-affiliated project, and pinned hashes make it verifiable.
- **Tuning on dev and reporting on dev:** this is common in papers, but it inflates numbers. MIND test labels aren't public, so holding out train's last day is the honest substitute.
- **Random holdout instead of time-based:** a random split leaks the future (popularity and recency would see "tomorrow"). News is strongly time-dependent.
- **Input-order tie-breaking (MIND script behaviour):** this is not neutral for tie-heavy baselines.
- **Committing a small MIND sample for CI:** it would violate the no-redistribution term.

## Consequences

- Our dev numbers are **not directly comparable** to MIND leaderboard numbers (those are on the test set). Reports say so.
- Anyone can reproduce the numbers with `make mind-download … ACCEPT_LICENSE=1` followed by `make eval`, because the data is pinned by hash.
- The CI gate protects the code, not data-dependent quality. Quality regressions on real data are caught locally before a release.
