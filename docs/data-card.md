# Data card

What data FirstRead uses, where it comes from, and what we never publish.

## MIND (Microsoft News Dataset): training and offline evaluation only

| | |
|---|---|
| What | About 160K English news articles (ID, category, subcategory, title, abstract, URL, entities) and impression logs from Microsoft News, collected October–November 2019. Users are anonymized ids. |
| Paper | Wu et al., "MIND: A Large-scale Dataset for News Recommendation", ACL 2020. https://msnews.github.io |
| Where we get it | `Recommenders/MIND` on Hugging Face, revision `2115068f457b577f93a268fffe22373e45727d89`. Each zip is SHA256-pinned in `ml/src/firstread_ml/mind/manifest.py`. The official Azure links have been dead since July 2024. |
| Splits used | `MINDsmall_train`, `MINDsmall_dev`, `MINDlarge_train`, `MINDlarge_dev`. `MINDlarge_test` is unused because it has no labels. |
| License | [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf): non-commercial research (public demonstrations count); **no redistribution**; results may be published only if no material portion of the data is included. |
| How to get it | `make mind-download VARIANT=small ACCEPT_LICENSE=1` (or `large`). The flag is required; without it the command prints the terms and exits. |
| Where it lives | `data/mind/` (gitignored): raw TSVs, parquet in `processed/`, and embedding caches in `data/cache/emb/`. |

### What we never do with MIND
- Commit, upload or share MIND files, MIND article text, or anything derived row by row (parquet, embedding caches, checkpoints) outside `data/`.
- Show MIND articles in the app.
- Put ids, titles or abstracts in reports. `ml/reports/` contains aggregates only (counts, quantiles, metrics, category names).

### What we do publish
- Aggregate statistics (`ml/reports/eda-mind-*.md`) and evaluation metrics (`ml/reports/results-*.md|json`).
- From Phase 1c: model weights trained on MIND, released on Hugging Face under a **non-commercial** license with a model card saying they are MIND-derived. They are not MIT.

### Known properties and biases (MIND-small; see `ml/reports/eda-mind-small.md`)
- Six days of train impressions (2019-11-09 to 11-14) and one day of dev (11-15).
- Two categories dominate: news (31%) and sports (30%). Four categories have fewer than 25 articles each.
- 52% of dev candidate rows are articles never clicked in training, so cold start is the normal case, not an edge case.
- 7% of dev impressions come from users with ≤ 2 history clicks; 3% have none.
- US-centric 2019 news from one publisher platform. Models trained on it inherit its topical and editorial skew.

## Live news (Phase 2): GDELT and publisher RSS
Documented in Phase 2. GDELT must be credited with a link to https://www.gdeltproject.org wherever its data appears.
