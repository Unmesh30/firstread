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

Credit: GDELT data comes from **the GDELT Project, https://www.gdeltproject.org**. Credit it, with that link, wherever GDELT-derived articles appear.

| | GDELT GKG 2.0 files | Publisher RSS |
|---|---|---|
| What we read | One zip every 15 min of English-language web articles: URL, outlet domain, page title, sharing image, crawl and publication time | Each permitted feed's items: headline, link, time, thumbnail, description |
| What we store | URL, headline, outlet domain (as source name), image URL, times, outlet country (ISO), topic, embedding | URL, headline, source name, image URL (unless the permission is text-only), times, topic, embedding. **Descriptions are never stored**; they only shape the embedding |
| Never stored | Article text, GKG themes, people, quotations, tone | Full text, descriptions, author names |
| Terms | Free for unlimited use with credit | Only feeds whose own terms explicitly allow website display with attribution (ADR 0007); each feed's verdict and terms URL are in `services/ingest/src/firstread_ingest/config/feeds.yaml` |
| Placement | Outlet country from GDELT's 2015–2021 domain lookup (pinned by SHA256) plus reviewed overrides | Set per feed (`geo`) |
| Filters | Blocklist (auto-generated stock tickers, syndicated radio copies), unplaceable domains, English-only, junk/short/stale titles, per-run cap of 50 ranked by story breadth | English-only (per item), junk/short/stale titles |

**Known properties and biases**
- **US coverage leans on GDELT.** The RSS feeds with explicit permission are mostly the BBC, plus tech, science, South America and climate outlets.
- **GDELT skews towards outlets that publish a lot,** and towards US local news. The breadth ranking favours stories many outlets carry, and the per-outlet cap limits any one site.
- **The domain lookup is 2015–2021** and estimates an outlet's country from what it mentions most. It misplaces international outlets (overrides fix the known ones) and doesn't know newer outlets.
- **Titles only for GDELT.** Embeddings from titles cost the ranking model about 2% nDCG@10 relative (ADR 0007).

**Steady state** (3-day replay, 2026-10-12 → 10-15; `docs/learn/data/02-replay.json`)
- About 4,800 GDELT articles a day (the per-run cap of 50 always binds) and about 55 from RSS: **GDELT is ~99% of volume**.
- Stored outlets are 49% US / 51% non-US.
- Topics: world 20%, business 14%, us 10%, sports 8%, entertainment 8%, the rest 4–6% each.
- Press releases removed before ranking: 1.1% of GDELT records (press-wire domains and PR-shaped titles).
- Live topic accuracy: 0.79 (95% CI 0.70–0.88), from LLM-labeled samples spot-checked by a second, independent LLM session and approved by the owner (`docs/learn/data/02-topic-audit-2b.json`).

**Retention** (runs at the end of every ingestion run)
- Articles: 14 days.
- `story_coverage`: 48 h (trending looks at 24 h).
- `ingestion_runs`: 30 days.

This comes to about 227 MB at steady state, within the 400 MB budget.
