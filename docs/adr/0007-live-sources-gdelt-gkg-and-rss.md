# ADR 0007: Live sources: GDELT GKG files and RSS feeds with explicit permission only

- Status: Accepted
- Date: 2026-10-14

## Context

SPEC §6 named the GDELT DOC 2.0 API as the primary live source and curated publisher RSS as the secondary one. Phase 2 had to verify both before building on them.

**GDELT DOC 2.0 API (probed 2026-10-14, `services/ingest/scripts/probe_gdelt.py`).**
- `timespan=15min` and an explicit 15-minute `startdatetime`/`enddatetime` window both return **HTTP 200 with the plain-text body `Timespan is too short.`**, not JSON.
- Every other request was throttled: **10 of 10 calls returned HTTP 429**, spaced 6 minutes apart over an hour, even after multi-minute cooldowns. The body says "Please limit requests to one every 5 seconds". Responses took 9–28 s and carried no `Retry-After` header.
- The API is unusable from this IP, and GitHub runners can't be tested until P5.

**GDELT GKG 2.0 files (`probe_gkg.py`).** `lastupdate.txt` lists one zip per 15 minutes (about 7 MB). The probe file had:
- 1,680 records; 99.8% with a page title (`<PAGE_TITLE>` in the Extras field), 90% with a sharing image, 100% with a URL;
- no translation info, because the main GKG stream is English-language sources;
- 84% of records on generic TLDs, so a TLD filter can't place outlets;
- GDELT's [domain → country lookup (2015–2021)](https://blog.gdeltproject.org/?p=14261) covers 97.9% of records. Its estimate (the country an outlet mentions most) misplaces international outlets that cover the US heavily, e.g. aljazeera.com → US.
- It is also noisy: auto-generated stock-ticker sites, and iheart.com radio-station copies (9% of records).

**Publisher RSS terms (checked on each publisher's own pages, 2026-10-14).**
- Most allow *personal, non-commercial* use only (NYT, Guardian, NPR, NBC, PBS, Al Jazeera).
- Some forbid website display outright (RNZ; CBS News and CBS Sports).
- Only a few explicitly allow showing headlines on a website with attribution: BBC, TechCrunch, ScienceDaily, MercoPress, and Yale Climate Connections (text only; its images are licensed from others).

## Decision

**GDELT: the GKG files, not the DOC API.** `services/ingest/src/firstread_ingest/sources/gkg.py`:
- **Catch-up:** a cursor (`source_cursor` table) records the last processed slot. Each run processes every 15-minute slot since then, at most 8 (2 h), with file names derived from the clock, because scheduled runs are often late or skipped.
- **Fresh index:** `lastupdate.txt` is fetched with `Cache-Control: no-cache` (a cached copy can be hours old). HTTPS throughout. The newest file is MD5-checked against `lastupdate.txt`.
- **Index ahead of files:** `lastupdate.txt` can list slots whose files still 404. Seen 2026-10-14: slots 22:30–23:15 404'd for over 40 minutes while listed. A 404 younger than 90 minutes is *pending*: later slots are still processed, but the cursor stops before it so the next run retries it. Older 404s are treated as skipped slots.
- **Staleness:** if the newest available file is over an hour old, the run summary marks GDELT as stale.
- **Placement:** an outlet's country comes from GDELT's lookup (pinned by SHA256, cached), overridden by `config/gdelt_domains.yaml` for known misplacements (e.g. aljazeera.com → Qatar, reuters.com → UK). Unplaceable domains are dropped. **Every placeable English-language outlet is kept**, worldwide: world coverage comes through GDELT, and the topic geo rule (ADR 0006) handles non-US items.
- **Blocklist** in the same file: the stock-ticker network and iheart.com syndication.
- **Titles** are HTML-unescaped before suffix stripping and fingerprinting (GDELT escapes non-ASCII).
- **Per-run cap (database budget, ~3.2 KB per article):** GKG supplies about 1,500 placeable articles per file. `select.py` keeps 50 per run, ranked by **breadth** (distinct outlets carrying the story in this run, counting duplicates), then recency, with at most 3 per outlet and a **topic floor** of 2 per topic. The newest 2,000 candidates are embedded per run (CPU bound). The top 20 outlets are logged per run, so new spam shows up.
- **Coverage:** every placeable record that joins a stored story gets a `story_coverage` row, whether inserted or not, so trending sees GDELT's full breadth (ADR 0010).

**RSS: only feeds whose own terms explicitly allow website display.**
- "Personal, non-commercial" means a person's own reading, not a public app, and App Store guideline 5.2.2 requires explicit permission for third-party content.
- `config/feeds.yaml` records each feed's `terms` verdict and `terms_url`. The loader refuses to enable a feed without `website_with_attribution`.
- Enabled (9): BBC (World, Technology, Health, Entertainment & Arts, Sport), TechCrunch, ScienceDaily, MercoPress, Yale Climate Connections (`allow_images: false`).
- 44 feeds stay disabled with their reason: personal/non-commercial terms (14), terms forbidding website use (3), no explicit permission found (27).
- The app must show each article's source name (P4).
- **Descriptions are used in memory for the embedding and never stored** (SPEC §4).

**No publisher page fetching** (owner decision). Title-only GDELT articles are fine for ranking: the skew study measured a 2.06% relative nDCG@10 loss when history and candidates are both title-only (`ml/reports/skew-title-only-mind-small.md`).

## Alternatives considered

- **Keep trying the DOC API, or call it at most once per run.** Every probe call was throttled, and its free-text error contract (HTTP 200 + plain text) is fragile. It could be revisited from GitHub runners in P5; the GKG adapter doesn't depend on it.
- **Only our original 6 countries in GKG.** That keeps 86% of records but starves world coverage once RSS is restricted to explicit permission. Rejected after review.
- **Treating "personal, non-commercial" feeds as allowed.** Rejected by the owner: a public app isn't personal use.
- **Asking publishers for permission.** Possible later; any approval just flips that feed's verdict.
- **Fetching publisher pages for summaries.** Not needed (2% skew cost), and the owner did not approve it.

## Consequences

- Volume: up to 50 GDELT + a few dozen RSS articles per run, about 5,000 a day, within the 400 MB budget at 14-day retention (measured 3.2 KB per article, HNSW included).
- US domestic and politics coverage now comes mostly from GDELT, because the permitted RSS feeds are mostly UK and international. The topic floor keeps thin topics alive; more permitted US feeds would help.
- GDELT's domain lookup is from 2015–2021; new outlets are "unplaceable" until added to the overrides (117 of 6,529 records in the probe run).
- The blocklist and overrides need occasional review, using the per-run top-domains log.
- GDELT must be credited, with a link to gdeltproject.org, wherever its data appears (data card, app About page).

## Amendment (2026-10-15, Phase 2b): press releases, a per-story cap, the pending horizon, replay findings

**Press releases are dropped before dedupe, ranking and trending.** Syndicated releases appear on many outlets at once and would otherwise win the breadth ranking and trending. Two filters:
- **Press-wire domains** (`gdelt_domains.yaml` → `press_wires`): prnewswire.com, globenewswire.com, businesswire.com, accesswire.com, einpresswire.com, newsfilecorp.com, pr-inside.com, finanznachrichten.de, equitybulls.com, prweb.com, newswire.ca, openpr.com.
- **PR-shaped titles** (`announcement_titles.yaml`): a fiscal period plus a year before "results"; "financial results"; *announced* conference/earnings calls; declared dividends; filings; the "Company Inc.: Company Announces…" prefix. News *about* results or calls stays, and is tested both ways.
- In the 3-day replay they removed **3,444 + 366 = 3,810 of 344,711 GDELT records (1.1%)**:

| Rule | Records |
|---|---:|
| `results_period_year` | 175 |
| `company_prefix` | 92 |
| `call_announced` | 60 |
| `dividend_declared` | 16 |
| `financial_results` | 15 |
| `dividend_change` | 4 |
| `call_and_webcast` | 4 |

**At most 3 stored articles per story** (`INGEST_GKG_MAX_PER_STORY`), counting ones already stored in the 48 h window. The first replay showed breadth ranking spent **26%** of inserts on a story's 3rd-or-later copy, which the feed never shows. With the cap it is 2.4%, and stories stored rose from 9,444 to 13,065 for the same 14,400 articles. Coverage still records every outlet, so trending breadth is unaffected.

**Pending horizon = the catch-up window (2 h).** The four slots that 404'd on 2026-10-14 were all present hours later, so they were delays, not gaps. Slots older than 2 h can't be retried anyway, because catch-up starts at most 8 slots back.

**Replay findings** (2026-10-12 02:30 → 10-15 02:15, 288 slots; `docs/learn/data/02-replay.json`):
- **Volume:** 50 GDELT inserts on every run (the cap always binds), out of ~750 candidates per run after dedupe. RSS adds ~55 a day, measured from one live fetch (RSS can't be replayed), so **GDELT is ~99% of volume**, more than the ~90% assumed.
- **Size:** 3,153 B per article; ~227 MB for 14 days, within the 400 MB budget.
- **Top domains** are dominated by blocklisted syndication (iheart.com) and US local TV/radio. The blocklist removed 27,538 records, and 6,020 were unplaceable.
- **Trending breadth favours UK local-news networks:** one PA wire story republished across dozens of UK local sites counts as dozens of outlets. An ownership-group weighting is a P3/P6 candidate.
- **Section pages** stored as articles ("Health - Carroll Broadcasting Inc.", seen in the 2b threshold sample) are dropped as `section_page`: a section name on one side of the separator, and an outlet or company name on the other (`junk_titles.yaml` → `sections`, `outlet_words`).

**Known limitations** (not fixed in Phase 2):
- **Corporate-milestone press releases pass the filter.** Titles like "QuickMD Surpasses 2 Million Patient Visits" have no PR-wire domain and no fiscal-results or dividend shape. They can rank like news and, when syndicated, trend. A learned press-release detector, or more patterns (e.g. "surpasses … users/visits/customers"), is a later option.
- **Local-TV suffixes aren't stripped.** Titles like "… – NBC4 Washington" or "… – NBC 7 San Diego" don't match the outlet's domain, so the suffix stays. Syndicated copies of one story then differ in title fingerprint and cosine, and **cluster as the same story instead of dropping as duplicates**. The per-story cap (3) limits the impact: at most 3 copies are stored, and coverage still counts every outlet.

## Follow-up (2026-10-15, after the first workflow run on `main`)

- **Yale Climate Connections disabled** (`enabled: false`). It answers **HTTP 403 to GitHub-hosted runners** (it works from a home IP). Ingestion runs on GitHub Actions in P5 too (Cloud Run only hosts the API), so this won't clear up there, and a feed that fails every run would make every run `partial` and hide real problems. **8 RSS feeds remain enabled.** Weather stays covered by the GDELT topic floor.
- **`indiankanoon.org` blocklisted:** a court-records database, not news. Its templated "X vs The State Of Y on <date>" titles also skewed the story-threshold sample (ADR 0009).
- **The run summary's top-20 outlets are counted after the blocklist and country filter,** so new spam stands out instead of known blocked outlets (iheart.com used to top the list).
