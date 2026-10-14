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
