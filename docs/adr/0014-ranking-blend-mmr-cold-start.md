# ADR 0014: Ranking: blend, normalization, MMR with a story cap, cold-start blend

- Status: Accepted. **The rules below were committed before any result was computed;** the Results section is added in a later commit and must not change them.
- Date: 2026-10-22

## Context

The feed ranks candidates with (SPEC §7.8):

`final = w_model·model_score + w_fresh·freshness(age) + w_trend·trending − penalties`

It then applies MMR diversity with at most 2 items per story, and gives cold-start readers a topic vector (SPEC §7.4). The constants have to come from data, and the same code must run offline and in the API (CLAUDE.md golden rule). ADR 0003 fixes the protocol: tune on the MIND holdout, report dev once.

MIND is an imperfect stand-in for live ranking. It has:
- no publish time (first-seen stands in, ADR 0003);
- no outlet breadth (clicks stand in for trending);
- no stated topics (revealed clicks stand in);
- curated impressions of ~37 candidates rather than a pool of ~1,000.

The rules say how each proxy is built and record the mismatches as limitations.

## Decision: the code (all in `packages/core/src/firstread_core/ranking/`)

| Function | Definition |
|---|---|
| `freshness(age_h, half_life_h)` | `2^(−max(age, 0)/half_life)` |
| `normalize(x, method, offsets=None)` | Per request (per impression offline). `raw`: x. `zscore`: (x − mean)/std, 0 when std = 0. `percentile`: (average rank − 0.5)/n, ties share their average rank. |
| `trend_feature(raw, offsets=None)` | `percentile(raw)` per request, 0 where raw ≤ 0 |
| `blend(model, fresh, trend, cfg)` | `normalize(model) + w_fresh·fresh + w_trend·trend` (w_model = 1) |
| `mmr(relevance, vectors, λ, groups, cap, k)` | Relevance min-max scaled to [0, 1] in the request. Greedy: pick argmax of `λ·rel_i − (1 − λ)·max_{j picked} cos(i, j)`, skipping items whose story group already has `cap` picks. Returns the picked order and the capped-out items. |
| `cold_start_vector(centroids, topics, click_vec, n, k, population)` | `normalize(w·c_topics + (1 − w)·u_click)` with `w = k/(k + n)` and `c_topics = normalize(mean of the chosen topics' centroids)`. No centroid or no topics: `c_topics = population`. n = 0 and no click vector: `c_topics`. |
| `reasons(...)` | Structured reasons from the actual components (P3d/e use them) |

- **User vectors** are L2-normalized before scoring, so `model = cos(u, article)`.
- **Hidden articles and sources are hard filters,** not penalties: there is no data to tune a penalty.
- **Ranking parameters live in one versioned file,** `ranking/rank-1.yaml` (package data). A test pins each version's SHA-256, so changing a parameter forces a new version.
- **Every API response carries `model_version = "<model tag>+<ranking version>"`,** e.g. `v0.2.0+rank-1`.

## Decision: the tuning protocol (pre-registered)

### Data and encoder

- **Tuning:** MIND-small **holdout** (fit = train minus its last day, holdout = 2019-11-14, 30,270 impressions).
- **The encoder for tuning** is NRMS-lite seed 0 trained on the **fit days only** (the cached P1 holdout checkpoint, config `ml/configs/nrms_lite.yaml`).
  - Not the shipped v0.2.0 encoder: that one was trained on MIND-large train, which includes the holdout day, so its scores there would be inflated and would pull the weights toward the model.
- **Article text:** title + abstract (the P1 cache). Live GDELT items are title-only. The skew study found the encoder's relative order unchanged (ADR 0006 / skew report), but the blend weights are not re-tuned for titles.
- **Ties** are broken by a seeded random key, as in ADR 0003.

### Proxies (offline → live)

- **Freshness:** age = impression time − the article's first-seen time (earliest candidate appearance across fit ∪ test, ADR 0003). Live: now − `first_seen_at`.
- **Trending:** raw = `Σ_{clicks of the article strictly before t and after t − 24 h} 2^(−(t − t_click)/6 h)`, over clicks in every fit and test impression. Live: distinct outlets in 24 h with the same 6 h half-life (ADR 0010/0013).
  - Both go through `trend_feature` (a per-request percentile), so **only their ranks are aligned.**
  - **Limitation:** clicks measure reader demand and outlets measure editorial supply.
- **Stated topics (cold start):** a user's top 3 app topics (through `ml/configs/topic_map.yaml`) among their clicks in **fit-period** impressions, for users with ≥ 3 such clicks. These are revealed, not stated, interests, so the gain is optimistic.
- **Topic centroids:** offline, `firstread_core.pools.topic_centroids` over the embeddings of every article that appears in a fit-period impression (candidate or history), labelled through the topic map. Live: the same function over the last 72 h of live articles (ADR 0012). **Limitation:** MIND centroids ≠ live centroids.

### Step 1: blend and normalization

- **Grid:** normalization ∈ {raw, zscore, percentile} × half-life ∈ {3, 6, 12, 24, 48} h × w_fresh ∈ {0, 0.025, 0.05, 0.1, 0.2, 0.4} × w_trend ∈ {0, 0.025, 0.05, 0.1, 0.2, 0.4}: **540 configs**, on all holdout impressions, without MMR or cold-start topics.
- **Objective:** J = ½ (nDCG@10 overall + nDCG@10 cold candidates).
- **Rule:**
  1. Drop configs whose cold-user nDCG@10 is more than 1% (relative) below model-only (raw normalization, w_fresh = w_trend = 0).
  2. Among the rest, take the best J.
  3. Among configs within 0.001 of that J, take the smallest w_fresh + w_trend.
  4. Then prefer raw < zscore < percentile, then the shorter half-life, then the higher J, then the smaller w_trend. The last two were added before any result, when a unit test showed the first three could leave a tie.
- **If w_fresh = 0 wins, it ships.** The 72 h candidate window still bounds staleness, and P3e reports the share of page-1 items under 2 h old per persona as a separate product check.

### Step 2: MMR with the story cap

- **Setup:**
  - relevance = Step 1's final score;
  - similarity = cosine of the article embeddings;
  - story groups = `firstread_core.stories.group` at τ_feed (MIND text is title + abstract, so the title_summary↔title_summary threshold applies; each article is its own cluster);
  - cap = 2;
  - capped-out items are **appended after** the MMR order (metrics need a full ranking; live drops them).
- **Grid:** λ ∈ {1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5}.
- **Report:** nDCG@10 (overall, cold candidates) and ILD@10 = mean pairwise (1 − cos) of the top 10.
- **Rule:** the smallest λ whose overall nDCG@10 is ≥ 99% of λ = 1.0's.

### Step 3: the cold-start blend constant k

- **Population:** holdout impressions whose user has stated-topic proxies, with Steps 1–2's choices. Report by history length bucket: 0, 1–2, 3–5, 6–10, 11–50.
- **Grid:** k ∈ {0, 0.5, 1, 2, 3, 5, 10, 20}.
- **Rule:** maximize nDCG@10 on cold users (history ≤ 2) subject to overall nDCG@10 (on the same population) ≥ 99.5% of k = 0's.
- **Soundness:** if fewer than **500** qualifying holdout impressions have history ≤ 2, the check isn't sound. Then **k = 3 by judgment** (topics dominate until about 3 clicks, then half weight), recorded as judgment, not as a result.

### Step 4: does the blend depend on history length?

- Re-run Step 1's grid on cold-user impressions only (history ≤ 2), with Step 1's normalization and half-life.
- **Adopt per-segment weights** (`cold_segment` in the config) only if cold-user nDCG@10 gains ≥ 1% relative over the shared weights. Otherwise no segment.

### Dev (reported once, never tuned on)

- **Data:** MIND-large dev (376,471 impressions) with the **shipped v0.2.0 ONNX encoder.** Centroids and stated topics come from MIND-large train.
- **Rows:** model-only; blend (Step 1); blend + MMR (Step 2); with the cold-start blend on the stated-topic population (Step 3).
- **Slices:** overall, cold candidates, cold users, per category.
- **MMR sample:** MMR runs per impression in Python, so the MMR rows, and the blend rows they're compared with, use a fixed sample of **50,000 impressions** (`numpy.random.default_rng(0)`, sorted). The blend rows are also reported on the full set.
- **Nothing is changed after dev.**

### Reports

- `ml/reports/ranking-mind-small-holdout.{md,json}` and `ml/reports/ranking-mind-large-dev.{md,json}`. **Aggregates only.**
- One command each: `make rank-tune` and `make rank-report`.
- `firstread-ml rank-tune --write-config` writes `rank-1.yaml` from Step 1–4's choices.

## Alternatives considered

- **Tuning weights on dev,** or on the shipped encoder's holdout scores. Inflated numbers (ADR 0003), and the shipped encoder saw the holdout day.
- **Learning the blend** (a small re-ranker on MIND). Learned weights would fit MSN's 2019 layout and still not know live outlet breadth. A learned re-ranker on FirstRead's own logs is in SPEC's "later" list.
- **Raw trending scores in the blend.** MIND click counts and live outlet counts have unrelated scales; only a rank transform makes one weight mean the same thing in both.
- **A followed-topic boost for warm readers.** Can't be tuned on MIND (no stated topics). Followed topics feed the candidate pools and the decaying `k/(k+n)` weight instead.

## Consequences

- One file, `rank-1.yaml`, holds every ranking constant. The API, the offline replay and the persona report read it through the same functions.
- The proxies' limitations go into learn chapter 03 next to the numbers.
- A new ranking version means a new YAML file (pinned SHA), new reports, and a `model_version` bump in responses.

## Results (added after the rules; the rules above are unchanged)

### Holdout: MIND-small, 30,270 impressions, fit-days encoder

Full table: `ml/reports/ranking-mind-small-holdout.md`.

**Step 1, blend:** raw cosine + 0.05 · freshness (3 h half-life) + 0.025 · trend.

| nDCG@10 | Model only | Chosen blend |
|---|---:|---:|
| overall | 0.3765 | 0.3843 (+2.1%) |
| cold candidates | 0.4525 | 0.4619 (+2.1%) |
| cold users | 0.3188 | 0.3318 (+4.1%) |

- The best raw config by J (h = 6 h, 0.05 / 0.05, J 0.4241 vs 0.4231) is within the 0.001 tie band, so the rule took the lighter weights.
- z-score and percentile normalization were worse at their best (J 0.4190 and 0.4189).

**Step 2, MMR:** λ = 0.7.
- nDCG@10 overall 0.3818, 99.3% of λ = 1's 0.3843; ILD@10 rises from 0.5502 to 0.5557.
- MIND's impressions are already diverse, so MMR moves little offline. On live candidate pools with many copies of big stories it matters more (P3e reports it per persona).

**Step 3, cold-start k:** k = 0, tuned. 737 cold (≤ 2 clicks) stated-topic impressions, over the 500 soundness threshold.
- The k values are within noise on cold users: 0.3151 (k = 0) vs 0.3148 (k = 3).
- **But the topic blend helps readers with 3–50 clicks:** k = 3 lifts nDCG@10 on the whole stated-topic population from 0.3787 to 0.3833 (+1.2%), with +0.016 at 6–10 clicks.
- The pre-registered rule optimizes the ≤ 2-click bucket only, so it chose k = 0. The rule is not changed after the fact; a different k would need a new rule and `rank-2`. That is the owner's call (raised in the 3b PR).
- With k = 0 the topic vector is used only at **zero** clicks (`cold_start_vector` returns the topic vector when n = 0, whatever k is). The first click replaces it.
- The zero-click bucket is identical across k by construction. This study does not compare the topic vector with the population vector for brand-new readers.

**Step 4, history-dependent weights:** adopted, marginally.
- Cold users (≤ 2 clicks) get w_trend = 0.1 instead of 0.025.
- Cold-user nDCG@10 0.3318 → 0.3353, +1.06% against the 1% threshold.

**Config:** `packages/core/src/firstread_core/ranking/rank-1.yaml` (SHA pinned in `test_ranking.py`).

### Dev: MIND-large, 376,471 impressions, shipped v0.2.0 encoder (reported once)

Full table: `ml/reports/ranking-mind-large-dev.md`.

**The single dev look.** The first attempt, on 2026-10-22, was killed for memory (exit 137, then a forced quit of the machine) **before it produced any dev metric**: its log holds only the command line and no report file was written. The report was then rewritten to stream 2,048 impressions at a time end to end, with a test that the result doesn't depend on the chunk size. It ran once, with a peak RSS of 4.13 GB, reached while deriving inputs from train. Nothing was tuned on these numbers.

**Check:** model-only reproduces the model card's shipped seed-0 numbers exactly (0.4149 / 0.5515 / 0.3625).

| nDCG@10 (full dev) | Model only | rank-1 blend |
|---|---:|---:|
| overall | 0.4149 | 0.4199 (+1.2%) |
| cold candidates | 0.5515 | 0.5522 (+0.1%) |
| cold users | 0.3625 | 0.3811 (+5.1%) |

- The blend gains or holds on 13 of 14 categories (largest: travel +0.019, news +0.013, weather +0.014). Finance dips by 0.001.
- **MMR,** on the fixed 50,000-impression sample: λ = 0.7 costs 0.9% nDCG@10 (0.4189 → 0.4151) for ILD@10 0.5502 → 0.5558. As on holdout, MIND's curated impressions leave MMR little to fix.
- **Cold start,** on the 32,284 sampled impressions whose users have stated-topic proxies (blend + MMR): giving zero-click readers the topic vector instead of the population vector raises cold-user nDCG@10 from 0.3628 to 0.3700 (+2.0%), with overall 0.4099 → 0.4102. With k = 0 that is the only place topics act.

**Against SPEC §11's quality target:** the shipped model already beats the best baseline (ADR 0003/model card), and the blend keeps it ahead overall and on cold candidates.

## Amendment (2026-10-23): rank-2 sets k = 3 as a product decision

**Decision (owner, PR #17 review):** `rank-2.yaml` is identical to `rank-1` except `cold_start_k: 3.0` and `cold_start_k_source: judgment`.
- It was the default (`model_version` = `v0.2.0+rank-2`) until rank-3 (amendment of 2026-10-23, below).
- `rank-1.yaml` stays for the record; both files' SHA-256s are pinned.
- The holdout rules and the rank-1 dev report above are unchanged.
- **Dev is not re-run:** dev was looked at once, for rank-1, and this change isn't tuned on anything.

**Why k = 3:**
- **With k = 0, one click replaces a new reader's chosen topics entirely.** NRMS-lite's residual weight α is about 0.02 (model card), so after one click the user vector is essentially that article.
- SPEC §7.4 asks for the topic vector to be *blended* with the click vector as clicks arrive.
- **rank-1's rule looked only at readers with ≤ 2 clicks,** where k barely matters (1–2 clicks: 0.3262 at k = 0 vs 0.3255 at k = 3, within noise).
- **On the same holdout, k = 2–3 helps readers with 3–10 clicks** by about 3–4% (3–5 clicks: 0.3685 → 0.3838; 6–10 clicks: 0.3958 → 0.4117).
- **A new MIND study wouldn't settle it,** because the stated-topic proxy is revealed interest, not stated.
- **k = 3 is the plan's pre-registered judgment value** (step 3's fallback): topics dominate until about 3 clicks, then half weight.

**Holdout k sweep** (stated-topic users, blend + MMR, nDCG@10; from `ml/reports/ranking-mind-small-holdout.md`):

| k | stated_overall | history_0-0 | history_1-2 | history_3-5 | history_6-10 | history_11-50 | stated_cold_users |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.3787 | 0.3065 | 0.3262 | 0.3685 | 0.3958 | 0.3846 | 0.3151 |
| 0.5 | 0.3796 | 0.3065 | 0.3227 | 0.3806 | 0.3979 | 0.3852 | 0.3135 |
| 1.0 | 0.3807 | 0.3065 | 0.3249 | 0.3825 | 0.4017 | 0.3863 | 0.3145 |
| 2.0 | 0.3820 | 0.3065 | 0.3247 | 0.3844 | 0.4067 | 0.3878 | 0.3144 |
| 3.0 | 0.3833 | 0.3065 | 0.3255 | 0.3838 | 0.4117 | 0.3891 | 0.3148 |
| 5.0 | 0.3832 | 0.3065 | 0.3198 | 0.3782 | 0.4055 | 0.3903 | 0.3123 |
| 10.0 | 0.3816 | 0.3065 | 0.3127 | 0.3660 | 0.3990 | 0.3888 | 0.3092 |
| 20.0 | 0.3761 | 0.3065 | 0.3147 | 0.3484 | 0.3821 | 0.3819 | 0.3101 |

- The zero-click column is identical by construction: with no clicks, the topic vector is used whatever k is.

**Follow-up check, P3e persona replay ("new reader"):** a reader follows 3 topics, then clicks one off-topic story. Report the share of page-2 items from the followed topics vs the clicked story's topic, under rank-1 and rank-2.

### Result of the new-reader check (P3e persona replay, 2026-10-23)

**Setup:** a reader follows 3 topics, reads page 1, clicks the top trending story outside those topics, then reads page 2. Run with the real API at the 16 steady clock times of the replay (≥ 24 h of news), for 3 topic sets (sports/health/entertainment, business/tech/politics, world/travel/food), under rank-1 and rank-2. 48 readers each, 960 page-2 items each. `docs/learn/data/03-personas.json`, `new_reader_check`.

| Page 2 items | rank-1 (k = 0) | rank-2 (k = 3) |
|---|---:|---:|
| from the 3 followed topics | 10.5% | **50.8%** |
| from the clicked story's topic | 48.5% | **13.4%** |

(Re-measured in the rank-3 run on 2026-10-23; the first run gave 11.5 / 47.3 and 51.3 / 12.5.)

- **With k = 0, one off-topic click takes over the feed:** almost half of page 2 comes from the clicked topic and about a tenth from the topics the reader chose.
- **With k = 3 the chosen topics keep about half of page 2** (75% topic weight after one click), and the click still shows.
- The rest of page 2 (36% under rank-2, 41% under rank-1) is neither: trending and fresh stories from other topics.
- This supports the owner's rank-2 decision. It is a behavioural check, not an accuracy measurement (there are no live clicks yet).

## Amendment (2026-10-23): rank-3 raises the shared freshness and trend weights, as an owner decision

**Decision (owner, PR #21 review):** `rank-3.yaml` is rank-2 with **`w_fresh` 0.1 and `w_trend` 0.05**. The cold segment (≤ 2 clicks: 0.05 / 0.1) and every other constant are unchanged.
- It is the default (`model_version` = `v0.2.0+rank-3`).
- rank-1 and rank-2 stay for the record, and all three files' SHA-256s are pinned.
- **Dev is not re-run.** The rules and results above are unchanged.

**Why:**
- **It ties on the pre-registered objective.** On the MIND-small holdout (`ml/reports/ranking-mind-small-holdout.json`, `step1.top`), raw normalization, h = 3 h, w_fresh 0.1, w_trend 0.05 has J = 0.42307, against 0.42309 for the shipped 0.05 / 0.025. That is inside the rule's 0.001 tie band; the shipped config won only on the "smallest w_fresh + w_trend" tie-break. Its cold-user nDCG@10 (0.3329) passes the rule's guard against model-only (0.3188).
- **The persona replay showed what that tie-break cost on live news.** Under rank-2 only a third of a persona's page 1 was under 2 h old, with a median age of 4.1 h. The holdout's freshness proxy (MIND's first-seen time) is too coarse to separate the tied configs.
- **The replay's earlier variant (w_fresh 0.1 with w_trend 0.025) wasn't chosen:** it isn't among the holdout's 15 best configs (J < 0.4214).

**Persona replay, rank-2 vs rank-3** (same replay, same 16 steady clock times, 1,600 page-1 items each; `docs/learn/data/03-personas.json`):

| Page 1, all personas | rank-2 | rank-3 |
|---|---:|---:|
| under 2 h old | 33.1% | 65.6% |
| under 30 min old | 6.3% | 14.8% |
| median age | 4.1 h | 1.3 h |

- **New readers are identical under rank-2 and rank-3, by construction:** with ≤ 2 clicks the cold segment's weights apply, and rank-3 kept them.
  - Their page 1 is 29.2% under 2 h (median 3.8 h).
  - After one off-topic click, page 2 is 50.8% followed topics and 13.4% the clicked topic (rank-1, k = 0: 10.5% and 48.5%).
- **Story grouping:** page-1 cap drops are about the same (mean 0.20 vs 0.23 per page); the largest group grew from 4 to 6 once (U.S. news), as fresher candidates bring more copies of developing stories into the re-rank set.
