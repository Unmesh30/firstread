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
