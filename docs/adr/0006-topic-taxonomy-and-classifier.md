# ADR 0006: App topic taxonomy and the v0.2.0 topic classifier

- Status: Accepted
- Date: 2026-10-14

## Context

Ingestion (P2) labels every live article so the app can offer topics (onboarding, per-topic candidate pools, "Why this?"). The v0.1.0 classifier predicted MIND's 15 categories, which is the wrong label set for a live news app:

- **No tech, science, politics or world.** In MIND these all sit inside the `news` category, which is 31% of articles. Most live articles would have been labelled "news".
- **Format categories.** `video` and `kids` describe formats or audiences, not subjects.
- **Rare categories.** games, middleeast and northamerica had too few articles to predict.

MIND's **subcategories** do separate these (`newsus`, `newspolitics`, `newsworld`, `newsscienceandtechnology`, …). They are, however, labels from a **US** news product in 2019: `newsus`, `newscrime` and `newspolitics` mean *US* domestic news.

Live sources add two more facts:

- **GDELT articles have no summary.** The title-only study (`ml/reports/skew-title-only-mind-small.md`) shows the user encoder loses 2.1% relative nDCG@10 when both history and candidates are titles only, so ranking is fine. The classifier still had to be trained for titles.
- **Many RSS feeds are topical** (a tech feed, a sports feed). They can help when the model is unsure.

## Decision

**13 app topics**, defined once in `packages/core/src/firstread_core/topics.py` and used by ml, ingest and the API:

| id | Meaning |
|---|---|
| `us` | US domestic news (incl. crime, offbeat, good news) |
| `world` | News about places outside the US |
| `politics` | US politics (incl. opinion, elections, fact checks) |
| `business`, `tech_science`, `health`, `sports`, `entertainment`, `lifestyle`, `food_drink`, `travel`, `autos`, `weather` | Global subjects |

**Labels come from one reviewed table**, `ml/configs/topic_map.yaml`, mapping MIND (category, subcategory) → topic or `exclude`.
- Pairs are matched inside their category, because subcategory names repeat (`video/news` ≠ `entertainment/news`).
- A category either has a `default` or must list every subcategory. An unmapped pair fails training, so nothing is dropped silently.
- Format-only pairs are excluded from training (4,043 MIND-large articles), for example `video/news` clips, photo galleries and `kids/fun`.
- Judgement calls, kept after review: `newsoffbeat → us`, `newsopinion → politics`.

**The classifier** is logistic regression on the frozen bge-small embedding (`ml/src/firstread_ml/models/topic_classifier.py`):
- Every training article contributes **two rows**: its title+abstract embedding and its title-only embedding.
- C and class weighting are chosen by the **mean** validation macro-F1 over both text modes, on an article-level split (both rows of an article stay on one side).
- It is tested on dev-only articles in both modes.

**Release gate** (in addition to classifier parity): every topic's title-only F1 ≥ 0.30, and title-only macro-F1 at most 0.10 below title+abstract. The release also checks that the classifier was trained with the committed `topic_map.yaml` (SHA256 in `policy.json`, the report and the manifest) and that its labels equal `TOPIC_IDS`.

**Resolving a live article's topic** (`firstread_core.topics.resolve_topic`, shared code):
1. The model's top-1 topic.
2. **Tie-breaker:** if the model's confidence is below the threshold (0.50, chosen on validation as the lowest confidence from which every 0.05 bin is ≥ 50% accurate; shipped in `manifest.json`), and the feed's `topic_hint` is among the model's top 3, use the hint.
3. **Geo rule:** if the item's geo is `intl` and the topic is `us` or `politics`, it becomes `world`. A feed's geo is set in `feeds.yaml` (Guardian US = `us`). A GDELT row is `us` only when its source country is the US.
4. `category_source` ∈ {model, feed_hint, geo_rule} records which step decided.

**Release v0.2.0** ships the topic classifier with the **v0.1.0 user encoder**, inherited byte for byte: `release build --inherit-user-encoder 0.1.0` downloads v0.1.0, verifies every hash against its manifest and copies the files. `manifest.json` records `user_encoder.inherited_from: v0.1.0`. The real-data eval gate still runs on the inherited encoder.

**Inheriting trusts a pin, not a tag.** Hub tags can move, so `release.PINNED_RELEASES` pins each inheritable release in code: v0.1.0 = Hub commit `f29ec56afbb760c04db94e638d9b60c3039658bd`, manifest SHA256 `cf094e8e…1477584`. `fetch_release` refuses if the tag no longer resolves to that commit, downloads at the commit (never the tag) and checks the manifest hash; `build` re-checks the manifest hash before copying. This is the same rule ingestion uses for the model it loads.

**Every release from v0.2.0 ships the topic classifier,** whether the encoder is inherited or retrained: `default_sources` always picks it, and `build` refuses a non-topic classifier for versions ≥ 0.2.0.

## Results (MIND-large, 2,457 dev-only articles)

| | Title + abstract | Title only |
|---|---:|---:|
| Accuracy | 0.7346 | 0.7184 |
| Macro-F1 | 0.6371 | 0.6244 |

The weakest topics from titles are world (F1 0.44, recall 0.35), travel (0.45) and autos (0.53); sports (0.92) and politics (0.81) are the strongest. Full report: `ml/reports/topic-classifier-mind-large.md`.

## Alternatives considered

- **Keep the 15 MIND categories and map them to topics at ingestion.** Simple, but "news" would cover most live articles, with no tech, politics or world. Rejected after review.
- **MIND topics plus feed hints for the missing ones.** Tech, world and politics would then exist only for RSS articles, and never for GDELT. Rejected: the topics would mean different things depending on the source.
- **Train on title+abstract only.** The test numbers show the augmented model loses little from titles alone; without augmentation, GDELT articles would be classified out of distribution.
- **A bigger model (fine-tuned bge, MLP).** More accuracy is possible, but the model would no longer be a 30 KB ONNX logistic regression that ingestion runs on a CPU runner. Deferred.
- **Re-export the user encoder for v0.2.0.** It would be functionally identical but not byte-identical, which makes "unchanged" harder to prove. Inheriting by hash is simpler and verifiable.

## Consequences

- `articles.category` stores an app topic id (SPEC §9 updated), plus `category_source` and `geo`.
- The app, API pools (`pool:topic:{topic}`) and personas use the 13 topic ids.
- A US-centric bias remains: the model learned what "us" and "world" meant to MSN in 2019. The geo rule fixes the most visible error (non-US local news filed as "us"), at the cost of a BBC story about US politics landing in `world`. Phase 2 audits ~100 live articles (half US, half non-US sources) and reports reweighted accuracy (learn chapter 02).
- Changing the taxonomy means editing `topic_map.yaml` (and `TOPIC_IDS` if a topic is added), retraining, a new release and a migration of stored categories. The SHA chain makes a mismatched classifier a build error.
