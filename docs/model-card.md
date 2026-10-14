---
license: other
license_name: microsoft-research-license-mind
license_link: https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf
library_name: onnx
language:
  - en
tags:
  - recommender-systems
  - news-recommendation
  - cold-start
  - onnx
  - mind
datasets:
  - Recommenders/MIND
---

# FirstRead models v0.2.0

Model artifacts for [FirstRead](https://github.com/Unmesh30/firstread), a real-time news recommender that ranks brand-new articles before they have any clicks.

- **Hub:** [Unmesh30/firstread-models](https://huggingface.co/Unmesh30/firstread-models). This release is the git tag [`v0.2.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.2.0). The previous release is [`v0.1.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.1.0).
- **Pin a tag, not `main`.** Every file's SHA256 is listed in `manifest.json`.

> **License: these weights are MIND-derived and non-commercial.** They were trained on the Microsoft News Dataset (MIND) and are released under its [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf), **not** under the MIT license of the FirstRead code. Use them for non-commercial research and demonstration only.

## What changed in v0.2.0

- **The classifier now predicts 13 app topics** (us, world, politics, business, tech_science, health, sports, entertainment, lifestyle, food_drink, travel, autos, weather) instead of MIND's 15 categories. Labels come from MIND (category, subcategory) pairs through one reviewed table, [`ml/configs/topic_map.yaml`](https://github.com/Unmesh30/firstread/blob/main/ml/configs/topic_map.yaml) ([ADR 0006](https://github.com/Unmesh30/firstread/blob/main/docs/adr/0006-topic-taxonomy-and-classifier.md)). MIND's categories have no tech, politics or world label, and "news" covered about a third of articles.
- **It is trained for headlines without summaries.** Each training article contributes its title+abstract embedding and its title-only embedding, because live GDELT articles have no summary.
- **The user encoder is unchanged.** `user_encoder.onnx`, `population.npy` and `config.yaml` are byte-for-byte copies of v0.1.0 (same SHA256), verified against the v0.1.0 manifest at build time. `manifest.json` → `user_encoder.inherited_from` says so. All user-encoder numbers below are from v0.1.0 and still apply.

## What's in this release

| File | What it is |
|---|---|
| `user_encoder.onnx` | NRMS-lite user encoder. Turns a reader's last 50 clicked-article embeddings into a user vector. Same file as v0.1.0. |
| `category_classifier.onnx` | Logistic regression that maps an article embedding to one of 13 app topics. |
| `labels.json` | Topic ids, in the classifier's output order. |
| `population.npy` | The 384-d user vector for a reader with no history: the normalized mean embedding of all training clicks. It is also baked into `user_encoder.onnx`. |
| `config.yaml` | The training config of the shipped model. |
| `metrics.json` | Dev results for all three seeds, bootstrap CIs, topic-classifier metrics (both text modes), parity, the real-data eval gate and latency. |
| `manifest.json` | SHA256 and size of every file, plus the git commit, embedding-model pins, feature constants and library versions. |

**What is not in this release:** MIND rows, article text, impression logs, user ids or embedding caches. Only learned weights and aggregate numbers are shipped. A build fails if its folder holds any file outside the list above.

## Model details

**User encoder (NRMS-lite).**
- **Input:**
  - `hist`: float32 `[batch, 50, 384]`. The reader's most recent clicked-article embeddings, oldest first and **right-aligned** (padding on the left, zeros).
  - `mask`: bool `[batch, 50]`. True where a click exists.
- **Output:** `user_vec`, float32 `[batch, 384]`.
- **Architecture:** multi-head self-attention (8 heads) over the history, then additive attention pooling, then a linear projection. The result is added **residually** to the normalized mean of the history:

  `u = normalize(mean(hist)) + α · proj(pool(MHSA(hist)))`

  α starts at 0, so the untrained model is the mean-of-history baseline. The trained α is 0.0203.
- **Empty histories** return `population.npy`.
- **Scoring:** `score = user_vec · article_embedding`.
  - The item tower is the identity on frozen article embeddings ([ADR 0004](https://github.com/Unmesh30/firstread/blob/main/docs/adr/0004-identity-item-tower.md)). Stored article vectors stay valid across model releases, and retrieval can run ANN directly on them.
  - The training temperature is `score_scale` = 7.26, recorded in `manifest.json` and the ONNX metadata. It doesn't change rankings.
- **Article embeddings:**
  - `BAAI/bge-small-en-v1.5`, the int8 ONNX export `Qdrant/bge-small-en-v1.5-onnx-Q` at a pinned revision, run through fastembed 0.9.0.
  - Text is `title + ". " + abstract`, cleaned (text-prep version 1). Vectors are L2-normalized and 384-d.
  - **Embed with exactly these pins** (see `manifest.json` → `embedding`). Vectors from any other embedding model are meaningless to this encoder.
- **Topic classifier:**
  - Input `embedding`: float32 `[batch, 384]`.
  - Outputs: `label` (int64 index into `labels.json`) and `probabilities` (float32 `[batch, 13]`).
  - Multinomial logistic regression, C = 1.0, trained on 97,651 MIND-large train articles × 2 text modes. C and class weighting were chosen by mean validation macro-F1 over both modes.
  - `manifest.json` → `category_classifier.low_confidence_threshold` = 0.50: below it, FirstRead lets a feed's topic hint break the tie if the hint is in the model's top 3. It is the lowest validation confidence from which every 0.05-wide bin is at least 50% accurate.
  - **Topic meaning:** `us` and `politics` are US domestic news (MIND is US news), `world` is news about places outside the US, the other ten are global subjects. FirstRead applies a geo rule on top: an article from a non-US source labelled us or politics becomes world.
- **Export:** PyTorch 2.14 dynamo exporter, opset 18; skl2onnx with zipmap off. Only the batch dimension is dynamic.
- **Parity (ONNX Runtime vs PyTorch/sklearn):**
  - Max abs difference ≤ 1e-4, checked on empty, partial and full histories, on real MIND histories and on mixed batches.
  - A row scores identically alone and inside a batch.
  - The measured diffs are in `metrics.json` → `parity` (all below 1e-6).

**Which weights ship:** seed 0, trained on all of MIND-large train (2019-11-09 to 2019-11-14). Seed 0 is the config's pre-declared seed, not the best of three. All three seeds are reported below.

## Intended use

- **Use it for:** ranking English news articles for a reader, given that reader's recent clicks, in the non-commercial FirstRead demo. In particular, scoring **brand-new articles with no clicks**, from their text alone.
- **Use the classifier for:** a topic label for live English headlines during ingestion, with or without a summary.
- **Out of scope:**
  - commercial use (the license forbids it);
  - non-English text;
  - decisions about people;
  - any domain other than news;
  - use with a different embedding model.

## Training data

**MIND-large** (Wu et al., ACL 2020): impression logs from 1M Microsoft News (MSN) users over six weeks of October–November 2019. Training uses the train split's impressions (Nov 9–14, 2.23M impressions) with K=4 negatives drawn from the same impression and a softmax cross-entropy loss.

- Hyperparameters were chosen on a **time-based holdout** (the last day of MIND-small train), never on dev.
- Data handling and the eval protocol are described in [ADR 0003](https://github.com/Unmesh30/firstread/blob/main/docs/adr/0003-mind-data-and-eval-protocol.md) and the [data card](https://github.com/Unmesh30/firstread/blob/main/docs/data-card.md).

## Evaluation

**Setup:**
- MIND-large **dev**: 376,471 impressions, evaluated once.
- Per-impression AUC, MRR, nDCG@5 and nDCG@10, averaged over impressions. Ties are broken by a seeded random key.
- Full report: [`ml/reports/results-mind-large-dev.md`](https://github.com/Unmesh30/firstread/blob/main/ml/reports/results-mind-large-dev.md).

| Model (nDCG@10) | Overall | Cold-start candidates | Cold users (≤ 2 clicks) |
|---|---:|---:|---:|
| random | 0.2864 | 0.4302 | 0.3119 |
| popularity | 0.3079 | 0.4303 | 0.3348 |
| recency | 0.2946 | 0.4830 | 0.3075 |
| mean_history (best baseline) | 0.3906 | 0.5332 | 0.3552 |
| **NRMS-lite seed 0 (shipped)** | **0.4149** | **0.5515** | **0.3625** |
| NRMS-lite, mean ± std over 3 seeds | 0.4145 ± 0.0004 | 0.5506 ± 0.0008 | 0.3623 ± 0.0001 |

- Shipped model, overall: AUC 0.6624, MRR 0.3190, nDCG@5 0.3536.
- **Cold-start candidates** are articles never clicked in training: 125,472 impressions. These are the articles FirstRead exists for.
- **Cold users:** 25,688 impressions, 11,270 of them with an empty history.

**Gain over mean_history (paired bootstrap):**
- Mean of the 3 seeds, 1,000 resamples of impressions.
- nDCG@10 +0.0239 overall (95% CI [+0.0232, +0.0248]) and +0.0174 on cold-start candidates (95% CI [+0.0159, +0.0188]).
- No resample showed a gain ≤ 0.

**Reference, not directly comparable:** NRMS in the MIND paper reports nDCG@10 0.4163 on the MIND-large **test** set (Wu et al., 2020, Table 3).
- It uses a different split (test, not our dev).
- It learns its own news encoder, while ours is frozen bge-small.

**Real-data release gate:**
- Before release, the exported ONNX file is scored on a fixed 5,000-impression MIND-small dev subset.
- It must beat mean_history and must not regress more than 1% against the committed reference.
- v0.1.0 and v0.2.0 (same encoder): nDCG@10 0.4217 vs 0.3946 overall, and 0.4600 vs 0.4329 on cold-start candidates.

### Where the model fails

**Per category,** NRMS-lite loses to the simple mean_history baseline on four categories (nDCG@10, seed 0 vs mean_history):

| Category | NRMS-lite | mean_history |
|---|---:|---:|
| tv | 0.7193 | 0.7481 |
| weather | 0.7970 | 0.8059 |
| video | 0.6953 | 0.7033 |
| lifestyle | 0.6754 | 0.6786 |

- It wins on the other ten categories with enough data, by the largest margins on news (+0.050) and travel (+0.054).
- A plausible reading, not verified: in these categories a reader's interest is narrow and stable, and the plain average of their history already captures it.
- **Cold users gain least:** +0.007 nDCG@10. With 0–2 clicks there is little history for attention to work with, and empty histories fall back to one population vector for everyone.

### Caveats on the numbers

- **The bootstrap resamples impressions, not users.** One user contributes many correlated impressions, so the CIs above somewhat **understate** the uncertainty. A user-level (cluster) bootstrap would be wider.
- **Seeds:** the std over 3 seeds measures training noise only, not data noise.
- **Dev is not test.** MIND test labels aren't public. Our dev numbers are reported once, after tuning on a separate holdout.

## Bias and limitations

- **2019, US, one platform.** MIND reflects what MSN users in the US clicked in late 2019, under MSN's own ranking and layout (position bias, editorial choices). The topics, publishers and events of that period don't match today's news. The model has never seen FirstRead's live sources (GDELT, RSS), whose distribution differs.
- **Popularity and exposure bias.** Clicks are only observed on what MSN chose to show.
- **Text-only.** The model knows an article only by its headline and summary, through a frozen general-purpose embedding. It can't see images, the publisher's reputation or the full text.
- **English only.**
- **The topic classifier learned US news from 2019.**
  - On 2,457 dev-only articles: accuracy 0.7346, macro-F1 0.6371 with title + abstract; accuracy 0.7184, macro-F1 0.6244 from titles only. Majority-class accuracy is 0.3268.
  - The weakest topics from titles only are world (F1 0.44; recall 0.35), travel (0.45) and autos (0.53). us/politics/world boundaries in MIND are noisy.
  - World has only 2,531 training articles. Live news from outside the US looks different from MIND's, so FirstRead audits the classifier on live articles separately (learn chapter 02).
  - Report: [`ml/reports/topic-classifier-mind-large.md`](https://github.com/Unmesh30/firstread/blob/main/ml/reports/topic-classifier-mind-large.md).
- **Headlines without summaries cost little.** On the real-gate subset, scoring with titles only (history and candidates) lowers the user encoder's nDCG@10 by 2.1% relative (0.4217 → 0.4130); it still beats mean_history (0.3748). Cold users lose more. Report: [`ml/reports/skew-title-only-mind-small.md`](https://github.com/Unmesh30/firstread/blob/main/ml/reports/skew-title-only-mind-small.md).

## Latency (ONNX Runtime, CPU)

User encoder, full 50-click histories, after 50 warm-up runs, 1,000 timed runs per row. **Measured on the owner's Mac (Apple Silicon) for v0.1.0, not on Cloud Run;** v0.2.0 ships the same file, and each build's own run is in `metrics.json`. P3 re-measures inside the API container.

Apple M4 (10 cores), macOS 26.3, ONNX Runtime 1.31.0:

| Batch | intra-op threads | p50 | p95 |
|---:|---:|---:|---:|
| 1 | 1 | 0.174 ms | 0.179 ms |
| 32 | 1 | 4.273 ms | 4.744 ms |
| 1 | 4 | 0.181 ms | 0.183 ms |
| 32 | 4 | 3.363 ms | 3.885 ms |

- Batching 32 readers costs about 0.13 ms per reader, against 0.17 ms one at a time.
- Extra threads help only at batch 32.
- Even if Cloud Run's vCPU is several times slower than an M4 core, the encoder should be a small part of the 20 ms scoring budget (SPEC §8). P3 verifies this.

## Commercial launch

These weights may not be used commercially. Before any commercial launch, FirstRead must:
- retrain on its own consented interaction data;
- switch to a licensed news source;
- publish the result as a new release under a new license.

## Verifying a download

```bash
hf download Unmesh30/firstread-models --revision v0.2.0 --local-dir firstread-models
cd firstread-models && python -c "
import hashlib, json; m = json.load(open('manifest.json'))
for n, f in m['files'].items():
    assert hashlib.sha256(open(n, 'rb').read()).hexdigest() == f['sha256'], n
print('all', len(m['files']), 'files match')"
```

Or, from the FirstRead repo: `make release-verify VERSION=0.2.0`.

## Citation

Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. 2020. *MIND: A Large-scale Dataset for News Recommendation.* In Proceedings of ACL 2020, pages 3597–3606.
