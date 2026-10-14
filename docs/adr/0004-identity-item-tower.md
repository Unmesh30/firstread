# ADR 0004: Identity item tower (frozen article embeddings) for NRMS-lite

- Status: Accepted
- Date: 2026-10-14

## Context

A two-tower recommender scores `score(user, article) = f(user) · g(article)`. NRMS learns both towers: a news encoder `g` over word embeddings, and a user encoder `f` over the encoded history.

FirstRead has constraints NRMS didn't:

- **Brand-new articles must be scorable immediately.** Ingestion embeds each article once, within minutes of it appearing, and stores the vector in Postgres (pgvector HNSW) for ANN retrieval (SPEC §8).
- **Models get retrained and re-released** (HF Hub versions). If the article tower changes, every stored article vector is stale and must be recomputed. The ANN index would also have to be rebuilt, and old and new vectors must never be mixed.
- **$0 budget, CPU serving.** The API scores with ONNX Runtime on Cloud Run. A small user encoder is cheap; running a text encoder per candidate at request time is not.
- **Train/serve consistency.** The embedding function exists once, in `packages/core` (ADR 0001, ADR 0003).

## Decision

- The **item tower is the identity** on the frozen `bge-small-en-v1.5` embedding: `g(article) = embed(article_text(title, summary))`, L2-normalized, 384-d, computed once by `packages/core`.
- Only the **user tower is learned**: NRMS-lite (multi-head self-attention over the last 50 clicked-article embeddings, then additive attention pooling) outputs a user vector in **the same 384-d space**. The score is `scale · dot(user, article)`.
- The user tower is residual on the zero-training baseline: `u = normalize(mean(history)) + α · proj(pool(MHSA(history)))` with `α` initialized to 0. An untrained model is exactly the mean-of-history baseline.

## Alternatives considered

- **Learned item tower (full NRMS news encoder, or a projection head on top of bge):** possibly more accurate, since the paper's NRMS gets nDCG@10 41.63 on MIND test with a learned news encoder. But every model release would mean re-embedding all stored articles and rebuilding the HNSW index, with version-skew risk in between. It also puts a text model in the request path or the release process. Deferred to v2: a learned *projection* on stored embeddings could be applied at query time to the user side only, or re-materialized in a migration.
- **Fine-tuning bge-small on MIND:** same re-embedding problem, plus a heavier training job than a 16 GB Mac comfortably runs.
- **No learned model (mean-of-history only):** simpler, but leaves accuracy on the table. NRMS-lite must beat it on nDCG@10 overall and on cold-start candidates (P1 acceptance).

## Consequences

- Shipping a new model version changes only the user encoder (an ONNX file pinned by the API). Stored article vectors and the ANN index stay valid across releases.
- Retrieval can run ANN directly on stored embeddings with the user vector as the query: inner product over normalized article vectors. The user vector's norm doesn't change the ranking.
- The model's ceiling is bounded by how well generic bge-small embeddings capture what makes users click. Our results are not directly comparable to NRMS with a learned news encoder, and reports say so.
- Changing the embedding model (or `TEXT_PREP_VERSION`) still requires re-embedding everything. The embedding cache key and golden test make that impossible to do by accident.
