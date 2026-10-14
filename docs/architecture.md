# FirstRead architecture

> Source of truth for *what* we build is [SPEC.md](../SPEC.md). This page describes the system and tracks **what exists today**.

## System overview

```mermaid
flowchart LR
  subgraph Sources
    G[GDELT DOC 2.0 API]
    R[Curated publisher RSS]
  end
  subgraph Ingest["Ingest job (GitHub Actions cron, every 15 min)"]
    I[fetch → normalize → dedupe → classify → embed → upsert]
  end
  subgraph Data
    PG[(Supabase Postgres + pgvector)]
    RD[(Upstash Redis)]
  end
  subgraph Serving["Google Cloud Run"]
    API[FastAPI recommender]
  end
  subgraph Clients
    W[Expo web on GitHub Pages]
    M[iOS / Android later]
  end
  HF[(Hugging Face Hub: versioned models)]
  SA[Supabase Auth]

  G --> I
  R --> I
  I --> PG
  I --> RD
  W -->|HTTPS + JWT| API
  M -.-> API
  W --> SA
  API --> PG
  API --> RD
  HF -->|pinned model version at build| API
  HF -->|pinned model version| I
```

## Components

| Component | Tech | Runs on |
|---|---|---|
| `apps/mobile` | Expo SDK 57 (React Native + TypeScript strict + Expo Router) | GitHub Pages (web) now; EAS builds later |
| `services/api` | Python 3.12, FastAPI, uvicorn, asyncpg, redis-py, pydantic-settings, structlog | Google Cloud Run (us-east1) |
| `services/ingest` | Python 3.12, httpx, feedparser, fastembed, asyncpg | GitHub Actions scheduled workflow |
| `ml` | Python 3.12, PyTorch, scikit-learn, MLflow, ONNX export (onnx, onnxscript, skl2onnx, onnxruntime) | Local Mac; Kaggle/Colab fallback |
| `packages/core` | Shared Python: text cleaning, URL canonicalization, embeddings, features, schemas | Imported by ml, ingest, api |
| Data | Supabase Postgres + pgvector, Upstash Redis | Managed free tiers |

**Golden rule:** embedding, text-cleaning and feature code exist once, in `packages/core` (prevents train/serve skew).

## Local development stack (Phase 0)

```mermaid
flowchart LR
  dev[Developer / curl / tests] -->|:8000| api
  subgraph compose["docker compose (infra/docker)"]
    api["api<br/>uvicorn --reload<br/>bind-mounted src"]
    db[("db<br/>pgvector/pgvector:0.8.7-pg17")]
    redis[("redis<br/>redis:7.4.11-alpine")]
  end
  api -->|asyncpg| db
  api -->|redis-py| redis
```

All ports bind to `127.0.0.1` only. `api` starts only after `db` and `redis` report healthy.

## Offline ML pipeline (Phase 1)

```mermaid
flowchart LR
  HF[(HF Recommenders/MIND<br/>pinned + SHA256)] -->|make mind-download| RAW[data/mind/raw]
  RAW --> PQ[processed parquet]
  PQ -->|packages/core: article_text + Embedder| EMB[embedding cache<br/>keyed by model + text prep]
  PQ --> SPL[holdout / dev splits]
  EMB --> SC[baselines / models]
  SPL --> SC --> MET[metrics x slices] --> REP[ml/reports: aggregates only]
```

The embedding function lives only in `packages/core` (`firstread-core[embed]`); the API image installs core without the `embed` extra, and CI fails if ML packages appear in it.

## Model release (Phase 1c)

```mermaid
flowchart LR
  CK[seed-0 MIND-large checkpoint<br/>+ classifier, data/models] --> EX[ONNX export<br/>dynamo opset 18 · skl2onnx]
  EX --> PA{parity ≤ 1e-4<br/>synthetic + real histories}
  PA --> GT{real-data gate<br/>5,000 MIND-small dev imps<br/>vs mean_history + reference}
  GT --> LAT[ORT CPU latency<br/>batch 1 / 32]
  LAT --> DIR[data/release/vX.Y.Z<br/>allowlisted files + manifest.json SHA256]
  DIR -->|make release-upload<br/>clean main == origin/main| HUB[(HF Hub Unmesh30/firstread-models<br/>tag vX.Y.Z)]
  HUB -->|re-download + verify hashes| OK[git tag vX.Y.Z = manifest git_commit<br/>+ GitHub release linking the Hub]
  HUB -.->|P2 pins v0.2.0| ING[ingest: topic classifier]
  HUB -.->|P3 pins tag| API2[API: user encoder]
```

- Hub repo: [Unmesh30/firstread-models](https://huggingface.co/Unmesh30/firstread-models), releases [`v0.1.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.1.0) and [`v0.2.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.2.0) (topic classifier; user encoder inherited from v0.1.0 by hash, ADR 0006).
- Model card: [docs/model-card.md](model-card.md). Decision record: [ADR 0005](adr/0005-model-artifacts-on-hf-hub.md).
- The weights are MIND-derived and **non-commercial** (MSR License Terms). The code is MIT.
- The release ships only weights and aggregates: no MIND rows, text or embedding caches.
- Services pin a tag and verify `manifest.json`. Article embeddings stay valid across releases (identity item tower, ADR 0004).

## Status by phase

| Phase | What exists |
|---|---|
| **P0 Bootstrap** ✅ | uv workspace (4 members), FastAPI `/healthz` + `/readyz`, structlog JSON logs, Expo app placeholder with web export under `/firstread`, Docker Compose stack, Makefile, pre-commit, CI (`python`, `mobile`, `docker`, `secrets`), Dependabot, docs |
| **P1 Offline ML** ✅ | **1a:** license-gated, SHA-pinned MIND download; parquet loader; shared embedder (`packages/core`, pinned bge-small revision, golden test, cache key); eval harness (AUC/MRR/nDCG, cold-candidate/cold-user/category slices, time-based holdout); baselines (random, popularity, recency, mean-history); EDA; synthetic CI eval gate. **1b:** NRMS-lite user encoder (identity item tower, ADR 0004), 3-seed results with paired bootstrap CIs, category classifier, MLflow (sqlite) tracking, `make reproduce`; beats every baseline on MIND-large dev overall and on cold-start. **1c:** ONNX export (user encoder, classifier) with parity tests, ONNX-based real-data eval gate, release builder with SHA256 manifest, model card, HF Hub release `v0.1.0`. |
| P2 Ingestion | **2a-ml:** title-only skew study (titles alone cost the user encoder 2.1% relative nDCG@10), 13 app topics in `packages/core` with the shared `resolve_topic` rule, reviewed MIND subcategory → topic map, topic classifier trained on title+abstract and title-only embeddings, release builder that inherits the v0.1.0 user encoder byte for byte (release `v0.2.0`, ADR 0006). |
| P3 API | — |
| P4 App | — |
| P5 Cloud | — |
| P6 Performance | — |
| P7 Hardening | — |
| P8 Showcase | — |

## Related

- ADRs: [0001 monorepo and tech stack](adr/0001-monorepo-and-tech-stack.md), [0002 free-tier hosting](adr/0002-free-tier-hosting.md), [0003 MIND data and eval protocol](adr/0003-mind-data-and-eval-protocol.md), [0004 identity item tower](adr/0004-identity-item-tower.md), [0005 model artifacts on the HF Hub](adr/0005-model-artifacts-on-hf-hub.md), [0006 topic taxonomy and classifier](adr/0006-topic-taxonomy-and-classifier.md)
- Learn chapters: [00 bootstrap](learn/00-bootstrap.md), [01 offline ML](learn/01-offline-ml.md)
- [Model card](model-card.md), [data card](data-card.md)
