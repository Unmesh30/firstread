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
| `ml` | Python 3.12, PyTorch, scikit-learn, MLflow, ONNX export | Local Mac; Kaggle/Colab fallback |
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

## Status by phase

| Phase | What exists |
|---|---|
| **P0 Bootstrap** ✅ | uv workspace (4 members), FastAPI `/healthz` + `/readyz`, structlog JSON logs, Expo app placeholder with web export under `/firstread`, Docker Compose stack, Makefile, pre-commit, CI (`python`, `mobile`, `docker`, `secrets`), Dependabot, docs |
| P1 Offline ML | — |
| P2 Ingestion | — |
| P3 API | — |
| P4 App | — |
| P5 Cloud | — |
| P6 Performance | — |
| P7 Hardening | — |
| P8 Showcase | — |

## Related

- ADRs: [0001 monorepo and tech stack](adr/0001-monorepo-and-tech-stack.md), [0002 free-tier hosting](adr/0002-free-tier-hosting.md)
- Learn chapters: [00 bootstrap](learn/00-bootstrap.md)
