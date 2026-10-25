# FirstRead architecture

> Source of truth for *what* we build is [SPEC.md](../SPEC.md). This page describes the system and tracks **what exists today**.

## System overview

```mermaid
flowchart LR
  subgraph Sources
    G[GDELT GKG 15-min files]
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
| `apps/mobile` | Expo SDK 57 (React Native + react-native-web + TypeScript strict + Expo Router, static rendering), TanStack Query, zustand, openapi-fetch, supabase-js (lazy) | GitHub Pages (web) now; EAS builds later |
| `services/api` | Python 3.12, FastAPI, uvicorn, asyncpg, redis-py, ONNX Runtime (user encoder only, files baked and verified at build), pydantic-settings, structlog | Google Cloud Run (us-east1) |
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
  HUB -.->|P3: Docker models stage<br/>downloads + verifies v0.2.0| API2[API: user encoder]
```

- Hub repo: [Unmesh30/firstread-models](https://huggingface.co/Unmesh30/firstread-models), releases [`v0.1.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.1.0) and [`v0.2.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.2.0) (topic classifier; user encoder inherited from v0.1.0 by hash, ADR 0006).
- Model card: [docs/model-card.md](model-card.md). Decision record: [ADR 0005](adr/0005-model-artifacts-on-hf-hub.md).
- The weights are MIND-derived and **non-commercial** (MSR License Terms). The code is MIT.
- The release ships only weights and aggregates: no MIND rows, text or embedding caches.
- Services pin a tag and verify `manifest.json`. Article embeddings stay valid across releases (identity item tower, ADR 0004).

## Live ingestion (Phase 2)

```mermaid
flowchart LR
  subgraph Sources
    RSS[8 RSS feeds<br/>explicit permission only<br/>conditional GET]
    GKG[GDELT GKG 15-min files<br/>catch-up cursor ≤ 2 h]
  end
  WIN[(runner-side 72 h window<br/>Actions cache; delta + checksum)] <--> X
  GKG --> PL[place outlet: GDELT domain lookup<br/>+ overrides, blocklist]
  RSS --> N[normalize: clean title, canonical URL,<br/>English, junk/short/stale filters]
  PL --> N
  N --> X[exact dedupe: URL, title fingerprint 72 h]
  X --> E[embed new items<br/>packages/core Embedder]
  E --> D[near-dup + story cluster<br/>per-pair-type cosine thresholds, 48 h]
  D --> C[topic: v0.2.0 ONNX classifier<br/>+ feed hint + geo rule]
  C --> S[select GDELT: breadth, ≤ 3/outlet,<br/>topic floor, 50/run]
  S --> PG[(Postgres: articles, story_coverage,<br/>feed_state, source_cursor, ingestion_runs)]
  PG --> RET[retention: 14 d / 48 h / 30 d]
  PG -->|delta only| WIN
  WIN --> POOLS[(Redis: one versioned pool snapshot<br/>fresh · 13 topics · trending · centroids · stats<br/>3 commands a run)]
  PG -->|story_coverage aggregate| POOLS
```

## Feed serving (Phase 3d)

```mermaid
flowchart LR
  REQ[GET /v1/feed] --> AU[1 auth<br/>cached JWKS]
  AU --> ST[2 state<br/>Redis reads ∥ profile row<br/>version check, rebuild]
  ST -->|cursor page, nothing changed| SL[(slate in Redis<br/>next items, no re-rank)]
  ST -->|first page, click, hide, slate low| EN[3 encode<br/>cached vec or micro-batched ONNX<br/>+ cold-start topic blend]
  EN --> CA[4 candidates ∥<br/>ANN HNSW top-200 · pool cosines<br/>exclusions in SQL]
  PC[(in-process pool snapshot<br/>refreshed ≤ 1/min)] --> CA
  CA --> HY[5 hydrate top 150<br/>article LRU, misses in 1 query]
  HY --> RK[6 rank<br/>τ_story read filter · story groups<br/>MMR, 2 per story across pages · reasons]
  RK --> PE[7 persist<br/>served_slates, then Redis slate]
  SL --> PE
  PE --> RES[page + next_cursor<br/>Server-Timing, degraded]
```

Each stage has a deadline and a tested way to fail; the table is in [ADR 0017](adr/0017-feed-serving-pipeline.md). Measured locally, a page costs 3–8 Redis commands and 0.1–77 KB of Postgres egress depending on whether it re-ranks; a first page on a cold instance costs ~209 KB (the article cache is empty).

## Personas, "Why this?" and stats (Phase 3e)

```mermaid
flowchart LR
  WIN[(runner-side 72 h window)] -->|daily, 48 h, confident topics,<br/>1 per story, ≤ 3 per domain| SEL[persona selection]
  SEL -->|INSERT … SELECT, vectors stay in Postgres| PA[(demo_personas<br/>persona_articles)]
  PA -->|5 summaries, ~2 KB a run| BLOB[(pool snapshot blob)]
  BLOB --> PC[API: in-process snapshot]
  PC --> LIST[GET /v1/demo/personas<br/>GET /v1/stats<br/>no Postgres, cached 60 s]
  ACT[POST …/activate<br/>409 history_exists unless replace] -->|copy into user_history,<br/>new versions| UH[(user_history)]
  WHY[GET /v1/articles/id/why] -->|2 Redis cmds + profile<br/>same blend and reasons| RS[components + reasons<br/>for the caller only]
```

Measured locally at Cloud Run's shape (1 vCPU, 512 MiB): one instance holds ~70 RPS with a 44 ms p95 server time (21 ms at 50 RPS), and saturates at 80 RPS, so SPEC's 100 RPS needs two instances; P5 starts at max concurrency 10 and 3 instances ([ADR 0017](adr/0017-feed-serving-pipeline.md)). A session costs 30–37 Redis commands and 153–373 KB of Postgres egress; Upstash commands bind at ~250–310 sessions a day ([ADR 0011](adr/0011-free-tier-budgets.md)).

## The app (Phase 4)

```mermaid
flowchart LR
  subgraph Build["expo export -p web (CI)"]
    R[src/app routes] -->|pre-rendered in Node| H[one .html per route<br/>theme CSS vars + pre-paint script,<br/>share-card tags, font preload]
    H --> B{check-dist<br/>secrets · budget 469.8 KB +5%}
  end
  subgraph Browser
    S[screens] --> Q[TanStack Query<br/>feed cursor pages, never auto-refetched]
    S --> Z[zustand<br/>event queue · toasts · theme]
    F[FlatList 50% crossings] --> V[viewability: 1 s, foreground]
    V --> Z
    Q --> C[openapi-fetch + AuthProvider<br/>dev X-Dev-User · Supabase JWT]
  end
  B -->|GitHub Pages, P5| Browser
  C -->|GET /v1/feed, /me, /why, /stats| API[FastAPI]
  Z -->|POST /v1/events: ≤ 100 events, ≤ 60 KB,<br/>keepalive, idempotent ids| API
```

- Every route is static HTML under `/firstread` (no dynamic segments: `/why?id=`), hydrated without mismatches; dark mode and the nav layout are right before JavaScript ([ADR 0019](adr/0019-app-architecture.md)).
- Impressions follow the MRC rule (≥ 50% for 1 continuous second in the foreground); events are batched, retried, persisted and sent with keepalive when the page hides ([ADR 0020](adr/0020-impressions-and-event-delivery.md)). A scripted session costs 33 Redis commands (≤ 37 asserted).
- First load 469.8 KB gzip (JS 391.5 + WOFF2 font 78.3); the Supabase chunk (65.5 KB) loads only in Supabase mode.
- Checked by 207 Jest tests and 49 Playwright tests against the local stack, including a full axe pass in light and dark, a keyboard-only walk, reflow at 320 px and 200% zoom, and reduced motion.

## Status by phase

| Phase | What exists |
|---|---|
| **P0 Bootstrap** ✅ | uv workspace (4 members), FastAPI `/healthz` + `/readyz`, structlog JSON logs, Expo app placeholder with web export under `/firstread`, Docker Compose stack, Makefile, pre-commit, CI (`python`, `mobile`, `docker`, `secrets`), Dependabot, docs |
| **P1 Offline ML** ✅ | **1a:** license-gated, SHA-pinned MIND download; parquet loader; shared embedder (`packages/core`, pinned bge-small revision, golden test, cache key); eval harness (AUC/MRR/nDCG, cold-candidate/cold-user/category slices, time-based holdout); baselines (random, popularity, recency, mean-history); EDA; synthetic CI eval gate. **1b:** NRMS-lite user encoder (identity item tower, ADR 0004), 3-seed results with paired bootstrap CIs, category classifier, MLflow (sqlite) tracking, `make reproduce`; beats every baseline on MIND-large dev overall and on cold-start. **1c:** ONNX export (user encoder, classifier) with parity tests, ONNX-based real-data eval gate, release builder with SHA256 manifest, model card, HF Hub release `v0.1.0`. |
| P2 Ingestion | **2a-ml:** title-only skew study (titles alone cost the user encoder 2.1% relative nDCG@10), 13 app topics in `packages/core` with the shared `resolve_topic` rule, reviewed MIND subcategory → topic map, topic classifier trained on title+abstract and title-only embeddings, release builder that inherits the v0.1.0 user encoder byte for byte (release `v0.2.0`, ADR 0006). **2a:** Postgres schema as SQL migrations with RLS on every table, checked on the Supabase image in CI (ADR 0008); RSS adapter (conditional GET, per-host serial, only feeds with explicit website permission) and GDELT GKG adapter (catch-up cursor, domain placement, blocklist, breadth-ranked topic-aware cap) (ADR 0007); URL canonicalization, title fingerprints, validated per-pair-type near-duplicate and story-clustering thresholds (ADR 0009); topic classification with the v0.2.0 classifier; idempotent upsert. A run inserts about 220 articles from a cold start (169 RSS + 50 GDELT) and 0 when repeated. **2b:** press-release filter (press-wire domains, PR-shaped titles); retention (articles 14 d, coverage 48 h); Redis pools (fresh, 13 topics, trending by outlet breadth with a 6 h half-life) rebuilt from Postgres within 30 commands per run (ADR 0010); per-story cap in GDELT selection; replay of archived GKG files with a stepped clock (steady state: 4,800 articles/day, ~227 MB for 14 days, all pools full); GDELT↔GDELT story threshold raised to 0.925 after a replay-based re-check; manual-dispatch `ingest.yml` with a Markdown job summary. Live topic accuracy 0.79 (LLM-labeled, independently spot-checked). |
| **P3 API** ✅ | **3a:** runner-side 72 h window snapshot in the Actions cache (delta + count/max/id-checksum, any mismatch reloads), cutting ingestion egress from ~14 MB to 84 KB a run with identical dedupe decisions (0 of 267,877 changed on the replay) (ADR 0012); pools published as one versioned blob in 3 commands a run (ADR 0012); trending v2: split stories merged by complete linkage at τ_feed, syndication networks counted once, representatives by quality (ADR 0013); free-tier budget table (ADR 0011). **3b:** ranking in `packages/core` (freshness, per-request normalization, rank-only trend feature, blend, MMR with a 2-per-story cap, cold-start vector, reasons, versioned configs), tuned on MIND-small holdout by pre-registered rules (`rank-1`) and reported once on MIND-large dev: nDCG@10 0.4149 → 0.4199 overall, cold users 0.3625 → 0.3811; `rank-2` adds the owner's cold-start k = 3 (ADR 0014; `rank-3` became the default in 3e). **3c:** Supabase JWT auth against a cached JWKS (ES256 allowlist, local dev bypass only) (ADR 0015); profiles, durable history, events and served slates with owner-only RLS and a local-only auth shim; idempotent events whose Redis side effects are append-only and versioned, rebuilt from Postgres on any gap; user-data retention with a size guard (ADR 0016); one error schema, rate limits, metrics, committed OpenAPI spec. **3d:** `GET /v1/feed` as seven stages with deadlines and a tested degraded mode each (Redis → Postgres copies, encoder → mean of history, ANN → pools, no snapshot → ANN only); the pinned encoder baked into the image by a verifying `models` stage and micro-batched across requests; candidates as ANN (HNSW iterative scans, ef_search 400, chosen by the pre-registered 14-day benchmark) ∥ pool cosines, both in SQL with read/hidden stories excluded server-side; 100-item slates with opaque cursors, no repeats, a 60 s page cache, and a re-rank after a click or hide; the 2-per-story cap held across pages; `Server-Timing` and per-stage metrics; 3–8 Redis commands a page (ADR 0017). **3e:** demo personas selected daily from live news on the runner and carried in the snapshot; activation with a 409 guard for real readers; `/why` recomputed for the caller with the feed's own blend and reasons; public `/v1/stats` and persona list served with no Postgres reads (ADR 0018); the persona replay (the real API on three replayed days: GDELT↔GDELT τ_feed fell back to 0.925 by ADR 0013's rule; k = 3 keeps 51% of a new reader's page 2 on their topics after an off-topic click, vs 10.5% with k = 0); the default ranking became `rank-3` (w_fresh 0.1, w_trend 0.05, a tie on the holdout objective; 66% of a persona's page 1 under 2 h old, vs 33%); local p95 at 1 vCPU (21 ms at 50 RPS, ~70 RPS per instance, after pinning OpenBLAS to one thread); measured session budgets (ADR 0011). |
| **P4 App** ✅ | **4a:** design tokens with a WCAG contrast test, core components, the headline font that never blocks first render, the typed client generated from `openapi.json`, the dev and Supabase auth providers (Supabase lazy-loaded), static routes for Pages with a 404 page, the E2E harness on a seeded database and its CI job; API: `X-Dev-User` in CORS only with the dev bypass, a 64 KiB `/v1/events` cap, un-hide a source. **4b:** onboarding, the infinite feed (never auto-refetched), cards as real links, like, hide with a pausable Undo, the Why sheet, every error/empty/offline/degraded state, the event queue (ids at creation, ≤ 100 events/60 KB batches, keepalive on page hide, retries) and the 50%/1 s viewability tracker; 33 Redis commands a session. **4c:** hydration error #418 fixed (width, Modal portal, color scheme via server snapshots) with a console-error guard on every route; demo readers with the 409 question; Account (Google linking via the provider, hidden sources, theme, delete my data); System, Privacy, About; the GDELT credit wherever its headlines appear. **4d:** dark surfaces before JavaScript (CSS-variable theme + a pre-paint script for the stored choice); deep links no longer fetch the feed underneath; Esc during a sheet's fade-in; middle-clicks recorded; Space toggles checkboxes/radios; focus to Undo after a keyboard hide; WOFF2 font and platform-split libraries (first load 501.2 → 469.8 KB gzip, budget lowered to +5%); Open Graph/Twitter cards with a 1200×630 image; full axe, keyboard, reflow, reduced-motion, image and list E2E checks (ADRs 0019, 0020). |
| P5 Cloud | — |
| P6 Performance | — |
| P7 Hardening | — |
| P8 Showcase | — |

## Related

- ADRs: [0001 monorepo and tech stack](adr/0001-monorepo-and-tech-stack.md), [0002 free-tier hosting](adr/0002-free-tier-hosting.md), [0003 MIND data and eval protocol](adr/0003-mind-data-and-eval-protocol.md), [0004 identity item tower](adr/0004-identity-item-tower.md), [0005 model artifacts on the HF Hub](adr/0005-model-artifacts-on-hf-hub.md), [0006 topic taxonomy and classifier](adr/0006-topic-taxonomy-and-classifier.md), [0007 live sources](adr/0007-live-sources-gdelt-gkg-and-rss.md), [0008 Postgres schema](adr/0008-postgres-schema-and-migrations.md), [0009 dedupe and clustering](adr/0009-dedupe-and-story-clustering.md), [0010 Redis pools and trending](adr/0010-redis-pools-and-trending.md), [0011 free-tier budgets](adr/0011-free-tier-budgets.md), [0012 runner window and pool snapshot](adr/0012-runner-window-and-pool-snapshot.md), [0013 trending v2](adr/0013-trending-v2.md), [0014 ranking](adr/0014-ranking-blend-mmr-cold-start.md), [0015 auth](adr/0015-auth-supabase-jwt.md), [0016 user data](adr/0016-user-data-rls-retention.md), [0017 feed serving](adr/0017-feed-serving-pipeline.md), [0018 personas, why, stats](adr/0018-personas-why-stats.md), [0019 app architecture](adr/0019-app-architecture.md), [0020 impressions and event delivery](adr/0020-impressions-and-event-delivery.md)
- Learn chapters: [00 bootstrap](learn/00-bootstrap.md), [01 offline ML](learn/01-offline-ml.md), [02 live ingestion](learn/02-live-ingestion.md), [03 recommendation API](learn/03-recommendation-api.md), [04 the app](learn/04-app.md)
- [Model card](model-card.md), [data card](data-card.md)
