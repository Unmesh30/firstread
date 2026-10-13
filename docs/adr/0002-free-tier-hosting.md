# ADR 0002: Free-tier hosting ($0/month)

- Status: Accepted
- Date: 2026-10-13

## Context

FirstRead v1 is a public demo that must cost **$0 per month** (SPEC §4) while still looking and behaving like a production system: an always-reachable web app, an autoscaling API, a managed database with vector search, a cache, scheduled ingestion and CI/CD.

## Decision

| Need | Service (free tier) | Notes |
|---|---|---|
| Source, CI, scheduled ingestion | GitHub public repo + GitHub Actions | Standard runners are free for public repos. Scheduled runs can be late and are disabled after 60 days of repo inactivity. |
| Web app hosting | GitHub Pages | Expo static export served under `/firstread` (`experiments.baseUrl`), `.nojekyll` so `_expo/` is served. |
| API | Google Cloud Run (always-free tier), request-based billing, min instances 0, max 3 | Guarded by a **$1 budget alert**. Images in Artifact Registry (< 0.5 GiB, keep last 2). |
| Postgres + vector search | Supabase Free (500 MB, pgvector) | Pauses after ~1 week idle; ingestion writes keep it active. Transaction pooler needs `statement_cache_size=0` in asyncpg. |
| Auth | Supabase Auth (anonymous + Google linking) | 50K MAU free. |
| Cache / pools | Upstash Redis Free (~500K commands/month) | We budget ≤ 300K commands/month. |
| Model artifacts | Hugging Face Hub public repo | Versioned releases; services pin a version. |
| Training | Local Apple Silicon Mac; Kaggle/Colab free GPUs as fallback | |

Free-tier limits are re-verified against official docs at the start of the phase that uses each service (P2 for GDELT, P5 for cloud services).

## Alternatives considered

- **Fly.io / Render / Railway** for the API: free tiers have changed often or require a card with usage billing. Cloud Run's always-free tier plus a budget alert is the most predictable.
- **Vercel / Netlify** for the web app: fine, but GitHub Pages keeps everything in one account and needs no extra service.
- **Self-hosted Postgres + Redis on a free VM:** more ops work, no managed backups, and we'd have to secure it ourselves.
- **Pinecone / Qdrant Cloud** for vectors: another service to manage; pgvector keeps vectors next to article metadata so one query can filter and rank.

## Consequences

- Cold starts on Cloud Run (min instances 0); SLO allows ≤ 5 s cold start.
- Quotas are real constraints: load tests run **only** against the local Docker stack.
- The local stack (Postgres 17 + pgvector, Redis 7) mirrors production so we develop without touching quotas.
- Any change that could incur cost requires explicit owner approval.
