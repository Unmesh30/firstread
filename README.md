# FirstRead

[![CI](https://github.com/Unmesh30/firstread/actions/workflows/ci.yml/badge.svg)](https://github.com/Unmesh30/firstread/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A real-time "For You" news feed that ranks breaking stories within minutes of publication, before they have a single click.**

Most recommenders learn from clicks, so a brand-new article is invisible until people find it some other way. That's the *item cold-start* problem. FirstRead understands each new article from its text (headline + summary) and each reader from what they've read, so a story published five minutes ago can already be ranked for the right readers.

**Live demo: coming soon.**

## Stack

| Area | Tech |
|---|---|
| App | Expo (React Native + TypeScript, Expo Router), web on GitHub Pages |
| API | Python 3.12, FastAPI, asyncpg, redis-py, ONNX Runtime, on Google Cloud Run |
| Ingestion | GDELT DOC 2.0 + curated RSS, every 15 min on GitHub Actions |
| ML | PyTorch (NRMS-lite user encoder), `bge-small-en-v1.5` embeddings, trained/evaluated on MIND |
| Data | Supabase Postgres + pgvector, Upstash Redis |
| Tooling | uv workspace, ruff, mypy, pytest, Jest, pre-commit, Docker Compose, GitHub Actions |

Everything runs on free tiers ($0/month). See [docs/architecture.md](docs/architecture.md) and [SPEC.md](SPEC.md).

## Local setup

Prerequisites: [uv](https://docs.astral.sh/uv/) ≥ 0.13, Node 24 (see `.node-version`; e.g. via `fnm`), Docker Desktop, GNU make.

```bash
git clone https://github.com/Unmesh30/firstread.git && cd firstread
make setup        # Python + JS deps, git hooks, .env from .env.example
make up           # Postgres+pgvector, Redis, API (hot reload) on http://localhost:8000
curl localhost:8000/healthz
curl localhost:8000/readyz
make lint typecheck test
cd apps/mobile && npx expo start   # app (press w for web)
```

`make help` lists every command. The step-by-step explanation of the tooling is in [docs/learn/00-bootstrap.md](docs/learn/00-bootstrap.md).

## Repository layout

```
apps/mobile/       Expo app (web now, iOS/Android later)
services/api/      FastAPI recommender
services/ingest/   News ingestion job
ml/                Offline training & evaluation on MIND
packages/core/     Shared Python code (text, embeddings, features, schemas)
infra/             docker-compose, Dockerfiles, DB migrations, GCP scripts
docs/              architecture, ADRs, learn chapters
```

## Data and credits

- News metadata from the [GDELT Project](https://www.gdeltproject.org/).
- Models are trained and evaluated on [MIND](https://msnews.github.io/) under the Microsoft Research License (research use only; not redistributed here).

## License

[MIT](LICENSE)
