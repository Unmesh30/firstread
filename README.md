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
| API | Python 3.12, FastAPI, asyncpg, redis-py, ONNX Runtime, on Google Cloud Run; a seven-stage feed with per-stage deadlines and tested degraded modes, the user encoder micro-batched across requests, HNSW candidates ([ADR 0017](docs/adr/0017-feed-serving-pipeline.md)); demo personas from live news, "Why this?" and public stats ([ADR 0018](docs/adr/0018-personas-why-stats.md)) |
| Ingestion | GDELT GKG files + RSS feeds with explicit publisher permission, every 15 min on GitHub Actions; the 72 h dedupe window stays on the runner (~84 KB of database egress a run) and pools are published as one versioned Redis snapshot ([ADR 0012](docs/adr/0012-runner-window-and-pool-snapshot.md)) |
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

Try the feed locally (the dev stack accepts `X-Dev-User` instead of a token):

```bash
make db-migrate && make ingest     # one live ingestion run into the local database
U=$(uuidgen | tr A-Z a-z); H="X-Dev-User: $U"
curl -X PUT localhost:8000/v1/me/topics -H "$H" -H 'Content-Type: application/json' \
  -d '{"topics": ["world", "business", "tech_science"]}'
curl -i "localhost:8000/v1/feed?limit=5" -H "$H"    # see the Server-Timing header
```

Or try a demo reader built from live news, and ask why an item was shown:

```bash
curl localhost:8000/v1/demo/personas
D="X-Dev-User: $(uuidgen | tr A-Z a-z)"     # a new reader (one with topics gets 409 unless ?replace=true)
curl -X POST localhost:8000/v1/demo/personas/markets-tech/activate -H "$D"
curl "localhost:8000/v1/feed?limit=5" -H "$D"
curl "localhost:8000/v1/articles/<an id from that feed>/why" -H "$D"
curl localhost:8000/v1/stats
```

**Local performance** (API container at Cloud Run's shape: 1 vCPU, 512 MiB; Postgres and Redis local): p95 server time **21 ms at 50 RPS**, ~70 RPS per instance before it saturates, so SPEC's 100 RPS takes two instances. A feed session costs 30–37 Redis commands and 153–373 KB of database egress, which keeps the demo free up to ~250–300 sessions a day ([learn chapter 03](docs/learn/03-recommendation-api.md), `make loadtest`).

`make help` lists every command. The step-by-step explanation of the tooling is in [docs/learn/00-bootstrap.md](docs/learn/00-bootstrap.md).

### Offline evaluation on MIND

MIND is license-gated: read the [terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf) first.

```bash
make mind-download VARIANT=small ACCEPT_LICENSE=1   # verified download into data/ (gitignored)
make reproduce VARIANT=small                         # EDA, 4 baselines + NRMS-lite (3 seeds), classifier (~50 min)
make reproduce VARIANT=large                         # final numbers (~9 h on an M-series Mac, overnight)
make topics                                          # 13-topic classifier on MIND-large (ADR 0006)
make skew                                            # title-only vs title+abstract ranking study
```

MIND-large dev, nDCG@10 (NRMS-lite: mean ± std over 3 seeds; gain: paired bootstrap 95% CI):

| | Best baseline (mean_history) | NRMS-lite | Gain |
|---|---:|---:|---|
| Overall | 0.3906 | **0.4145 ± 0.0004** | +0.0239 [+0.0232, +0.0248] |
| Cold-start candidates | 0.5332 | **0.5506 ± 0.0008** | +0.0174 [+0.0159, +0.0188] |

Results and the evaluation protocol: [ml/reports/](ml/reports/), [ADR 0003](docs/adr/0003-mind-data-and-eval-protocol.md), [docs/learn/01-offline-ml.md](docs/learn/01-offline-ml.md).

### Models on the Hugging Face Hub

The trained models are released as ONNX at **[Unmesh30/firstread-models](https://huggingface.co/Unmesh30/firstread-models)**. Current release: [`v0.2.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.2.0) (adds the 13-topic classifier; the user encoder is byte-identical to [`v0.1.0`](https://huggingface.co/Unmesh30/firstread-models/tree/v0.1.0)).

- **What's in it:** the NRMS-lite user encoder and the topic classifier ([ADR 0006](docs/adr/0006-topic-taxonomy-and-classifier.md)), plus a SHA256 `manifest.json` and a [model card](docs/model-card.md).
- **Headlines without summaries:** title-only ranking costs the user encoder 2.1% relative nDCG@10 (MIND-small dev). On live news at steady state the topic classifier is **0.79 accurate** (95% CI 0.70–0.88; LLM-labeled samples spot-checked by a second, independent LLM session, approved by the owner; [learn chapter 02](docs/learn/02-live-ingestion.md)).
- **License:** the weights are MIND-derived and **non-commercial**, under the [MSR License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf), not MIT.
- **Latency:** ONNX Runtime on one CPU thread (Apple M4) encodes a 50-click history in 0.17 ms p50.

```bash
make release-verify VERSION=0.2.0     # re-download from the Hub and check every SHA256
```

How releases are built, gated and tagged: [ADR 0005](docs/adr/0005-model-artifacts-on-hf-hub.md).

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
- Models are trained and evaluated on [MIND](https://msnews.github.io/) under the Microsoft Research License (research use only; not redistributed here). See [docs/data-card.md](docs/data-card.md). The released weights are MIND-derived and non-commercial; see the [model card](docs/model-card.md).

## License

Code: [MIT](LICENSE). Model weights on the Hugging Face Hub: non-commercial, MIND-derived (see the [model card](docs/model-card.md)).
