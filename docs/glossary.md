# Glossary

Project-wide terms. Each learn chapter also has its own glossary.

| Term | Meaning |
|---|---|
| **ADR** | Architecture Decision Record: a short doc recording a decision, the context, the alternatives and the consequences. Lives in `docs/adr/`. |
| **ANN** | Approximate nearest neighbour search: finding the most similar vectors quickly by trading a little accuracy for speed (e.g. pgvector HNSW). |
| **Cold start (item)** | A new article has no clicks, so click-based recommenders can't rank it. FirstRead's core problem. |
| **Cold start (user)** | A new reader has no history; we start from their chosen topics. |
| **Conventional Commits** | Commit message format `type: summary` (`feat:`, `fix:`, `docs:`...). |
| **CI** | Continuous Integration: every push/PR is automatically linted, type-checked, tested and built. |
| **Docker Compose** | Tool that starts several containers (db, redis, api) together from one YAML file. |
| **Embedding** | A fixed-length vector (here 384 numbers) representing the meaning of a text. |
| **Expo Router** | File-based routing for Expo/React Native: each file in `src/app/` is a screen/URL. |
| **GDELT** | Global Database of Events, Language and Tone; our free live news source. Must be credited with a link to gdeltproject.org. |
| **Healthcheck** | A command/endpoint that tells an orchestrator whether a service is working. |
| **Liveness vs readiness** | `/healthz`: is the process alive? `/readyz`: can it serve traffic (dependencies reachable)? |
| **Lockfile** | A file (`uv.lock`, `package-lock.json`) recording the exact version of every dependency so installs are reproducible. |
| **MIND** | Microsoft News Dataset; used for training and offline evaluation only (research license). |
| **mypy** | Static type checker for Python. |
| **nDCG / MRR / AUC** | Ranking-quality metrics used on MIND. |
| **ONNX** | Portable model format; ONNX Runtime runs it fast on CPU without PyTorch. |
| **pgvector** | Postgres extension adding vector columns and similarity search. |
| **pre-commit** | Framework that runs checks (format, lint, secret scan) before each git commit. |
| **ruff** | Very fast Python linter and formatter. |
| **SLO** | Service Level Objective: a measurable target such as p95 latency ≤ 150 ms. |
| **Train/serve skew** | When training and production compute features differently, so offline results don't hold in production. |
| **uv** | Fast Python package and project manager (installs Python, resolves and locks dependencies, runs commands). |
| **uv workspace** | Several Python packages in one repo that share a lockfile and virtualenv. |
