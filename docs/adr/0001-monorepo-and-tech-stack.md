# ADR 0001: Monorepo with a uv workspace, FastAPI and Expo

- Status: Accepted
- Date: 2026-10-13

## Context

FirstRead has four Python codebases (shared core, API, ingestion job, ML training) and one app. The single biggest correctness risk in a recommender is **train/serve skew**: if training and serving embed or clean text differently, offline metrics lie. We also want one place for CI, docs and versioning, and a stack that is boring, well documented and free.

## Decision

- **One repository** (monorepo) holding the app, services, ML code, infra and docs.
- **uv workspace** for Python. The root `pyproject.toml` is a virtual (non-package) root with members `packages/core`, `services/api`, `services/ingest`, `ml`. There is a single `uv.lock` and one virtualenv, and members depend on `firstread-core` via `workspace = true`. Each member uses the `src/` layout and the `uv_build` backend.
- **Python 3.12** managed by uv (`.python-version`).
- **FastAPI + uvicorn** for the API; asyncpg and redis-py (asyncio) for I/O; pydantic-settings for config; structlog for JSON logs.
- **Expo (SDK 57) with Expo Router and strict TypeScript** for the app, so one codebase serves web now and iOS/Android later. It is a standalone npm project in `apps/mobile` (no JS workspace tool, since there is only one JS package).
- **Quality tooling:** ruff (lint + format), mypy (strict for `packages/` and `services/`, relaxed for `ml/`), pytest; ESLint (expo config) + Prettier + Jest (jest-expo). pre-commit runs ruff, mypy, gitleaks and hygiene hooks.
- **Version pinning** through committed lockfiles (`uv.lock`, `package-lock.json`), exact Docker image tags, exact JS dev-dependency versions; Dependabot proposes updates.

## Alternatives considered

- **Polyrepo** (one repo per service): clean boundaries, but shared code would have to be published as a package and versioned in lockstep, which is the opposite of what we need to prevent skew.
- **Poetry / pip-tools / Pants / Bazel:** Poetry has no first-class workspaces; pip-tools would need hand-rolled glue; Pants/Bazel are powerful but heavy for a solo project. uv is fast, has workspaces and a cross-platform lockfile, and manages Python itself.
- **Flask / Django:** FastAPI gives async I/O, pydantic validation and OpenAPI generation for free, which we need for the typed app client in P4.
- **Plain React web app:** smaller, but would need a rewrite for the App Store / Play Store. Expo web exports to static files for GitHub Pages.
- **Turborepo / Nx / pnpm workspaces** for JS: unnecessary with a single JS package; revisit if we add shared TS packages.

## Consequences

- `packages/core` is imported by every Python service, so a change there is visible in all of them in the same PR, and CI tests them together.
- One lockfile means every member shares dependency versions; a heavy ML dependency (PyTorch) would be resolved for the API too. The API image installs only `--package firstread-api`, so it does not ship ML deps.
- Contributors need uv and Node 24; `make setup` installs everything else.
