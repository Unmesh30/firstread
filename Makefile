# FirstRead developer commands. Run `make help` for a list.
SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env
MOBILE  := apps/mobile

.PHONY: help setup fmt lint typecheck test up down logs db-migrate ingest mind-download eda eval eval-gate eval-gate-real topics skew reproduce mlflow-ui release-build release-upload release-verify release-verify-tag

VARIANT ?= small

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python + JS deps, git hooks, and create .env
	uv sync --all-packages
	cd $(MOBILE) && npm ci
	uv run pre-commit install
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

fmt: ## Auto-format and auto-fix (ruff, prettier)
	uv run ruff format .
	uv run ruff check --fix .
	cd $(MOBILE) && npm run -s format

lint: ## Lint Python (ruff) and the app (eslint, prettier)
	uv run ruff check .
	uv run ruff format --check .
	cd $(MOBILE) && npm run -s lint
	cd $(MOBILE) && npm run -s format:check

typecheck: ## Type-check Python (mypy) and the app (tsc)
	uv run mypy
	cd $(MOBILE) && npm run -s typecheck

test: ## Run Python (pytest) and app (jest) tests
	uv run pytest
	cd $(MOBILE) && npm test

up: ## Start the local stack (Postgres+pgvector, Redis, API with hot reload)
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build --wait
	$(COMPOSE) ps

down: ## Stop the local stack (keeps the database volume)
	$(COMPOSE) down

logs: ## Follow logs from the local stack
	$(COMPOSE) logs -f --tail=100

# ---- Ingestion (Phase 2) ----
db-migrate: ## Apply infra/db/migrations to DATABASE_URL (local stack by default)
	uv run firstread-ingest migrate

ingest: ## One ingestion run against the local stack (DRY_RUN=1 writes nothing)
	uv run firstread-ingest run $(if $(DRY_RUN),--dry-run)

# ---- Offline ML (Phase 1) ----
mind-download: ## Download + verify MIND (VARIANT=small|large, needs ACCEPT_LICENSE=1)
	uv run firstread-ml download --variant $(VARIANT) $(if $(ACCEPT_LICENSE),--accept-license)

eda: ## Aggregate EDA report for MIND (VARIANT=small|large) -> ml/reports/
	uv run firstread-ml eda --variant $(VARIANT)

eval: ## Reproduce the baseline eval: prepare, embed, holdout + dev reports (VARIANT=small|large)
	uv run firstread-ml eval --variant $(VARIANT) --split holdout
	uv run firstread-ml eval --variant $(VARIANT) --split dev

ALL_MODELS := random,popularity,recency,mean_history,nrms_lite
SEEDS ?= 0,1,2

reproduce: ## All Phase 1 numbers for VARIANT: EDA, holdout + dev eval incl. NRMS-lite, classifier
	uv run firstread-ml eda --variant $(VARIANT)
	uv run firstread-ml eval --variant $(VARIANT) --split holdout --models $(ALL_MODELS) --seeds $(SEEDS)
	uv run firstread-ml eval --variant $(VARIANT) --split dev --models $(ALL_MODELS) --seeds $(SEEDS)
	uv run firstread-ml classify --variant $(VARIANT)

eval-gate: ## Synthetic eval-regression gate (no MIND needed; also runs in CI via pytest)
	uv run firstread-ml eval-gate

RELEASE ?= data/release/dry-run
topics: ## Train + evaluate the topic classifier on MIND-large (ADR 0006) -> ml/reports/
	uv run firstread-ml topics --variant large

skew: ## Title-only vs title+abstract ranking study on MIND-small dev (needs data/release/v0.1.0)
	uv run firstread-ml skew --release data/release/v0.1.0

eval-gate-real: ## Real-data gate: ONNX user encoder in RELEASE vs mean_history on MIND-small dev (local)
	uv run firstread-ml eval-gate --real --release $(RELEASE)

# ---- Model release (Phase 1c, ADR 0005) ----
release-build: ## Build data/release/vVERSION (DRY_RUN=1 on a branch, INHERIT=0.1.0 to reuse that encoder)
	@test -n "$(VERSION)" || (echo "usage: make release-build VERSION=0.1.0 [DRY_RUN=1]"; exit 2)
	uv run firstread-ml release build --version $(VERSION) $(if $(DRY_RUN),--dry-run) $(if $(INHERIT),--inherit-user-encoder $(INHERIT))

# .env is sourced in a subshell; the token is never echoed.
release-upload: ## Upload data/release/vVERSION to the HF Hub, tag it and verify hashes (needs HF_TOKEN)
	@test -n "$(VERSION)" || (echo "usage: make release-upload VERSION=0.1.0"; exit 2)
	@set -a && . ./.env && set +a && uv run firstread-ml release upload --version $(VERSION)

release-verify: ## Re-download vVERSION from the HF Hub and check every SHA256
	uv run firstread-ml release verify --version $(VERSION)

release-verify-tag: ## Check that git tag vVERSION is the commit recorded in the HF manifest
	uv run firstread-ml release verify-tag --version $(VERSION)

mlflow-ui: ## Browse local MLflow runs (sqlite store in data/mlruns) at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///$(CURDIR)/data/mlruns/mlflow.db --host 127.0.0.1
