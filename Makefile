# FirstRead developer commands. Run `make help` for a list.
SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env
MOBILE  := apps/mobile

.PHONY: help setup fmt lint typecheck test up down logs

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
