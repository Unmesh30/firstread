# syntax=docker/dockerfile:1
# FirstRead API image. Build context = repository root:
#   docker build -f infra/docker/api.Dockerfile --target runtime -t firstread-api .

ARG PYTHON_IMAGE=python:3.12.15-slim

# ---- builder: resolve and install dependencies with uv ----
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# 1) Third-party deps only (cached layer; changes only when the lockfile changes).
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY services/api/pyproject.toml services/api/pyproject.toml
COPY services/ingest/pyproject.toml services/ingest/pyproject.toml
COPY ml/pyproject.toml ml/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package firstread-api --no-install-workspace

# 2) Our own code.
COPY packages/core packages/core
COPY services/api services/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package firstread-api --no-editable

# ---- runtime: slim image with just the virtualenv ----
FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000
USER app
EXPOSE 8000
# Cloud Run injects $PORT; default 8000 locally.
CMD ["sh", "-c", "exec uvicorn firstread_api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]

# ---- dev: editable install + hot reload (used by docker-compose) ----
FROM builder AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package firstread-api
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "firstread_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", \
     "--reload-dir", "/app/services/api/src", "--reload-dir", "/app/packages/core/src", "--no-access-log"]
