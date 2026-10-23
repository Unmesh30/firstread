# syntax=docker/dockerfile:1
# FirstRead API image. Build context = repository root:
#   docker build -f infra/docker/api.Dockerfile --target runtime -t firstread-api .

ARG PYTHON_IMAGE=python:3.12.15-slim
# The model release the image serves; it must be pinned in firstread_core.release.PINNED.
ARG MODEL_TAG=v0.2.0

# ---- models: fetch the pinned encoder release and verify it (build time only) ----
# Downloads at the pinned Hub commit and checks the manifest SHA256, each file's hash
# and the manifest's feature blocks against packages/core. huggingface_hub lives only
# in this stage; the runtime image gets the three verified files and nothing else.
FROM ${PYTHON_IMAGE} AS models
ARG MODEL_TAG
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/release-env \
    FIRSTREAD_MODEL_CACHE=/tmp/model-cache
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY services/api/pyproject.toml services/api/pyproject.toml
COPY services/ingest/pyproject.toml services/ingest/pyproject.toml
COPY ml/pyproject.toml ml/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package firstread-core --extra release --no-install-workspace
COPY packages/core packages/core
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package firstread-core --extra release --no-editable
RUN /opt/release-env/bin/python -m firstread_core.release "${MODEL_TAG}" "/models/${MODEL_TAG}" \
    user_encoder.onnx population.npy

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

# ---- runtime: slim image with just the virtualenv and the verified model files ----
FROM ${PYTHON_IMAGE} AS runtime
ARG MODEL_TAG
RUN useradd --create-home --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=models /models /app/models
# The API re-verifies these files at startup (hashes, feature blocks): no network.
# One BLAS thread: numpy's OpenBLAS otherwise starts a thread per *visible* host CPU,
# and those threads spin under Cloud Run's 1-vCPU quota. Measured locally at 1 vCPU:
# CPU throttled in 88% of scheduler periods and requests failing from 10 RPS, vs 0
# errors and a 21 ms p95 at 50 RPS with one thread (docs/learn/03, §6.2).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PORT=8000 \
    API_MODEL_TAG=${MODEL_TAG} \
    API_MODEL_DIR=/app/models/${MODEL_TAG}
USER app
EXPOSE 8000
# Cloud Run injects $PORT; default 8000 locally.
# One worker per container: prometheus-client then needs no multiprocess mode, and
# Cloud Run scales by instances. uvicorn trusts X-Forwarded-For only from
# FORWARDED_ALLOW_IPS (default 127.0.0.1; the Cloud Run value is set in P5).
CMD ["sh", "-c", "exec uvicorn firstread_api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --no-access-log"]

# ---- dev: editable install + hot reload (used by docker-compose) ----
FROM builder AS dev
ARG MODEL_TAG
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package firstread-api
COPY --from=models /models /app/models
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    API_MODEL_TAG=${MODEL_TAG} API_MODEL_DIR=/app/models/${MODEL_TAG}
EXPOSE 8000
CMD ["uvicorn", "firstread_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", \
     "--reload-dir", "/app/services/api/src", "--reload-dir", "/app/packages/core/src", "--no-access-log"]
