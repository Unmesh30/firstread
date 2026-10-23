#!/usr/bin/env bash
# Local p95 run (P3e). Local Docker stack only, never production.
#
#   make up                                # db + redis (the dev API may keep running)
#   uv run python loadtest/seed_local.py   # firstread_loadtest DB, Redis db 2, tokens
#   loadtest/run_local.sh                  # this script
#
# Runs the runtime image at Cloud Run's shape (1 vCPU, 512 MiB), real ES256 JWT
# verification against loadtest/.tokens/jwks.json, rate limits raised for the test;
# then measures startup-to-ready (median of 5) and runs k6 (pinned image).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=firstread-api:loadtest
NAME=firstread-api-lt
NET=firstread_default
K6=grafana/k6:2.3.0
PORT=8001
OUT=loadtest/results
mkdir -p "$OUT"

docker build -q -f infra/docker/api.Dockerfile --target runtime -t "$IMAGE" . >/dev/null
docker image inspect "$IMAGE" --format '{{.Size}}' > "$OUT/image-bytes.txt"

start() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --network "$NET" -p "127.0.0.1:$PORT:8000" \
    --cpus=1 --memory=512m \
    -e APP_ENV=local \
    -e DATABASE_URL=postgresql://firstread:firstread@db:5432/firstread_loadtest \
    -e REDIS_URL=redis://redis:6379/2 \
    -e AUTH_LOCAL_JWKS_FILE=/lt/jwks.json \
    -e AUTH_ISSUER=https://test-project.supabase.co/auth/v1 \
    -e RL_IP_PER_MIN=10000000 -e RL_USER_FEED_PER_MIN=1000000 \
    -e RL_USER_EVENTS_PER_MIN=1000000 -e RL_USER_WRITES_PER_MIN=1000000 \
    -e LOG_LEVEL=WARNING \
    -v "$PWD/loadtest/.tokens:/lt:ro" "$IMAGE" >/dev/null
}

ready_ms() {  # milliseconds from `docker run` to the first /readyz 200
  local t0 t1
  t0=$(python3 -c 'import time; print(time.time())')
  start
  until curl -fs "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; do sleep 0.05; done
  t1=$(python3 -c 'import time; print(time.time())')
  python3 -c "print(round(($t1 - $t0) * 1000))"
}

: > "$OUT/startup-ms.txt"
for _ in 1 2 3 4 5; do ready_ms >> "$OUT/startup-ms.txt"; done
echo "startup to /readyz (ms): $(tr '\n' ' ' < "$OUT/startup-ms.txt")"

# Fresh container for the run (cold caches, like a new Cloud Run instance).
start
until curl -fs "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; do sleep 0.05; done
docker run --rm --network "$NET" -v "$PWD/loadtest:/lt" "$K6" run --quiet \
  --out "csv=/lt/results/raw.csv" -e BASE="http://$NAME:8000" /lt/k6/feed.js
docker stats --no-stream --format '{{.MemUsage}}' "$NAME" > "$OUT/memory-after.txt"
curl -fs "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-after.txt"
echo "k6 done: $OUT/k6-summary.json"
