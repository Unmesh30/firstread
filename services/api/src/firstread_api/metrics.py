"""Prometheus metrics (one uvicorn worker per container, so no multiprocess mode).

Labels use route *templates* (``/v1/articles/{id}/why``), never raw paths, so ids
can't explode the label set. Budget counters make ADR 0011's costs observable:
every Redis command (pipelined ones included, as Upstash bills them) and an
estimate of the bytes each query returns (Supabase bills pooler egress).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["route", "method", "status"], registry=REGISTRY
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)
REDIS_COMMANDS = Counter(
    "redis_commands_total", "Redis commands sent (Upstash bills each)", ["command"],
    registry=REGISTRY,
)  # fmt: skip
DB_BYTES = Counter(
    "db_bytes_estimated_total", "Estimated Postgres result bytes (Supabase egress)", ["query"],
    registry=REGISTRY,
)  # fmt: skip
DB_QUERIES = Counter("db_queries_total", "Postgres queries", ["query"], registry=REGISTRY)
AUTH_FAILURES = Counter("auth_failures_total", "Rejected tokens", ["reason"], registry=REGISTRY)
JWKS_REFRESH = Counter("jwks_refresh_total", "JWKS fetches", ["result"], registry=REGISTRY)
RATE_LIMITED = Counter("rate_limited_total", "429 responses", ["scope"], registry=REGISTRY)
EVENTS = Counter("events_total", "Client events", ["type", "result"], registry=REGISTRY)


def render() -> bytes:
    out: bytes = generate_latest(REGISTRY)
    return out
