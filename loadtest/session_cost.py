"""Measure what one feed session costs (ADR 0011), against the local load-test container.

    uv run python loadtest/session_cost.py --base http://127.0.0.1:8001 --sessions 20

A session (ADR 0011's definition) for a returning persona reader:

1. ``GET /v1/me``;
2. page 1 (computed);
3. events: 20 impressions + 1 click;
4. page 2 (re-ranked after the click: computed);
5. events: 20 impressions;
6. page 3 (served from the slate);
7. events: 1 click;
8. ``GET /v1/articles/{id}/why``.

Per session it reads the API's own counters from ``/metrics`` before and after
(``redis_commands_total``, ``db_bytes_estimated_total``: what Upstash and Supabase
bill), Redis' ``INFO stats`` network bytes (Upstash bandwidth, both directions), and
each request's duration rounded up to 100 ms (Cloud Run's billable instance time).
Run it right after a container restart for a **cold** instance (empty article LRU,
no pool snapshot), then again for **warm** ones. Sessions run one after another, so
nothing else touches the counters; ``INFO`` itself is measured and subtracted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import uuid
from pathlib import Path
from typing import Any

import httpx
from redis.asyncio import Redis

HERE = Path(__file__).resolve().parent
_SAMPLE = re.compile(r"^([a-z_]+)(\{[^}]*\})? ([0-9.e+-]+)$")


def counters(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _SAMPLE.match(line)
        if m:
            out[m[1]] = out.get(m[1], 0.0) + float(m[3])
    return out


async def net(redis: Redis) -> int:
    info = await redis.info("stats")
    return int(info["total_net_input_bytes"]) + int(info["total_net_output_bytes"])


async def session(client: httpx.AsyncClient, token: str) -> tuple[int, float]:
    """One session; returns (requests, billable seconds at 100 ms rounding)."""
    h = {"Authorization": f"Bearer {token}"}
    billed = 0.0
    n = 0

    async def call(method: str, url: str, **kw: Any) -> dict[str, Any]:
        nonlocal billed, n
        r = await client.request(method, url, headers=h, **kw)
        r.raise_for_status()
        billed += math.ceil(r.elapsed.total_seconds() * 10) / 10
        n += 1
        body: dict[str, Any] = r.json()
        return body

    def batch(ids: list[str], clicks: list[str]) -> dict[str, Any]:
        evs = [{"event_id": str(uuid.uuid4()), "type": "impression", "article_id": a,
                "position": k} for k, a in enumerate(ids)]  # fmt: skip
        evs += [{"event_id": str(uuid.uuid4()), "type": "click", "article_id": a} for a in clicks]
        return {"events": evs}

    await call("GET", "/v1/me")
    p1 = await call("GET", "/v1/feed", params={"limit": 20})
    ids1 = [i["id"] for i in p1["items"]]
    await call("POST", "/v1/events", json=batch(ids1, [ids1[0]]))
    p2 = await call("GET", "/v1/feed", params={"limit": 20, "cursor": p1["next_cursor"]})
    ids2 = [i["id"] for i in p2["items"]]
    await call("POST", "/v1/events", json=batch(ids2, []))
    p3 = await call("GET", "/v1/feed", params={"limit": 20, "cursor": p2["next_cursor"]})
    ids3 = [i["id"] for i in p3["items"]]
    await call("POST", "/v1/events", json=batch([], [ids3[0]]))
    await call("GET", f"/v1/articles/{ids3[1]}/why")
    return n, billed


async def main(base: str, redis_url: str, sessions: int, skip: int, label: str) -> None:
    readers = json.loads((HERE / ".tokens" / "sessions.json").read_text())[skip : skip + sessions]
    redis = Redis.from_url(redis_url)
    rows = []
    try:
        async with httpx.AsyncClient(base_url=base, timeout=10) as client:
            probe = await net(redis)
            info_cost = await net(redis) - probe  # one INFO round trip
            for u in readers:
                before = counters((await client.get("/metrics")).text)
                net0 = await net(redis)
                n, billed = await session(client, u["token"])
                net1 = await net(redis)
                after = counters((await client.get("/metrics")).text)
                rows.append({
                    "requests": n,
                    "billed_s": round(billed, 2),
                    "redis_commands": after["redis_commands_total"]
                    - before.get("redis_commands_total", 0),
                    "db_bytes_est": after["db_bytes_estimated_total"]
                    - before.get("db_bytes_estimated_total", 0),
                    "redis_net_bytes": net1 - net0 - info_cost,
                })  # fmt: skip
    finally:
        await redis.aclose()
    keys = ("requests", "billed_s", "redis_commands", "db_bytes_est", "redis_net_bytes")
    summary = {
        "label": label,
        "sessions": len(rows),
        "median": {k: statistics.median(r[k] for r in rows) for k in keys},
        "max": {k: max(r[k] for r in rows) for k in keys},
        "rows": rows,
    }
    out = HERE / "results" / f"session-{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("label", "sessions", "median", "max")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--redis-url", default="redis://localhost:6379/2")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0, help="readers already used")
    parser.add_argument("--label", default="warm")
    a = parser.parse_args()
    asyncio.run(main(a.base, a.redis_url, a.sessions, a.skip, a.label))
