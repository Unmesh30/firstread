"""Seed the web app's end-to-end database (P4). Local stack and CI only, never production.

    uv run python services/api/scripts/e2e_seed.py [--per-topic 30]

1. Recreates a separate ``firstread_e2e`` database (all migrations) next to
   ``DATABASE_URL`` and flushes Redis db 3, so the E2E never touches dev data.
2. Inserts invented headlines on ``*.example.com`` outlets across the 13 topics:
   no MIND rows and no real publishers (SPEC section 4), with synthetic topic vectors
   (``firstread_api.testing``), so no embedding model is needed.
3. Adds the five demo personas and publishes the pool snapshot (with stats and the
   persona summaries) the way ingestion does.

Prints the DSN and Redis URL the E2E API should use.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

from redis.asyncio import Redis

from firstread_api.testing import (
    Seeded,
    insert_article,
    insert_persona,
    publish_pools,
    topic_vectors,
)
from firstread_core.db import apply_migrations, connect, default_migrations_dir
from firstread_core.dbtest import with_database
from firstread_core.personas import PersonaSample, PersonaSummary
from firstread_core.topics import TOPIC_IDS

DB_NAME, REDIS_DB = "firstread_e2e", 3
DEFAULT_DSN = "postgresql://firstread:firstread@localhost:5432/firstread"

OUTLETS = [
    ("The Example Ledger", "ledger.example.com"),
    ("Example Daily", "daily.example.com"),
    ("Sample Wire", "wire.example.com"),
    ("Placeholder Post", "post.example.com"),
    ("Demo Courier", "courier.example.com"),
]
# Invented headline parts per topic: subjects x events, so titles read like news.
PARTS: dict[str, tuple[list[str], list[str]]] = {
    "us": (
        ["City council", "State lawmakers", "A Midwest county", "Coastal towns"],
        [
            "approves new transit plan",
            "debates school funding",
            "braces for budget cuts",
            "opens new public library",
        ],
    ),
    "world": (
        ["European ministers", "Pacific nations", "A UN panel", "Regional leaders"],
        [
            "agree on fishing limits",
            "meet over trade routes",
            "warn of drought",
            "sign water-sharing accord",
        ],
    ),
    "politics": (
        ["Senate committee", "Governor's office", "Party leaders", "Election officials"],
        [
            "unveils ethics proposal",
            "delays key vote",
            "spar over spending bill",
            "publish turnout figures",
        ],
    ),
    "business": (
        ["Regional banks", "A grocery chain", "Shipping firms", "Startup investors"],
        [
            "report steady earnings",
            "announce hiring freeze",
            "cut delivery fees",
            "bet on battery storage",
        ],
    ),
    "tech_science": (
        ["Researchers", "A satellite team", "Chip designers", "Volunteer coders"],
        [
            "map a distant galaxy",
            "test a quieter drone",
            "unveil low-power processor",
            "release a new compiler",
        ],
    ),
    "health": (
        ["Hospitals", "A nutrition study", "Public health officials", "Clinic networks"],
        [
            "expand weekend hours",
            "links sleep to memory",
            "urge flu shots early",
            "trial shorter wait times",
        ],
    ),
    "sports": (
        ["The home team", "A rookie striker", "Marathon organizers", "The league"],
        ["wins in overtime", "scores twice in debut", "add a night race", "reviews replay rules"],
    ),
    "entertainment": (
        ["A streaming drama", "The film festival", "A touring band", "Critics"],
        [
            "renews for a second season",
            "names its jury",
            "adds ten new dates",
            "praise an animated short",
        ],
    ),
    "lifestyle": (
        ["Urban gardeners", "Home designers", "Weekend hikers", "Book clubs"],
        [
            "swap seeds at spring fair",
            "favor warmer colors",
            "find new trails",
            "pick a summer classic",
        ],
    ),
    "food_drink": (
        ["A bakery", "Coffee roasters", "Street-food vendors", "A family diner"],
        [
            "revives a rye recipe",
            "try a lighter roast",
            "gather for a night market",
            "celebrates 50 years",
        ],
    ),
    "travel": (
        ["Airlines", "A mountain town", "Rail operators", "Island ferries"],
        [
            "add winter routes",
            "limits day-trip parking",
            "restore a scenic line",
            "extend summer schedules",
        ],
    ),
    "autos": (
        ["Carmakers", "A used-car market", "EV charging firms", "Safety regulators"],
        [
            "recall seat sensors",
            "cools after record year",
            "open highway stations",
            "test new crash ratings",
        ],
    ),
    "weather": (
        ["Forecasters", "Farmers", "Coastal residents", "Ski resorts"],
        [
            "expect an early frost",
            "welcome steady rain",
            "prepare for high tides",
            "open a week early",
        ],
    ),
}
PERSONAS = [
    ("markets-tech", "Markets & tech", ["business", "tech_science"]),
    ("sports-health", "Sports & health", ["sports", "health"]),
    ("world-affairs", "World affairs", ["world", "politics"]),
    ("culture-food", "Culture & food", ["entertainment", "food_drink", "lifestyle"]),
    ("us-news", "U.S. news", ["us", "politics", "weather"]),
]


def local_only(dsn: str) -> None:
    host = urlsplit(dsn).hostname
    if host not in ("localhost", "127.0.0.1"):
        raise SystemExit(f"refusing to seed {host!r}: the E2E runs on the local stack only")


SUFFIXES = ["", " this week", " after a long review"]  # 4 x 4 x 3 = 48 distinct per topic


def headline(topic: str, k: int) -> str:
    subjects, events = PARTS[topic]
    n = len(subjects) * len(events)
    event = events[(k // len(subjects)) % len(events)]
    return f"{subjects[k % len(subjects)]} {event}{SUFFIXES[(k // n) % len(SUFFIXES)]}"


async def recreate(admin_dsn: str) -> str:
    admin = await connect(admin_dsn)
    try:
        await admin.execute(f'drop database if exists "{DB_NAME}" with (force)')
        await admin.execute(f'create database "{DB_NAME}"')
    finally:
        await admin.close()
    return with_database(admin_dsn, DB_NAME)


async def main(per_topic: int, redis_base: str) -> None:
    admin_dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    local_only(admin_dsn)
    local_only(redis_base)
    dsn = await recreate(admin_dsn)
    vecs = topic_vectors(TOPIC_IDS, per_topic, seed=4)
    conn = await connect(dsn)
    seeded: dict[str, list[Seeded]] = {}
    try:
        await apply_migrations(conn, default_migrations_dir())
        for t in TOPIC_IDS:
            for k, v in enumerate(vecs[t]):
                name, domain = OUTLETS[(k + len(t)) % len(OUTLETS)]
                a = await insert_article(conn, v, category=t, domain=domain,
                                         age_hours=0.2 + k * 0.6, title=headline(t, k))  # fmt: skip
                await conn.execute("update public.articles set source_name = $2 where id = $1",
                                   a.id, name)  # fmt: skip
                seeded.setdefault(t, []).append(a)
        summaries = []
        for pid, pname, topics in PERSONAS:
            picks = [a for t in topics for a in seeded[t][: 20 // len(topics) + 1]][:20]
            await insert_persona(conn, pid, topics, [a.id for a in picks])
            await conn.execute("update public.demo_personas set name = $2, description = $3 "
                               "where id = $1", pid, pname,
                               f"A demo reader who follows {', '.join(topics)}")  # fmt: skip
            samples = tuple(PersonaSample(headline(topics[0], k), OUTLETS[k][0], topics[0])
                            for k in range(3))  # fmt: skip
            summaries.append(PersonaSummary(
                pid, pname, f"A demo reader who follows {', '.join(topics)}", tuple(topics),
                len(picks), int(datetime.now(UTC).timestamp()), samples,
            ).to_dict())  # fmt: skip
    finally:
        await conn.close()
    redis_url = urlunsplit(urlsplit(redis_base)._replace(path=f"/{REDIS_DB}"))
    redis = Redis.from_url(redis_url)
    try:
        await redis.flushdb()
        everything = [a for t in TOPIC_IDS for a in seeded[t]]
        trending = [(seeded[t][0], 1.0 - i * 0.05, 3) for i, t in enumerate(TOPIC_IDS[:6])]
        now = datetime.now(UTC).isoformat()
        stats = {"runs_24h": 96, "runs_ok_24h": 95, "runs_partial_24h": 1,
                 "last_run_finished": now, "last_run_status": "ok",
                 "articles_24h": len(everything), "sources_24h": len(OUTLETS),
                 "median_lag_s": 840.0}  # fmt: skip
        await publish_pools(redis, everything, trending, version="e2e-1",
                            personas=summaries, stats=stats)  # fmt: skip
    finally:
        await redis.aclose()
    print(f"DATABASE_URL={dsn}")
    print(f"REDIS_URL={redis_url}")
    print(f"seeded {sum(len(v) for v in seeded.values())} articles, {len(PERSONAS)} personas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-topic", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(main(args.per_topic, os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
