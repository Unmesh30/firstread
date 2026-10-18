"""Replay archived GDELT GKG files through the real pipeline with a stepped clock.

Steady-state numbers (volumes, pools, trending, database size, the topic audit)
shouldn't need a week of waiting. GDELT keeps every 15-minute GKG file, so replay:

1. downloads the window's files once into ``data/replay/gkg/`` (serial, polite,
   resumable; a slot that still 404s is recorded as missing);
2. uses a separate database (``firstread_replay``) and Redis db 1, never the dev ones;
3. for each slot ``s`` calls the unchanged ``pipeline.run`` with ``now = s + 20 min``
   and an ``httpx`` client on a ``MockTransport`` that serves ``lastupdate.txt`` for
   ``s``, the cached zips and the cached domain lookup. Any other request is an
   error, so the replay loop can't touch the network. The cursor, per-run cap,
   dedupe windows, coverage, retention and pools all behave as in production;
4. snapshots the top 10 trending stories at 23:45 of each replayed day, and writes
   the full report (with headlines) to ``data/replay/report.json`` and aggregates
   only (no headlines or URLs) to ``docs/learn/data/02-replay.json``.

RSS can't be replayed (feeds only show their current items), so replay is
GDELT-only. ``rss_estimate`` measures the live feeds' daily volume and topic mix
separately, and the report shows totals with and without that RSS term.

Re-running continues from the replay database's cursor; ``--fresh`` starts over.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from firstread_core.db import apply_migrations, connect, default_migrations_dir
from firstread_core.topics import TOPIC_IDS
from firstread_ingest import pools, store
from firstread_ingest.config import Settings
from firstread_ingest.http import fetch_with_retry, make_client
from firstread_ingest.models import TopicClassifier
from firstread_ingest.pipeline import EmbedFn, run
from firstread_ingest.sources import gkg
from firstread_ingest.sources.gdelt_domains import (
    LOOKUP_URL,
    DomainCountries,
    ensure_lookup,
    load_domain_countries,
)

REPO = Path(__file__).resolve().parents[4]
CACHE = REPO / "data" / "replay" / "gkg"
REPORT = REPO / "data" / "replay" / "report.json"
AGGREGATES = REPO / "docs" / "learn" / "data" / "02-replay.json"
RUN_DELAY = timedelta(minutes=20)  # each slot is "processed" 20 min after it starts
SNAPSHOT_TIME = (23, 45)  # top-10 trending per replayed day, at this slot


def window(end: datetime, days: int) -> list[datetime]:
    """``days`` x 96 slots ending at ``end`` (inclusive), oldest first."""
    end = gkg.slot_floor(end)
    n = days * 96
    return [end - gkg.SLOT * (n - 1 - i) for i in range(n)]


def default_end(now: datetime) -> datetime:
    """The newest slot that is surely complete: 45 minutes back."""
    return gkg.slot_floor(now - timedelta(minutes=45))


def cached(slot: datetime, cache: Path = CACHE) -> Path:
    return cache / f"{gkg.slot_name(slot)}.gkg.csv.zip"


async def download(
    client: httpx.AsyncClient,
    slots: Sequence[datetime],
    cache: Path = CACHE,
    spacing: float = 1.0,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> dict[str, list[str]]:
    """Fetch every slot not already cached. Returns the missing (404) and failed slots."""
    cache.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (one-off batch tool, local disk)
    missing: list[str] = []
    failed: list[str] = []
    todo = [s for s in slots if not cached(s, cache).is_file()]
    for i, slot in enumerate(todo):
        name = gkg.slot_name(slot)
        try:
            r = await fetch_with_retry(client, gkg.slot_url(slot))
            if r.status_code == 404:
                missing.append(name)
            else:
                r.raise_for_status()
                tmp = cached(slot, cache).with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.rename(cached(slot, cache))
        except Exception:
            failed.append(name)
        if (i + 1) % 24 == 0:
            print(f"  downloaded {i + 1}/{len(todo)}", flush=True)
        if i + 1 < len(todo):
            await sleep(spacing)
    return {"missing": missing, "failed": failed}


def replay_transport(slot: datetime, cache: Path, lookup: Path) -> httpx.MockTransport:
    """Serve GDELT as it looked when ``slot`` was the newest file. Nothing else."""
    newest = cached(slot, cache)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == gkg.LASTUPDATE:
            name = gkg.slot_name(slot)
            blob = newest.read_bytes() if newest.is_file() else b""
            md5 = hashlib.md5(blob).hexdigest()  # noqa: S324 (GDELT publishes MD5)
            line = f"{len(blob)} {md5} http://data.gdeltproject.org/gdeltv2/{name}.gkg.csv.zip"
            return httpx.Response(200, text=line + "\n")
        if url.startswith(gkg.BASE) and url.endswith(".gkg.csv.zip"):
            name = url.rsplit("/", 1)[-1].split(".", 1)[0]
            path = cache / f"{name}.gkg.csv.zip"
            if path.is_file() and gkg.parse_slot(name) <= slot:
                return httpx.Response(200, content=path.read_bytes())
            return httpx.Response(404)
        if url == LOOKUP_URL:
            return httpx.Response(200, content=lookup.read_bytes())
        raise RuntimeError(f"replay made an unexpected request: {url}")

    return httpx.MockTransport(handler)


@dataclass
class Replay:
    runs: list[dict[str, Any]] = field(default_factory=list)
    trending_days: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    downloads: dict[str, list[str]] = field(default_factory=dict)


async def snapshot_trending(conn: Any, now: datetime) -> list[dict[str, Any]]:
    """Top 10 trending stories with their topic, breadth and top outlets."""
    top = await pools.trending(conn, now, limit=10)
    out = []
    for t in top:
        art = await conn.fetchrow(
            "select title, category from public.articles where id = $1::uuid", t["id"]
        )
        doms = await conn.fetch(
            "select source_domain from public.story_coverage where story_cluster_id = $1 "
            "order by first_seen_at limit 5",
            t["story_cluster_id"],
        )
        out.append(
            {
                "headline": art["title"],
                "topic": art["category"],
                "domains": t["domains"],
                "score": round(t["score"], 3),
                "top_domains": [d["source_domain"] for d in doms],
            }
        )
    return out


async def replay(
    settings: Settings,
    classifier: TopicClassifier,
    slots: Sequence[datetime],
    conn: Any,
    redis: Any,
    places: DomainCountries,
    lookup: Path,
    cache: Path = CACHE,
    embed: EmbedFn | None = None,
    progress: Callable[[str], None] = print,
) -> Replay:
    """Run the pipeline once per slot with the clock set to that slot (GDELT only)."""
    result = Replay()
    cursor = await store.load_cursor(conn, gkg.SOURCE_ID)
    start = gkg.parse_slot(cursor) if cursor else None
    todo = [s for s in slots if start is None or s > start]
    for i, slot in enumerate(todo):
        now = slot + RUN_DELAY
        async with httpx.AsyncClient(transport=replay_transport(slot, cache, lookup)) as client:
            summary = await run(
                settings,
                classifier,
                embed=embed,
                sources=("gdelt",),
                places=places,
                client=client,
                conn=conn,
                redis=redis,
                now=now,
            )
        d = summary.as_dict()
        result.runs.append(
            {
                "slot": gkg.slot_name(slot),
                **{
                    k: d[k]
                    for k in (
                        "inserted",
                        "duplicates",
                        "near_duplicates",
                        "clustered",
                        "errors",
                        "redis_commands",
                        "status",
                    )
                },
                "dropped": d["details"]["dropped"],
                "gdelt": {
                    k: summary.gdelt.get(k)
                    for k in ("records", "dropped", "candidates", "selected", "capped", "stale")
                },
                "files": [
                    {k: f.get(k) for k in ("slot", "status", "records", "pending", "missing")}
                    for f in summary.gdelt.get("files", [])
                ],
                "topics": d["details"]["topics"],
                "retention": d["details"]["retention"],
                "seconds": d["details"]["seconds"].get("total"),
            }
        )
        if (slot.hour, slot.minute) == SNAPSHOT_TIME or i == len(todo) - 1:
            result.trending_days[slot.strftime("%Y-%m-%d")] = await snapshot_trending(conn, now)
        if (i + 1) % 8 == 0 or i == len(todo) - 1:
            done = sum(r["inserted"] for r in result.runs)
            progress(
                f"  replayed {i + 1}/{len(todo)} slots ({gkg.slot_name(slot)}), {done} inserted"
            )
    return result


async def final_state(conn: Any, redis: Any) -> dict[str, Any]:
    """Pool sizes, table sizes and article mix at the end of the replay."""
    await conn.execute("vacuum analyze public.articles")
    await conn.execute("vacuum analyze public.story_coverage")
    keys = [pools.FRESH, pools.TRENDING, *(pools.topic_key(t) for t in TOPIC_IDS)]
    zcard = {k: int(await redis.zcard(k)) for k in keys}
    sizes = await conn.fetchrow(
        "select count(*) as articles, "
        "pg_relation_size('public.articles') as heap, "
        "pg_indexes_size('public.articles') as indexes, "
        "pg_relation_size('public.articles_embedding_hnsw') as hnsw, "
        "pg_total_relation_size('public.articles') as total, "
        "pg_total_relation_size('public.story_coverage') as coverage, "
        "(select count(*) from public.story_coverage) as coverage_rows, "
        "pg_database_size(current_database()) as database "
        "from public.articles"
    )
    mix = await conn.fetch(
        "select category, origin, geo, count(*) as n from public.articles group by 1, 2, 3"
    )
    return {
        "zcard": zcard,
        "sizes": dict(sizes),
        "mix": [dict(r) for r in mix],
    }


def press_release_counts(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_rule: Counter[str] = Counter()
    wires = paths = records = 0
    for r in runs:
        for k, v in r["dropped"].items():
            if k.startswith("press_release:"):
                by_rule[k.split(":", 1)[1]] += v
        g = r["gdelt"]
        wires += (g.get("dropped") or {}).get("press_wire_domain", 0)
        paths += (g.get("dropped") or {}).get("press_wire_path", 0)
        records += g.get("records") or 0
    titles = sum(by_rule.values())
    return {
        "press_wire_domain": wires,
        "press_wire_path": paths,
        "announcement_titles": titles,
        "by_pattern": dict(by_rule.most_common()),
        "gdelt_records": records,
        "share_of_gdelt_records": round((wires + paths + titles) / records, 4) if records else None,
    }


def size_estimate(
    state: dict[str, Any], gdelt_per_day: float, rss_per_day: float, budget_mb: int = 400
) -> dict[str, Any]:
    s = state["sizes"]
    per_article = s["total"] / max(s["articles"], 1)
    out = {
        "bytes_per_article": round(per_article),
        "heap_per_article": round(s["heap"] / max(s["articles"], 1)),
        "indexes_per_article": round(s["indexes"] / max(s["articles"], 1)),
        "hnsw_per_article": round(s["hnsw"] / max(s["articles"], 1)),
        "coverage_bytes_48h": s["coverage"],
        "budget_mb": budget_mb,
    }
    for label, per_day in (
        ("gdelt_only", gdelt_per_day),
        ("with_rss", gdelt_per_day + rss_per_day),
    ):
        mb = (per_article * per_day * 14 + s["coverage"]) / 1e6
        out[label] = {
            "articles_per_day": round(per_day),
            "mb_14_days": round(mb, 1),
            "within_budget": mb < budget_mb,
        }
    return out


def aggregates(report: dict[str, Any]) -> dict[str, Any]:
    """The committed copy: no headlines or URLs (same rule as the threshold pairs)."""
    agg: dict[str, Any] = json.loads(json.dumps(report, default=str))
    for day in agg.get("trending_days", {}).values():
        for story in day:
            story.pop("headline", None)
    agg.pop("runs", None)  # per-run detail stays local; totals below are enough
    return agg


async def rss_estimate(
    settings: Settings, classifier: TopicClassifier, embed: EmbedFn
) -> dict[str, Any]:
    """Daily volume and topic mix of the live RSS feeds (RSS can't be replayed)."""
    from firstread_ingest.pipeline import run as run_pipeline

    now = datetime.now(UTC)
    summary = await run_pipeline(
        settings, classifier, embed=embed, sources=("rss",), dry_run=True, now=now
    )
    # Items newer than the 72 h staleness cutoff, after dedupe among themselves.
    per_day = summary.inserted / 3
    return {
        "measured_at": now.isoformat(timespec="seconds"),
        "items_72h_after_dedupe": summary.inserted,
        "per_day": round(per_day, 1),
        "topics": dict(summary.topics),
        "note": "RSS feeds expose only their current items, so this is an estimate "
        "from one fetch: items published in the last 72 h, deduped, divided by 3.",
    }


async def main(days: int, end: datetime | None, db: str, redis_db: int, fresh: bool) -> None:
    from redis.asyncio import Redis

    from firstread_core.dbtest import with_database
    from firstread_core.embedding import Embedder
    from firstread_ingest.main import load_classifier

    settings = Settings()
    end = end or default_end(datetime.now(UTC))
    slots = window(end, days)
    print(f"replay {gkg.slot_name(slots[0])} .. {gkg.slot_name(slots[-1])} ({len(slots)} slots)")
    async with make_client() as client:
        lookup = await ensure_lookup(client)
        downloads = await download(client, slots)
    print(f"downloads: {len(downloads['missing'])} missing, {len(downloads['failed'])} failed")
    admin_dsn = settings.database_url.get_secret_value()
    admin = await connect(admin_dsn)
    try:
        exists = await admin.fetchval("select 1 from pg_database where datname = $1", db)
        if fresh and exists:
            await admin.execute(f'drop database "{db}" with (force)')
            exists = None
        if not exists:
            await admin.execute(f'create database "{db}"')
    finally:
        await admin.close()
    conn = await connect(with_database(admin_dsn, db))
    redis_url = settings.redis_url.get_secret_value().rsplit("/", 1)[0] + f"/{redis_db}"
    redis = Redis.from_url(redis_url)
    try:
        await apply_migrations(conn, default_migrations_dir())
        if fresh or not exists:
            await redis.flushdb()
        classifier = load_classifier(settings)
        embedder = Embedder()
        places = load_domain_countries(lookup)
        result = await replay(
            settings, classifier, slots, conn, redis, places, lookup, embed=embedder.embed
        )
        state = await final_state(conn, redis)
        rss = await rss_estimate(settings, classifier, embedder.embed)
    finally:
        await redis.aclose()
        await conn.close()

    previous = json.loads(REPORT.read_text()) if REPORT.is_file() and not fresh else {}
    runs = previous.get("runs", []) + result.runs
    trending_days = previous.get("trending_days", {}) | result.trending_days
    replayed_days = max(len({r["slot"][:8] for r in runs}), 1)
    gdelt_per_day = sum(r["inserted"] for r in runs) / (len(runs) / 96) if runs else 0.0
    totals: dict[str, Any] = defaultdict(int)
    for r in runs:
        for k in (
            "inserted",
            "duplicates",
            "near_duplicates",
            "clustered",
            "errors",
            "redis_commands",
        ):
            totals[k] += r[k]
    report = {
        "window": {
            "first": runs[0]["slot"] if runs else None,
            "last": runs[-1]["slot"] if runs else None,
            "slots": len(runs),
            "days": replayed_days,
        },
        "downloads": downloads,
        "totals": dict(totals),
        "redis_commands_per_run": {
            "max": max((r["redis_commands"] for r in runs), default=0),
            "mean": round(totals["redis_commands"] / max(len(runs), 1), 1),
        },
        "statuses": dict(Counter(r["status"] for r in runs)),
        "missing_or_pending_files": sum(
            1 for r in runs for f in r["files"] if f.get("missing") or f.get("pending")
        ),
        "press_releases": press_release_counts(runs),
        "trending_days": trending_days,
        "final": state,
        "rss_estimate": rss,
        "size_estimate": size_estimate(state, gdelt_per_day, rss["per_day"]),
        "runs": runs,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=str) + "\n")
    AGGREGATES.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATES.write_text(json.dumps(aggregates(report), indent=1, default=str) + "\n")
    print(f"wrote {REPORT} and {AGGREGATES}")
