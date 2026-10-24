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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from firstread_core.db import apply_migrations, connect, default_migrations_dir
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
WINDOW_FILE = REPO / "data" / "replay" / "window.npz"  # the same snapshot path as cron
# Phase 2's committed copy (02-replay.json) is history; P3 replays write 03-*.
AGGREGATES = REPO / "docs" / "learn" / "data" / "03-replay.json"
EQUIVALENCE = REPO / "docs" / "learn" / "data" / "03-replay-equivalence.json"
AfterRun = Callable[[datetime, datetime], Awaitable[None]]
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
    trending_days: dict[str, dict[str, Any]] = field(default_factory=dict)
    downloads: dict[str, list[str]] = field(default_factory=dict)


async def _describe(
    conn: Any, rep: Any, clusters: Sequence[Any], score: float, outlets: int
) -> dict[str, Any]:
    art = await conn.fetchrow("select title, category from public.articles where id = $1", rep)
    doms = await conn.fetch(
        "select source_domain from public.story_coverage where story_cluster_id = any($1) "
        "order by first_seen_at limit 5",
        list(clusters),
    )
    return {
        "headline": art["title"],
        "topic": art["category"],
        "domains": outlets,
        "score": round(score, 3),
        "clusters": [str(c) for c in clusters],
        "top_domains": [d["source_domain"] for d in doms],
    }


NETWORK_SHARE_SQL = """
select coalesce(sum(case when source_domain = any($3::text[]) then w end), 0)::float8
         / nullif(sum(w), 0)::float8 as share
from (
  select source_domain, power(2.0, -extract(epoch from ($1 - first_seen_at)) / 3600.0 / 6.0) as w
  from public.story_coverage
  where story_cluster_id = any($2::uuid[])
    and first_seen_at > $1 - interval '24 hours' and first_seen_at <= $1
) t
"""


async def snapshot_trending(
    conn: Any, now: datetime, window_file: Path | None = WINDOW_FILE
) -> dict[str, Any]:
    """Top 10 trending stories three ways (ADR 0013 evidence):

    * ``p2``: the Phase 2 SQL (one cluster per entry, every domain an outlet);
    * ``merged``: near-identical stories merged, representative by quality order;
    * ``networks``: merged, and each syndication network counted as one outlet
      (from ``config/outlet_networks.yaml`` whether or not it's applied).
    """
    from firstread_ingest.config import load_outlet_networks
    from firstread_ingest.window import WindowSnapshot

    p2 = [
        await _describe(conn, t["id"], [t["story_cluster_id"]], t["score"], t["domains"])
        for t in await pools.trending(conn, now, limit=10)
    ]
    out: dict[str, Any] = {"p2": p2}
    if window_file is None or not window_file.is_file():  # noqa: ASYNC240 (local batch tool)
        return out
    win, _ = WindowSnapshot.load(window_file)
    nets = load_outlet_networks()
    all_nets = {d: n for n, ds in nets.networks.items() for d in ds}
    merged, info = await pools.trending_entries(conn, win, now, networks={}, limit=10)
    out["merged"] = [await _describe(conn, e.id, e.clusters, e.score, e.outlets) for e in merged]
    out["merge_samples"] = [
        [r["title"] for r in await conn.fetch(
            "select title from public.articles where id = any($1)", ids
        )]
        for ids in info.samples
    ]  # fmt: skip
    if all_nets:
        with_nets, _ = await pools.trending_entries(conn, win, now, networks=all_nets, limit=10)
        out["networks"] = [
            await _describe(conn, e.id, e.clusters, e.score, e.outlets) for e in with_nets
        ]
        share = await conn.fetchval(
            NETWORK_SHARE_SQL, now, [c for e in merged for c in e.clusters], sorted(all_nets)
        )
        out["network_share_of_top10_score"] = round(share or 0.0, 4)
        mine = [set(e["clusters"]) for e in out["merged"]]
        theirs = [set(e["clusters"]) for e in out["networks"]]
        out["top10_changed_by_networks"] = sum(1 for a in mine if not any(a & b for b in theirs))
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
    window_file: Path | None = WINDOW_FILE,
    compare: bool = False,
    after_run: AfterRun | None = None,
) -> Replay:
    """Run the pipeline once per slot with the clock set to that slot (GDELT only).

    ``after_run(slot, now)`` is awaited after each run: the database and the published
    pools then hold exactly what production would have had at ``now`` (the persona
    replay report ranks feeds there)."""
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
                window_file=window_file,
                compare_legacy=compare,
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
                "window": d["details"]["window"],
                "egress_bytes_est": d["details"]["egress_bytes_est"],
                "pool_blob_bytes": d["details"]["pool_blob_bytes"],
                **({"compare": d["details"]["compare"]} if compare else {}),
            }
        )
        if after_run is not None:
            await after_run(slot, now)
        if (slot.hour, slot.minute) == SNAPSHOT_TIME or i == len(todo) - 1:
            result.trending_days[slot.strftime("%Y-%m-%d")] = await snapshot_trending(
                conn, now, window_file
            )
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
    from firstread_core import pools as core_pools

    ver = await redis.get(core_pools.VER_KEY)
    blob = await redis.get(core_pools.blob_key(ver.decode())) if ver else None
    snapshot = core_pools.decode(blob, ver.decode()) if blob and ver else None
    pool_sizes = dict(core_pools.summarize(snapshot)) if snapshot else {}
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
        "pools": pool_sizes,
        "pool_blob_bytes": len(blob) if blob else 0,
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


def compare_totals(runs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Snapshot path vs the pre-P3 SQL paths over every compared run."""
    compared = [r for r in runs if "compare" in r]
    if not compared:
        return None
    dedupe = [r["compare"].get("dedupe", {}) for r in compared]
    pool = [r["compare"].get("pools", {}) for r in compared]
    return {
        "runs_compared": len(compared),
        "dedupe_inputs_differ_runs": sum(1 for d in dedupe if not d.get("inputs_equal", True)),
        "decisions": sum(d.get("decisions", 0) for d in dedupe),
        "decisions_changed": sum(d.get("decisions_changed", 0) for d in dedupe),
        "fresh_pool_differs_runs": sum(1 for p in pool if not p.get("fresh_equal", True)),
        "topic_pools_differ_runs": sum(1 for p in pool if not p.get("topics_equal", True)),
    }


def egress_totals(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimated Postgres result bytes per run (Supabase egress, ADR 0011)."""
    vals = sorted(r.get("egress_bytes_est", 0) for r in runs)
    reloads = sum(1 for r in runs if (r.get("window") or {}).get("start") == "reloaded")
    if not vals:
        return {}
    return {
        "per_run_median": vals[len(vals) // 2],
        "per_run_max": vals[-1],
        "per_run_mean": round(sum(vals) / len(vals)),
        "window_reloads": reloads,
        "note": "estimated from result sizes; the first run (empty window) and any "
        "reloads carry the full-window cost",
    }


def aggregates(report: dict[str, Any]) -> dict[str, Any]:
    """The committed copy: no headlines or URLs (same rule as the threshold pairs)."""
    agg: dict[str, Any] = json.loads(json.dumps(report, default=str))
    for day in agg.get("trending_days", {}).values():
        stories = (
            day
            if isinstance(day, list)
            else [s for k in ("p2", "merged", "networks") for s in day.get(k, [])]
        )
        for story in stories:
            story.pop("headline", None)
            story.pop("clusters", None)
        if isinstance(day, dict):
            day.pop("merge_samples", None)
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


async def main(
    days: int,
    end: datetime | None,
    db: str,
    redis_db: int,
    fresh: bool,
    compare: bool = False,
    after_run: AfterRun | None = None,
    write_reports: bool = True,
) -> Replay:
    """Replay ``days`` of GKG files. ``write_reports=False`` (the persona replay) skips
    the final state, the live RSS estimate and every report file; the caller keeps
    the returned per-run summaries instead."""
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
            WINDOW_FILE.unlink(missing_ok=True)
        classifier = load_classifier(settings)
        embedder = Embedder()
        places = load_domain_countries(lookup)
        result = await replay(
            settings,
            classifier,
            slots,
            conn,
            redis,
            places,
            lookup,
            embed=embedder.embed,
            compare=compare,
            after_run=after_run,
        )
        if not write_reports:
            return result
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
        "egress": egress_totals(runs),
        **({"equivalence": compare_totals(runs)} if compare_totals(runs) else {}),
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
    if "equivalence" in report:
        evidence = {
            "window": report["window"],
            "equivalence": report["equivalence"],
            "egress": report["egress"],
            "redis_commands_per_run": report["redis_commands_per_run"],
            "pool_blob_bytes": report["final"].get("pool_blob_bytes"),
        }
        EQUIVALENCE.write_text(json.dumps(evidence, indent=1, default=str) + "\n")
    print(f"wrote {REPORT} and {AGGREGATES}")
    return result
