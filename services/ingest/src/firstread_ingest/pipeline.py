"""One ingestion run: fetch -> normalize -> dedupe -> embed -> classify -> select -> upsert.

Only ``store`` touches Postgres; with ``dry_run`` nothing is written and dedupe runs
against an empty history. A failing source (or feed) is recorded and the run goes
on; the run's status is ``ok``, ``partial`` (something failed) or ``failed``.

Sources: curated RSS (``rss``) and GDELT's GKG files (``gdelt``). GDELT delivers far
more than the database budget allows, so its inserts are ranked and capped per run
(``select.select_gdelt``); RSS inserts are always kept.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_MODEL, Embedder, embedding_cache_key
from firstread_core.topics import TopicDecision, resolve_topic
from firstread_ingest import pools, store
from firstread_ingest.config import Feed, Settings, load_feeds
from firstread_ingest.dedupe import VALIDATED_BY_PAIR, Decision, Item, Recent, Thresholds, decide
from firstread_ingest.http import make_client
from firstread_ingest.items import Candidate, RawItem
from firstread_ingest.models import TopicClassifier
from firstread_ingest.normalize import normalize_all
from firstread_ingest.select import Choice, select_gdelt
from firstread_ingest.sources import gkg, rss
from firstread_ingest.sources.gdelt_domains import (
    DomainCountries,
    ensure_lookup,
    load_domain_countries,
)

EMBEDDING_MODEL_ID = f"{EMBED_MODEL}#{embedding_cache_key()[0][:12]}"
EmbedFn = Callable[[Sequence[str]], NDArray[np.float32]]
SOURCES = ("rss", "gdelt")


@dataclass
class RunSummary:
    started_at: datetime
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    near_duplicates: int = 0
    clustered: int = 0
    errors: int = 0
    gdelt_calls: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    feed_errors: dict[str, str] = field(default_factory=dict)
    topics: Counter[str] = field(default_factory=Counter)
    category_sources: Counter[str] = field(default_factory=Counter)
    geos: Counter[str] = field(default_factory=Counter)
    origins: Counter[str] = field(default_factory=Counter)
    text_sources: Counter[str] = field(default_factory=Counter)
    gdelt: dict[str, Any] = field(default_factory=dict)
    redis_commands: int = 0
    pools: dict[str, int] = field(default_factory=dict)
    retention: dict[str, int] = field(default_factory=dict)
    size: dict[str, int] = field(default_factory=dict)
    size_warning: bool = False
    seconds: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    model_version: str = ""
    window_start: datetime | None = None
    window_end: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched,
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "near_duplicates": self.near_duplicates,
            "clustered": self.clustered,
            "errors": self.errors,
            "gdelt_calls": self.gdelt_calls,
            "redis_commands": self.redis_commands,
            "status": self.status,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "details": {
                "dropped": dict(self.dropped),
                "per_source": self.per_source,
                "feed_errors": self.feed_errors,
                "topics": dict(self.topics),
                "category_sources": dict(self.category_sources),
                "geos": dict(self.geos),
                "origins": dict(self.origins),
                "text_sources": dict(self.text_sources),
                "gdelt": self.gdelt,
                "pools": self.pools,
                "retention": self.retention,
                "size": self.size,
                "size_warning": self.size_warning,
                "seconds": self.seconds,
                "model_version": self.model_version,
            },
        }


def needs_embedding(cands: Sequence[Candidate], recent: Recent) -> list[int]:
    """Indices whose URL/fingerprint isn't already known: only these are embedded.

    Mirrors the exact-duplicate rules in ``dedupe.decide``, so a skipped row is
    always decided as a duplicate before its (zero) vector would be looked at.
    """
    urls, prints = set(recent.url_cluster), set(recent.fingerprint_cluster)
    out = []
    for i, c in enumerate(cands):
        if c.url_canonical in urls or c.title_fingerprint in prints:
            continue
        urls.add(c.url_canonical)
        prints.add(c.title_fingerprint)
        out.append(i)
    return out


def limit_gdelt(
    cands: list[Candidate], recent: Recent, max_embed: int
) -> tuple[list[Candidate], int]:
    """Keep every candidate except the oldest new GDELT ones beyond ``max_embed``.

    Exact duplicates of stored articles cost nothing (they aren't embedded), so they
    stay and still count towards coverage.
    """
    todo = set(needs_embedding(cands, recent))
    fresh_gdelt = [i for i in todo if cands[i].raw.origin == "gdelt"]
    if len(fresh_gdelt) <= max_embed:
        return cands, 0
    fresh_gdelt.sort(key=lambda i: cands[i].raw.seen_at, reverse=True)
    drop = set(fresh_gdelt[max_embed:])
    return [c for i, c in enumerate(cands) if i not in drop], len(drop)


def classify(
    cands: Sequence[Candidate],
    decisions: Sequence[Decision],
    vectors: NDArray[np.float32],
    classifier: TopicClassifier,
) -> dict[int, TopicDecision]:
    """Topic for every candidate that dedupe would insert."""
    idx = [i for i, d in enumerate(decisions) if d.action == "insert"]
    if not idx:
        return {}
    probs = classifier.probabilities(vectors[idx])
    out = {}
    for k, i in enumerate(idx):
        raw = cands[i].raw
        out[i] = resolve_topic(
            probs[k], classifier.labels, raw.geo, classifier.threshold, raw.topic_hint
        )
    return out


def make_row(
    c: Candidate, d: Decision, vector: NDArray[np.float32], topic: TopicDecision
) -> store.ArticleRow:
    if d.article_id is None:
        raise RuntimeError("an inserted decision has no article id")
    raw = c.raw
    return store.ArticleRow(
        id=d.article_id,
        url=raw.url,
        url_canonical=c.url_canonical,
        title=c.title,
        source_name=raw.source_name,
        source_domain=raw.source_domain,
        image_url=raw.image_url,
        language="en",
        source_country=raw.source_country,
        geo=raw.geo,
        category=topic.topic,
        category_confidence=topic.confidence,
        category_source=topic.source,
        published_at=raw.published_at,
        first_seen_at=raw.seen_at,
        embedding=vector,
        embedding_model=EMBEDDING_MODEL_ID,
        text_source=c.text_source,
        title_fingerprint=c.title_fingerprint,
        story_cluster_id=d.cluster_id,
        origin=raw.origin,
    )


async def run(
    settings: Settings,
    classifier: TopicClassifier,
    *,
    embed: EmbedFn | None = None,
    sources: Sequence[str] = SOURCES,
    feeds: Sequence[Feed] | None = None,
    places: DomainCountries | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    client: Any = None,
    conn: Any = None,
    redis: Any = None,
) -> RunSummary:
    now = now or datetime.now(UTC)
    summary = RunSummary(started_at=now, model_version=classifier.version)
    embed = embed or Embedder().embed
    owns_conn = conn is None and not dry_run
    if owns_conn:
        from firstread_core.db import connect

        conn = await connect(settings.database_url.get_secret_value())
    run_id = None
    try:
        if conn is not None:
            await store.register_codecs(conn)
            run_id = await store.start_run(conn, now)
        summary = await _run(
            settings, classifier, embed, sources, feeds, places, conn, now, client, summary
        )
        if conn is not None:
            await _update_pools(settings, conn, redis, now, summary)
    except Exception as exc:
        summary.status = "failed"
        summary.feed_errors["_run"] = f"{type(exc).__name__}: {exc}"[:300]
        summary.errors += 1
        raise
    finally:
        if conn is not None and run_id is not None:
            await store.finish_run(conn, run_id, summary.as_dict(), datetime.now(UTC))
        if owns_conn and conn is not None:
            await conn.close()
    return summary


async def _update_pools(
    settings: Settings, conn: Any, redis: Any, now: datetime, summary: RunSummary
) -> None:
    """Rebuild the Redis pools from Postgres. A Redis failure makes the run partial;
    Postgres stays the source of truth and the next run rebuilds everything."""
    owns = redis is None
    try:
        if owns:
            from redis.asyncio import Redis

            redis = Redis.from_url(settings.redis_url.get_secret_value())
        result = await pools.rebuild(redis, conn, now, summary.topics.keys())
        summary.redis_commands = result.commands
        summary.pools = result.sizes
    except Exception as exc:
        summary.feed_errors["_redis"] = f"{type(exc).__name__}: {exc}"[:300]
        summary.errors += 1
        summary.status = "partial"
    finally:
        if owns and redis is not None:
            await redis.aclose()


async def _fetch(
    settings: Settings,
    sources: Sequence[str],
    feeds: Sequence[Feed] | None,
    places: DomainCountries | None,
    conn: Any,
    now: datetime,
    client: Any,
    summary: RunSummary,
) -> tuple[list[RawItem], dict[str, rss.FeedState], gkg.GkgResult | None]:
    raw: list[RawItem] = []
    states: dict[str, rss.FeedState] = {}
    gdelt: gkg.GkgResult | None = None
    if "rss" in sources:
        old_states = await store.load_feed_states(conn) if conn is not None else {}
        feed_list = list(feeds) if feeds is not None else load_feeds()
        results = await rss.fetch_all(
            client, feed_list, old_states, settings.ingest_rss_concurrency, now
        )
        for r in results:
            states[r.feed.id] = r.state
            summary.per_source[r.feed.id] = {"fetched": len(r.items)}
            if r.error:
                summary.feed_errors[r.feed.id] = r.error
                summary.errors += 1
            raw.extend(r.items)
    if "gdelt" in sources:
        if places is None:
            places = load_domain_countries(await ensure_lookup(client))
        cursor_text = await store.load_cursor(conn, gkg.SOURCE_ID) if conn is not None else None
        cursor = gkg.parse_slot(cursor_text) if cursor_text else None
        gdelt = await gkg.fetch_gkg(
            client,
            cursor,
            now,
            places,
            settings.ingest_gkg_max_files,
            timedelta(minutes=settings.ingest_gkg_stale_minutes),
        )
        summary.gdelt_calls = 1 + len(gdelt.files)
        summary.per_source[gkg.SOURCE_ID] = {"fetched": len(gdelt.items)}
        if gdelt.error or any("error" in f for f in gdelt.files):
            summary.feed_errors[gkg.SOURCE_ID] = gdelt.error or "a GKG file failed"
            summary.errors += 1
        if gdelt.files:
            summary.window_start = gkg.parse_slot(gdelt.files[0]["slot"])
            summary.window_end = gkg.parse_slot(gdelt.files[-1]["slot"]) + gkg.SLOT
        raw.extend(gdelt.items)
    return raw, states, gdelt


async def _run(
    settings: Settings,
    classifier: TopicClassifier,
    embed: EmbedFn,
    sources: Sequence[str],
    feeds: Sequence[Feed] | None,
    places: DomainCountries | None,
    conn: Any,
    now: datetime,
    client: Any,
    summary: RunSummary,
) -> RunSummary:
    t0 = time.perf_counter()
    owns_client = client is None
    client = client or make_client()
    try:
        raw, states, gdelt = await _fetch(
            settings, sources, feeds, places, conn, now, client, summary
        )
    finally:
        if owns_client:
            await client.aclose()
    summary.fetched = len(raw)
    summary.seconds["fetch"] = round(time.perf_counter() - t0, 2)

    cands, dropped = normalize_all(raw, now, settings.ingest_min_title_words)
    recent = (
        await store.load_recent(
            conn,
            now,
            timedelta(hours=settings.ingest_dedupe_window_hours),
            timedelta(hours=settings.ingest_fingerprint_window_hours),
        )
        if conn is not None
        else Recent.empty()
    )
    cands, over_cap = limit_gdelt(cands, recent, settings.ingest_gkg_max_embed)
    if over_cap:
        dropped["gdelt_embed_cap"] += over_cap
    summary.dropped = dropped

    t1 = time.perf_counter()
    todo = needs_embedding(cands, recent)
    vectors = np.zeros((len(cands), recent.embeddings.shape[1]), np.float32)
    if todo:
        vectors[todo] = embed([cands[i].embed_text for i in todo])
    summary.seconds["embed"] = round(time.perf_counter() - t1, 2)

    items = [
        Item(c.url_canonical, c.title_fingerprint, c.text_source, c.raw.seen_at) for c in cands
    ]
    thresholds = Thresholds(
        settings.ingest_tau_dup,
        settings.ingest_tau_story,
        VALIDATED_BY_PAIR if settings.ingest_per_pair_thresholds else {},
    )
    decisions = decide(items, vectors, recent, thresholds)
    topics = classify(cands, decisions, vectors, classifier)

    gdelt_choices = [
        Choice(i, t.topic) for i, t in topics.items() if cands[i].raw.origin == "gdelt"
    ]
    kept_gdelt, capped = select_gdelt(
        cands,
        decisions,
        gdelt_choices,
        settings.ingest_gkg_max_per_run,
        settings.ingest_gkg_max_per_domain,
        settings.ingest_gkg_topic_min,
        settings.ingest_gkg_max_per_story,
        Counter(recent.clusters),
    )
    keep = [i for i in topics if cands[i].raw.origin != "gdelt" or i in kept_gdelt]
    rows = [make_row(cands[i], decisions[i], vectors[i], topics[i]) for i in sorted(keep)]

    # Coverage: every item whose story ends up stored (already, or by this run),
    # inserted or not, so trending sees each outlet that carried it.
    stored_clusters = set(recent.clusters) | set(recent.url_cluster.values())
    stored_clusters |= set(recent.fingerprint_cluster.values())
    stored_clusters |= {r.story_cluster_id for r in rows}
    covered = [
        (c, d) for c, d in zip(cands, decisions, strict=True) if d.cluster_id in stored_clusters
    ]
    coverage = [(d.cluster_id, c.raw.source_domain, c.raw.seen_at) for c, d in covered]

    inserted: set[str] = set()
    if conn is not None:
        async with conn.transaction():
            inserted = await store.insert_articles(conn, rows)
            await store.insert_coverage(conn, coverage)
            await store.save_feed_states(conn, states, now)
            if gdelt is not None and gdelt.cursor is not None:
                await store.save_cursor(conn, gkg.SOURCE_ID, gkg.slot_name(gdelt.cursor), now)
        async with conn.transaction():
            summary.retention = await store.apply_retention(
                conn,
                now,
                articles=timedelta(days=settings.ingest_retention_article_days),
                coverage=timedelta(hours=settings.ingest_retention_coverage_hours),
                runs=timedelta(days=settings.ingest_retention_runs_days),
                # Forget state only for feeds that left the configuration.
                feed_ids=[f.id for f in (feeds or load_feeds(enabled_only=False))]
                if "rss" in sources
                else None,
            )
        summary.size = await store.database_size(conn)
        if summary.size["db"] > settings.ingest_size_warn_mb * 1_000_000:
            summary.size_warning = True
    else:
        inserted = {r.url_canonical for r in rows}

    for d in decisions:
        if d.action == "duplicate":
            summary.duplicates += 1
        elif d.action == "near_duplicate":
            summary.near_duplicates += 1
    for row in rows:
        if row.url_canonical not in inserted:  # lost a race with a concurrent writer
            summary.duplicates += 1
            continue
        summary.inserted += 1
        summary.topics[row.category] += 1
        summary.category_sources[row.category_source] += 1
        summary.geos[row.geo] += 1
        summary.origins[row.origin] += 1
        summary.text_sources[row.text_source] += 1
        if row.story_cluster_id != row.id:
            summary.clustered += 1
    if gdelt is not None:
        summary.gdelt = gdelt.summary() | {
            "candidates": len(gdelt_choices),
            "selected": len(kept_gdelt),
            "capped": capped,
            "coverage_rows": sum(1 for c, _ in covered if c.raw.origin == "gdelt"),
        }
        if gdelt.stale:
            summary.feed_errors.setdefault(
                gkg.SOURCE_ID, "stale: newest GKG file is over an hour old"
            )
    if summary.errors:
        summary.status = "partial"
    summary.seconds["total"] = round(time.perf_counter() - t0, 2)
    return summary
