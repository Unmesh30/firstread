"""Demo personas: 20 live articles each, selected once a day from the window (ADR 0018).

A persona is a synthetic reader for recruiters and first-time visitors (SPEC §2). Its
history is made of **live** articles in its topics (``config/personas.yaml``), never
MIND rows. Each run:

1. reads the stored personas (one small query, ~2 KB): their config, refresh time,
   article count and 3 sample headlines, which go into the pool snapshot;
2. refreshes them in the first run after 06:00 UTC, whenever the last refresh is
   over 24 h old (a skipped run can't leave them stale), or when the config changed.

**Selection** is deterministic and runs on the runner-side window (no vectors are
read from Postgres). From the last 48 h, per persona:

* only confidently placed articles: ``category_source = model`` with confidence
  ≥ 0.50 (the release threshold), or an RSS feed's own topic hint;
* titles of at least 6 words (checked in SQL for the best candidates only: ids in,
  ids out);
* one article per story: not the same ``story_cluster_id`` and not ≥ τ_feed to any
  article already picked (complete linkage, ``firstread_core.stories``);
* at most 3 per domain;
* best first: trending score of the article's story, then recency, then id;
* 20 split across the persona's topics, interleaved; a topic short of articles
  leaves its places to the others.

**Writing** replaces ``persona_articles`` with ``INSERT … SELECT`` from ``articles``,
so the vectors are copied inside Postgres (no egress).
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from firstread_core import stories
from firstread_core.personas import PersonaSample, PersonaSummary
from firstread_core.pools import TrendingEntry
from firstread_core.topics import TOPIC_IDS
from firstread_ingest.config import PersonaSpec, load_personas
from firstread_ingest.window import CATEGORY_SOURCES, TEXT_SOURCES, WindowSnapshot

PER_PERSONA = 20
SELECT_WINDOW = timedelta(hours=48)
MIN_CONFIDENCE = 0.50  # the release's low_confidence_threshold
MIN_TITLE_WORDS = 6
MAX_PER_DOMAIN = 3
TITLE_CHECKS_PER_TOPIC = 300  # best candidates per topic whose title length SQL checks
REFRESH_HOUR = 6  # UTC
REFRESH_EVERY = timedelta(hours=24)


# ---- selection (pure) ----


def eligible(win: WindowSnapshot, now: datetime) -> np.ndarray:
    """Window rows of the last 48 h whose topic is confidently known."""
    lo, hi = int((now - SELECT_WINDOW).timestamp() * 1e6), int(now.timestamp() * 1e6)
    recent = (win.first_seen_us > lo) & (win.first_seen_us <= hi)
    model = (win.category_source == CATEGORY_SOURCES.index("model")) & (
        win.confidence >= MIN_CONFIDENCE
    )
    hint = win.category_source == CATEGORY_SOURCES.index("feed_hint")
    out: np.ndarray = np.flatnonzero(recent & (model | hint))
    return out


def trend_by_cluster(entries: Sequence[TrendingEntry]) -> dict[uuid.UUID, float]:
    out: dict[uuid.UUID, float] = {}
    for e in entries:
        for c in e.clusters:
            out.setdefault(c, e.score)
    return out


def ranked(
    win: WindowSnapshot, idx: np.ndarray, topic: str, trend: Mapping[uuid.UUID, float]
) -> list[int]:
    """Rows of ``idx`` in ``topic``, best first: trend score, then newest, then id."""
    t = TOPIC_IDS.index(topic)
    rows = [int(i) for i in idx if win.category[i] == t]
    return sorted(
        rows,
        key=lambda i: (-trend.get(win.clusters[i], 0.0), -int(win.first_seen_us[i]),
                       win.ids[i].bytes),
    )  # fmt: skip


def quotas(topics: Sequence[str], total: int = PER_PERSONA) -> dict[str, int]:
    """``total`` split across topics, the remainder to the first ones (20 → 7/7/6)."""
    base, extra = divmod(total, len(topics))
    return {t: base + (k < extra) for k, t in enumerate(topics)}


def select(
    win: WindowSnapshot,
    spec: PersonaSpec,
    by_topic: Mapping[str, Sequence[int]],
    long_titles: set[uuid.UUID],
    total: int = PER_PERSONA,
) -> list[int]:
    """Window rows for one persona, in display order (topics interleaved)."""
    picked: list[int] = []
    clusters: set[uuid.UUID] = set()
    domains: Counter[str] = Counter()
    cursor = dict.fromkeys(spec.topics, 0)

    def acceptable(i: int) -> bool:
        if win.ids[i] not in long_titles or win.clusters[i] in clusters:
            return False
        if domains[win.source_domain[i]] >= MAX_PER_DOMAIN:
            return False
        if not picked:
            return True
        src = TEXT_SOURCES[win.text_source[i]]
        sims = win.embedding[picked].astype(np.float32) @ win.embedding[i].astype(np.float32)
        return not any(
            s >= stories.tau_feed(src, TEXT_SOURCES[win.text_source[j]])  # type: ignore[arg-type]
            for s, j in zip(sims.tolist(), picked, strict=True)
        )

    def take(topic: str) -> bool:
        rows = by_topic.get(topic, ())
        while cursor[topic] < len(rows):
            i = rows[cursor[topic]]
            cursor[topic] += 1
            if acceptable(i):
                picked.append(i)
                clusters.add(win.clusters[i])
                domains[win.source_domain[i]] += 1
                return True
        return False

    left = quotas(spec.topics, total)
    # Interleave: one per topic per round while each has quota; then anyone fills the rest.
    for respect_quota in (True, False):
        progress = True
        while progress and len(picked) < total:
            progress = False
            for t in spec.topics:
                if len(picked) >= total or (respect_quota and left[t] <= 0):
                    continue
                if take(t):
                    left[t] -= 1
                    progress = True
    return picked


# ---- Postgres ----

_LONG_TITLES = """
select id from public.articles
where id = any($1::uuid[])
  and cardinality(regexp_split_to_array(btrim(title), '\\s+')) >= $2
"""

_LOAD = """
select d.id, d.name, d.description, d.topics, d.refreshed_at,
       (select count(*) from public.persona_articles pa where pa.persona_id = d.id) as n,
       (select coalesce(json_agg(json_build_object(
                 'title', s.title, 'source_name', s.source_name, 'category', s.category)
               order by s.position), '[]'::json)::text
        from (select title, source_name, category, position from public.persona_articles
              where persona_id = d.id order by position limit 3) s) as samples
from public.demo_personas d
"""

_COPY = """
insert into public.persona_articles (persona_id, position, article_id, title, source_name,
  source_domain, url, category, story_cluster_id, text_source, embedding)
select p.persona, p.pos, a.id, a.title, a.source_name, a.source_domain, a.url, a.category,
  a.story_cluster_id, a.text_source, a.embedding
from unnest($1::text[], $2::int[], $3::uuid[]) as p(persona, pos, id)
join public.articles a on a.id = p.id
"""


@dataclass
class Stored:
    spec: PersonaSpec
    refreshed_at: datetime
    articles: int
    samples: list[PersonaSample]


async def load(conn: Any) -> dict[str, Stored]:
    out = {}
    for r in await conn.fetch(_LOAD):
        spec = PersonaSpec(
            id=r["id"], name=r["name"], description=r["description"], topics=tuple(r["topics"])
        )
        samples = [PersonaSample(**s) for s in json.loads(r["samples"])]
        out[r["id"]] = Stored(spec, r["refreshed_at"], int(r["n"]), samples)
    return out


def due(stored: Mapping[str, Stored], specs: Sequence[PersonaSpec], now: datetime) -> str | None:
    """Why a refresh is due now (None: not due)."""
    if [stored[s.id].spec if s.id in stored else None for s in specs] != list(specs):
        return "config"
    if len(stored) != len(specs):
        return "config"
    last = min(s.refreshed_at for s in stored.values())
    if now - last >= REFRESH_EVERY:
        return "stale"
    six = now.astimezone(UTC).replace(hour=REFRESH_HOUR, minute=0, second=0, microsecond=0)
    if now >= six > last:
        return "daily"
    if any(s.articles == 0 for s in stored.values()):
        return "empty"
    return None


async def long_title_ids(conn: Any, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
    if not ids:
        return set()
    return {r["id"] for r in await conn.fetch(_LONG_TITLES, list(ids), MIN_TITLE_WORDS)}


async def refresh(
    conn: Any,
    win: WindowSnapshot,
    now: datetime,
    trending: Sequence[TrendingEntry],
    specs: Sequence[PersonaSpec],
) -> dict[str, list[uuid.UUID]]:
    """Select and store every persona's articles; returns the ids per persona."""
    idx = eligible(win, now)
    trend = trend_by_cluster(trending)
    topics = sorted({t for s in specs for t in s.topics})
    by_topic = {t: ranked(win, idx, t, trend) for t in topics}
    check = sorted({win.ids[i] for t in topics for i in by_topic[t][:TITLE_CHECKS_PER_TOPIC]})
    long_titles = await long_title_ids(conn, check)
    picks = {s.id: [win.ids[i] for i in select(win, s, by_topic, long_titles)] for s in specs}
    async with conn.transaction():
        ids = [s.id for s in specs]
        await conn.execute("delete from public.demo_personas where not (id = any($1::text[]))", ids)
        for s in specs:
            await conn.execute(
                "insert into public.demo_personas (id, name, description, topics, refreshed_at) "
                "values ($1, $2, $3, $4, $5) on conflict (id) do update set "
                "name = excluded.name, description = excluded.description, "
                "topics = excluded.topics, refreshed_at = excluded.refreshed_at",
                s.id, s.name, s.description, list(s.topics), now,
            )  # fmt: skip
        await conn.execute(
            "delete from public.persona_articles where persona_id = any($1::text[])", ids
        )
        rows = [(p, k, a) for p, arts in picks.items() for k, a in enumerate(arts)]
        await conn.execute(_COPY, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    return picks


@dataclass
class PersonaResult:
    summaries: list[PersonaSummary] = field(default_factory=list)
    refreshed: str | None = None  # why it refreshed this run, if it did

    def as_dict(self) -> dict[str, Any]:
        return {
            "refreshed": self.refreshed,
            "articles": {s.id: s.articles for s in self.summaries},
        }


async def refresh_or_load(
    conn: Any,
    win: WindowSnapshot,
    now: datetime,
    trending: Sequence[TrendingEntry],
    specs: Sequence[PersonaSpec] | None = None,
) -> PersonaResult:
    """The persona summaries for this run's snapshot, refreshing them when due."""
    specs = list(load_personas() if specs is None else specs)
    stored = await load(conn)
    why = due(stored, specs, now)
    if why is not None:
        await refresh(conn, win, now, trending, specs)
        stored = await load(conn)
    summaries = []
    for s in specs:
        if s.id in stored:
            st = stored[s.id]
            refreshed = int(st.refreshed_at.timestamp())
            summaries.append(PersonaSummary(
                s.id, s.name, s.description, s.topics, st.articles, refreshed, tuple(st.samples)
            ))  # fmt: skip
    return PersonaResult(summaries, why)
