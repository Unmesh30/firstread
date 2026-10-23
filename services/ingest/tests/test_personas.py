"""Demo persona selection (pure) and refresh (real Postgres) (ADR 0018)."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_core.pools import TrendingEntry
from firstread_core.topics import TOPIC_IDS
from firstread_ingest import personas, store, window
from firstread_ingest.config import PersonaSpec, load_personas
from firstread_ingest.window import CATEGORY_SOURCES, WindowSnapshot

NOW = datetime(2026, 10, 15, 12, 20, tzinfo=UTC)
SPEC = PersonaSpec(id="p", name="P", description="d", topics=("sports", "health"))


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=384)
    return (v / np.linalg.norm(v)).astype(np.float32)


def win_of(rows: list[dict[str, Any]]) -> WindowSnapshot:
    """An in-memory window: one dict per row (topic, age_h, and optional overrides)."""
    snap = WindowSnapshot.empty("sha", "m")
    for k, r in enumerate(rows):
        aid = r.get("id", uuid.UUID(int=k + 1))
        add = WindowSnapshot(
            ids=[aid], clusters=[r.get("cluster", aid)],
            text_source=np.array([0], np.int8),
            embedding=np.asarray([r.get("vec", unit(k))], np.float16),
            url_canonical=[f"u{k}"], fingerprint=[f"f{k}"],
            first_seen_us=np.array([int((NOW - timedelta(hours=r["age_h"])).timestamp() * 1e6)]),
            created_us=np.array([int(NOW.timestamp() * 1e6)]),
            category=np.array([TOPIC_IDS.index(r["topic"])], np.int8),
            category_source=np.array([CATEGORY_SOURCES.index(r.get("source", "model"))], np.int8),
            confidence=np.array([r.get("conf", 0.9)], np.float32),
            origin=np.array([0], np.int8), source_domain=[r.get("domain", f"d{k}.com")],
            has_image=np.array([False]), schema_sha="sha", embedding_model="m",
        )  # fmt: skip
        snap = snap._concat(add)
    return snap


def pick(win: WindowSnapshot, spec: PersonaSpec = SPEC, total: int = 20, **kw: Any) -> list[int]:
    trend = kw.get("trend", {})
    idx = personas.eligible(win, NOW)
    by_topic = {t: personas.ranked(win, idx, t, trend) for t in spec.topics}
    long_titles = kw.get("long_titles", set(win.ids))
    return personas.select(win, spec, by_topic, long_titles, total)


def test_quotas_split_twenty_across_topics() -> None:
    assert personas.quotas(["a", "b"]) == {"a": 10, "b": 10}
    assert personas.quotas(["a", "b", "c"]) == {"a": 7, "b": 7, "c": 6}


def test_only_recent_confident_rows_are_eligible() -> None:
    win = win_of([
        {"topic": "sports", "age_h": 1},
        {"topic": "sports", "age_h": 49},  # outside 48 h
        {"topic": "sports", "age_h": 1, "conf": 0.3},  # not confident
        {"topic": "sports", "age_h": 1, "source": "feed_hint", "conf": 0.1},  # a feed's hint
        {"topic": "sports", "age_h": 1, "source": "geo_rule"},
    ])  # fmt: skip
    assert personas.eligible(win, NOW).tolist() == [0, 3]


def test_order_is_trend_then_recency_then_id_and_topics_interleave() -> None:
    rows = [{"topic": "sports", "age_h": 1 + k} for k in range(3)]
    rows += [{"topic": "health", "age_h": 1 + k} for k in range(3)]
    win = win_of(rows)
    trend = {win.clusters[2]: 5.0}  # the oldest sports story is trending: it leads
    got = pick(win, total=6, trend=trend)
    assert got == [2, 3, 0, 4, 1, 5]


def test_one_per_story_three_per_domain_and_long_titles_only() -> None:
    same = uuid.uuid4()
    v = unit(99)
    rows = [
        {"topic": "sports", "age_h": 1, "cluster": same},
        {"topic": "sports", "age_h": 2, "cluster": same},  # same cluster id
        {"topic": "sports", "age_h": 3, "vec": v},
        {"topic": "sports", "age_h": 4, "vec": v},  # near-identical (≥ τ_feed) to row 2
        *({"topic": "sports", "age_h": 5 + k, "domain": "big.com"} for k in range(4)),
        {"topic": "sports", "age_h": 20},  # short title
    ]
    win = win_of(rows)
    short = {win.ids[8]}
    got = pick(win, long_titles=set(win.ids) - short)
    assert got == [0, 2, 4, 5, 6]  # one per story, ≤ 3 from big.com, no short title


def test_a_short_topic_leaves_its_places_to_the_others() -> None:
    rows = [{"topic": "sports", "age_h": 1 + k} for k in range(30)]
    rows += [{"topic": "health", "age_h": 1 + k} for k in range(4)]
    got = pick(win_of(rows))
    assert len(got) == 20
    assert sum(1 for i in got if i >= 30) == 4


def test_selection_is_deterministic_whatever_the_row_order() -> None:
    rows = [{"topic": t, "age_h": 1 + (k % 7), "id": uuid.UUID(int=1000 + k)}
            for k, t in enumerate(["sports", "health"] * 25)]  # fmt: skip
    a = win_of(rows)
    b = win_of(list(reversed(rows)))
    assert [a.ids[i] for i in pick(a)] == [b.ids[i] for i in pick(b)]


def stored(spec: PersonaSpec, at: datetime, n: int = 20) -> personas.Stored:
    return personas.Stored(spec, at, n, [])


def test_refresh_is_due_daily_after_six_when_stale_or_when_the_config_changes() -> None:
    specs = [SPEC]
    morning = datetime(2026, 10, 15, 5, 50, tzinfo=UTC)
    assert personas.due({}, specs, NOW) == "config"
    changed = PersonaSpec(id="p", name="P", description="d", topics=("sports",))
    assert personas.due({"p": stored(changed, NOW)}, specs, NOW) == "config"
    early = morning + timedelta(minutes=5)
    assert personas.due({"p": stored(SPEC, morning)}, specs, early) is None
    assert personas.due({"p": stored(SPEC, morning)}, specs, NOW) == "daily"  # first run after 6
    after = datetime(2026, 10, 15, 6, 5, tzinfo=UTC)
    assert personas.due({"p": stored(SPEC, after)}, specs, NOW) is None
    assert personas.due({"p": stored(SPEC, NOW - timedelta(hours=25))}, specs, NOW) == "stale"
    assert personas.due({"p": stored(SPEC, after, n=0)}, specs, NOW) == "empty"


def test_the_shipped_personas_are_valid() -> None:
    specs = load_personas()
    assert len(specs) == 5
    assert all(1 < len(s.topics) <= 3 for s in specs)


def test_bad_persona_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="slug"):
        PersonaSpec(id="Bad Id", name="x", description="x", topics=("sports",))
    with pytest.raises(ValueError, match="topics"):
        PersonaSpec(id="x", name="x", description="x", topics=("nope",))


# ---- refresh against Postgres ----

COPIED = """
select pa.position, pa.article_id, pa.embedding::text = a.embedding::text as same_vec
from public.persona_articles pa join public.articles a on a.id = pa.article_id
order by pa.position
"""


def article(n: int, topic: str, words: int = 8) -> store.ArticleRow:
    aid = uuid.uuid4()
    return store.ArticleRow(
        id=aid, url=f"https://s{n}.com/{n}", url_canonical=f"https://s{n}.com/{n}",
        title=" ".join(["word"] * (words - 1) + [str(n)]), source_name=f"S{n}",
        source_domain=f"s{n}.com", image_url=None, language="en", source_country="US",
        geo="us", category=topic, category_confidence=0.9, category_source="model",
        published_at=None, first_seen_at=NOW - timedelta(hours=1 + n), embedding=unit(n),
        embedding_model="m", text_source="title", title_fingerprint=f"fp{n}",
        story_cluster_id=aid, origin="gdelt",
    )  # fmt: skip


@pytest.mark.integration
async def test_refresh_copies_articles_server_side_and_loads_summaries() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            await store.register_codecs(conn)
            rows = [article(k, "sports") for k in range(12)]
            rows += [article(20 + k, "health") for k in range(12)]
            rows.append(article(40, "health", words=4))  # too short for a persona
            await store.insert_articles(conn, rows)
            win = (await window.sync(conn, None, NOW, "m")).snapshot
            trending = [TrendingEntry(rows[5].id, 9.0, 12, (rows[5].story_cluster_id,))]
            first = await personas.refresh_or_load(conn, win, NOW, trending, [SPEC])
            assert first.refreshed == "config"
            (s,) = first.summaries
            assert (s.id, s.topics, s.articles) == ("p", ("sports", "health"), 20)
            assert s.samples[0].title == rows[5].title  # the trending story leads
            copied = await conn.fetch(COPIED)
            assert [r["position"] for r in copied] == list(range(20))
            assert all(r["same_vec"] for r in copied)
            assert rows[-1].id not in {r["article_id"] for r in copied}
            again = await personas.refresh_or_load(
                conn, win, NOW + timedelta(minutes=15), trending, [SPEC]
            )
            assert again.refreshed is None
            assert again.summaries == first.summaries
            # A persona that left the config is removed with its articles.
            other = PersonaSpec(id="q", name="Q", description="d", topics=("health",))
            third = await personas.refresh_or_load(conn, win, NOW, trending, [other])
            assert [x.id for x in third.summaries] == ["q"]
            assert (
                await conn.fetchval(
                    "select count(*) from public.persona_articles where persona_id = 'p'"
                )
                == 0
            )
        finally:
            await conn.close()
