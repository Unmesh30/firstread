"""End-to-end runs over recorded fixtures: dry run, then real Postgres (throwaway DB)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_ingest.config import Feed, Settings
from firstread_ingest.models import TopicClassifier
from firstread_ingest.pipeline import EmbedFn, run

FIX = Path(__file__).parent / "fixtures" / "rss"
# Fixture items are from 2026-10-14; run "shortly after" so none are stale.
NOW = datetime(2026, 10, 14, 21, 0, tzinfo=UTC)
FEEDS = [
    Feed(
        id=p.stem,
        url=f"https://{p.stem.replace('_', '-')}.example.com/feed.xml",
        source_name="Example",
        domain=f"{p.stem.replace('_', '-')}.example.com",
        region="US",
        geo="us" if i % 2 else "intl",
        topic_hint="world",
        terms="website_with_attribution",
    )
    for i, p in enumerate(sorted(FIX.glob("*.xml")))
]


def mock_feeds(router: respx.MockRouter) -> None:
    for f in FEEDS:
        router.get(f.url).mock(
            return_value=httpx.Response(
                200, content=(FIX / f"{f.id}.xml").read_bytes(), headers={"ETag": f'"{f.id}"'}
            )
        )


def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


async def test_dry_run_writes_nothing_and_classifies_everything(
    no_network: respx.MockRouter, classifier: TopicClassifier, fake_embed: EmbedFn
) -> None:
    mock_feeds(no_network)
    s = await run(settings(), classifier, embed=fake_embed, feeds=FEEDS, dry_run=True, now=NOW)
    assert s.fetched == 25
    assert s.inserted + s.duplicates + s.near_duplicates + sum(s.dropped.values()) == 25
    assert s.inserted > 0
    assert s.status == "ok"
    assert sum(s.topics.values()) == s.inserted
    assert set(s.category_sources) <= {"model", "feed_hint", "geo_rule"}


async def test_stale_items_are_dropped(
    no_network: respx.MockRouter, classifier: TopicClassifier, fake_embed: EmbedFn
) -> None:
    mock_feeds(no_network)
    later = NOW + timedelta(days=10)
    s = await run(settings(), classifier, embed=fake_embed, feeds=FEEDS, dry_run=True, now=later)
    assert s.inserted == 0
    assert s.dropped["stale"] == 25


@pytest.mark.integration
async def test_second_run_over_the_same_items_inserts_nothing(
    no_network: respx.MockRouter, classifier: TopicClassifier, fake_embed: EmbedFn
) -> None:
    await skip_or_fail_if_unreachable_async()
    mock_feeds(no_network)
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            first = await run(
                settings(), classifier, embed=fake_embed, feeds=FEEDS, conn=conn, now=NOW
            )
            assert first.inserted > 0
            second = await run(settings(), classifier, embed=fake_embed, feeds=FEEDS, conn=conn,
                               now=NOW + timedelta(minutes=15))  # fmt: skip
            assert second.inserted == 0
            assert (
                second.duplicates + second.near_duplicates
                == first.inserted + first.duplicates + first.near_duplicates
            )
            n = await conn.fetchval("select count(*) from public.articles")
            assert n == first.inserted
            runs = await conn.fetch(
                "select status, inserted from public.ingestion_runs order by started_at"
            )
            assert [(r["status"], r["inserted"]) for r in runs] == [
                ("ok", first.inserted),
                ("ok", 0),
            ]
            etags = await conn.fetch("select feed_id, etag from public.feed_state")
            assert {r["feed_id"]: r["etag"] for r in etags} == {f.id: f'"{f.id}"' for f in FEEDS}
            cov = await conn.fetchval("select count(*) from public.story_coverage")
            assert cov >= await conn.fetchval(
                "select count(distinct story_cluster_id) from public.articles"
            )
            nulls = await conn.fetchval(
                "select count(*) from public.articles "
                "where summary is not null or embedding is null"
            )
            assert nulls == 0  # descriptions are never stored
        finally:
            await conn.close()
