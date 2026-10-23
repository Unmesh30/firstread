"""End-to-end runs over recorded fixtures: dry run, then real Postgres (throwaway DB)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from firstread_core import pools as core_pools
from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_ingest.config import Feed, Settings, load_personas
from firstread_ingest.models import TopicClassifier
from firstread_ingest.pipeline import EmbedFn, run
from firstread_ingest.sources import gkg
from firstread_ingest.sources.gdelt_domains import DomainCountries

FIX = Path(__file__).parent / "fixtures" / "rss"
RSS = ("rss",)
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
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    # The synthetic fixtures' titles are near-identical by construction (one "story");
    # the per-story cap has its own tests in test_select.py.
    s.ingest_gkg_max_per_story = 100
    return s


async def test_dry_run_writes_nothing_and_classifies_everything(
    no_network: respx.MockRouter, classifier: TopicClassifier, fake_embed: EmbedFn
) -> None:
    mock_feeds(no_network)
    s = await run(
        settings(), classifier, embed=fake_embed, feeds=FEEDS, sources=RSS, dry_run=True, now=NOW
    )
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
    s = await run(
        settings(), classifier, embed=fake_embed, feeds=FEEDS, sources=RSS, dry_run=True, now=later
    )
    assert s.inserted == 0
    assert s.dropped["stale"] == 25


@pytest.mark.integration
async def test_second_run_over_the_same_items_inserts_nothing(
    no_network: respx.MockRouter,
    classifier: TopicClassifier,
    fake_embed: EmbedFn,
    redis_test: Any,
) -> None:
    await skip_or_fail_if_unreachable_async()
    mock_feeds(no_network)
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            first = await run(
                settings(), classifier, embed=fake_embed, feeds=FEEDS, sources=RSS, conn=conn,
                redis=redis_test,
                now=NOW,
            )  # fmt: skip
            assert first.inserted > 0
            later = NOW + timedelta(minutes=15)
            second = await run(settings(), classifier, embed=fake_embed, feeds=FEEDS, sources=RSS,
                               conn=conn, redis=redis_test, now=later)  # fmt: skip
            assert second.inserted == 0
            # Personas: refreshed by the first run, then only read; both blobs carry them.
            assert first.personas["refreshed"] == "config"
            assert second.personas["refreshed"] in (None, "empty")
            ver = (await redis_test.get(core_pools.VER_KEY)).decode()
            blob = core_pools.decode(await redis_test.get(core_pools.blob_key(ver)), ver)
            assert [p["id"] for p in blob.personas] == [p.id for p in load_personas()]
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


GKG_FIX = next((Path(__file__).parent / "fixtures" / "gkg").glob("*.gkg.csv.zip"))
GKG_SLOT = gkg.parse_slot(GKG_FIX.name.split(".")[0])


def mock_gkg(router: respx.MockRouter) -> None:
    import hashlib

    blob = GKG_FIX.read_bytes()
    md5 = hashlib.md5(blob).hexdigest()  # noqa: S324 (GDELT publishes MD5)
    name = gkg.slot_name(GKG_SLOT)
    router.get(gkg.LASTUPDATE).mock(
        return_value=httpx.Response(200, text=f"{len(blob)} {md5} http://x/{name}.gkg.csv.zip\n")
    )
    router.get(gkg.slot_url(GKG_SLOT)).mock(return_value=httpx.Response(200, content=blob))
    router.get(url__regex=r".*\.gkg\.csv\.zip$").mock(return_value=httpx.Response(404))


@pytest.mark.integration
async def test_rss_and_gdelt_together_with_cursor_and_coverage(
    no_network: respx.MockRouter,
    classifier: TopicClassifier,
    fake_embed: EmbedFn,
    places: DomainCountries,
    redis_test: Any,
) -> None:
    await skip_or_fail_if_unreachable_async()
    mock_feeds(no_network)
    mock_gkg(no_network)
    now = GKG_SLOT + timedelta(minutes=20)  # also keeps the RSS fixture items fresh
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            first = await run(settings(), classifier, embed=fake_embed, feeds=FEEDS,
                              places=places, conn=conn, redis=redis_test, now=now)  # fmt: skip
            assert first.origins["gdelt"] == 8  # 11 records - 2 blocked - 1 unplaced
            assert first.origins["rss"] > 0
            assert first.gdelt["dropped"] == {"blocked_domain": 2, "unplaced_domain": 1}
            assert first.gdelt["selected"] == 8
            assert first.redis_commands == 2  # blob + version; nothing to expire yet
            assert first.pools["fresh"] == first.inserted
            ver = (await redis_test.get("pools:ver")).decode()
            blob = await redis_test.get(f"pools:blob:{ver}")
            assert len(core_pools.decode(blob, ver).fresh) == first.inserted
            assert first.window["start"] == "reloaded"  # no snapshot file in this test
            assert first.egress_bytes_est > 0
            assert not first.gdelt["stale"]
            cursor = await conn.fetchval(
                "select cursor from public.source_cursor where source = 'gdelt-gkg'"
            )
            # Older mocked slots 404; those inside the 2 h catch-up horizon are pending,
            # so the cursor holds before them and the next run retries them.
            assert cursor == gkg.slot_name(GKG_SLOT - timedelta(minutes=105))
            domains = await conn.fetchval(
                "select count(distinct source_domain) from public.story_coverage "
                "where source_domain like '%.co.uk' or source_domain = 'wisn.com'"
            )
            assert domains == 3
            geo = dict(await conn.fetch(
                "select source_domain, geo from public.articles where origin = 'gdelt'"
            ))  # fmt: skip
            assert geo["wisn.com"] == "us"
            assert geo["jpost.com"] == "intl"
            later = now + timedelta(minutes=5)
            second = await run(settings(), classifier, embed=fake_embed, feeds=FEEDS,
                               places=places, conn=conn, redis=redis_test, now=later)  # fmt: skip
            assert second.inserted == 0  # the retried slots only hold duplicates now
            assert second.redis_commands == 3  # blob + version + expire the previous blob
            assert second.gdelt["files"][0]["slot"] == gkg.slot_name(
                GKG_SLOT - timedelta(minutes=90)
            )
        finally:
            await conn.close()


async def test_gdelt_cap_keeps_the_run_within_budget(
    no_network: respx.MockRouter,
    classifier: TopicClassifier,
    fake_embed: EmbedFn,
    places: DomainCountries,
) -> None:
    mock_gkg(no_network)
    s = settings()
    s.ingest_gkg_max_per_run = 3
    s.ingest_gkg_topic_min = 0
    out = await run(s, classifier, embed=fake_embed, sources=("gdelt",), places=places,
                    dry_run=True, now=GKG_SLOT + timedelta(minutes=20))  # fmt: skip
    assert out.inserted == 3
    assert out.gdelt["capped"] == {"run_cap": 5}
