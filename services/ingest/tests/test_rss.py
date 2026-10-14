import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from firstread_core.topics import Geo
from firstread_ingest.config import Feed
from firstread_ingest.http import make_client
from firstread_ingest.sources import rss

FIX = Path(__file__).parent / "fixtures" / "rss"
NOW = datetime(2026, 10, 14, 20, 0, tzinfo=UTC)


def feed(fid: str = "f1", host: str = "feeds.example.com", geo: Geo = "us") -> Feed:
    return Feed(id=fid, url=f"https://{host}/{fid}.xml", source_name="Example News",
                domain="example.com", region="US", geo=geo, topic_hint="world")  # fmt: skip


@pytest.mark.parametrize("name", sorted(p.name for p in FIX.glob("*.xml")))
def test_every_recorded_shape_parses(name: str) -> None:
    items = rss.parse_feed(feed(), (FIX / name).read_bytes(), NOW)
    assert len(items) == 5
    for it in items:
        assert it.url.startswith("https://example.com")
        assert it.title.startswith("Synthetic")
        assert it.published_at is not None
        assert it.published_at.tzinfo is not None
        assert it.description  # used for the embedding only
        assert it.language.startswith("en")
        assert (it.origin, it.geo, it.topic_hint, it.seen_at) == ("rss", "us", "world", NOW)


@pytest.mark.parametrize(
    "name", ["rss2_media_thumbnail.xml", "rss2_media_content.xml", "rss2_enclosure_cdata.xml"]
)
def test_thumbnails_come_from_media_tags_or_enclosures(name: str) -> None:
    items = rss.parse_feed(feed(), (FIX / name).read_bytes(), NOW)
    assert all(it.image_url for it in items)


async def test_conditional_get_sends_validators_and_304_yields_nothing(
    no_network: respx.MockRouter,
) -> None:
    route = no_network.get("https://feeds.example.com/f1.xml").mock(
        return_value=httpx.Response(304)
    )
    state = rss.FeedState(etag='W/"abc"', last_modified="Wed, 14 Oct 2026 16:47:52 GMT")
    async with make_client() as c:
        r = await rss.fetch_feed(c, feed(), state, NOW)
    sent = route.calls.last.request.headers
    assert sent["if-none-match"] == 'W/"abc"'
    assert sent["if-modified-since"] == "Wed, 14 Oct 2026 16:47:52 GMT"
    assert r.not_modified
    assert r.items == []
    assert r.error is None
    assert (r.state.etag, r.state.last_status) == ('W/"abc"', 304)


async def test_200_stores_new_validators(no_network: respx.MockRouter) -> None:
    no_network.get("https://feeds.example.com/f1.xml").mock(
        return_value=httpx.Response(
            200, content=(FIX / "atom.xml").read_bytes(), headers={"ETag": '"v2"'}
        )
    )
    async with make_client() as c:
        r = await rss.fetch_feed(c, feed(), rss.FeedState(etag='"v1"', consecutive_failures=2), NOW)
    assert len(r.items) == 5
    assert (r.state.etag, r.state.last_modified, r.state.consecutive_failures) == ('"v2"', None, 0)


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(200, content=b"<html><body>not a feed"), "unparseable"),
        (httpx.Response(404), "HTTP 404"),
    ],
)
async def test_bad_feeds_record_an_error(
    no_network: respx.MockRouter, response: httpx.Response, error: str
) -> None:
    no_network.get("https://feeds.example.com/f1.xml").mock(return_value=response)
    async with make_client() as c:
        r = await rss.fetch_feed(c, feed(), rss.FeedState(consecutive_failures=1), NOW)
    assert r.error is not None
    assert error in r.error
    assert r.items == []
    assert r.state.consecutive_failures == 2


async def test_one_failing_feed_never_fails_the_others(
    no_network: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    no_network.get("https://a.example.com/ok.xml").mock(
        return_value=httpx.Response(200, content=(FIX / "rdf.xml").read_bytes())
    )
    no_network.get("https://b.example.com/down.xml").mock(side_effect=httpx.ConnectError("down"))
    feeds = [feed("ok", "a.example.com"), feed("down", "b.example.com")]
    async with make_client() as c:
        results = await rss.fetch_all(c, feeds, {}, concurrency=2)
    by_id = {r.feed.id: r for r in results}
    assert len(by_id["ok"].items) == 5
    assert by_id["ok"].error is None
    assert by_id["down"].error
    assert "ConnectError" in by_id["down"].error


async def test_same_host_feeds_are_fetched_one_at_a_time(no_network: respx.MockRouter) -> None:
    active, peak = 0, 0

    async def slow(_: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(304)

    no_network.get(url__regex=r"https://same\.example\.com/.*").mock(side_effect=slow)
    feeds = [feed(f"f{i}", "same.example.com") for i in range(4)]
    async with make_client() as c:
        results = await rss.fetch_all(c, feeds, {}, concurrency=8)
    assert len(results) == 4
    assert peak == 1
