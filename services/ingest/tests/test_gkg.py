import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from firstread_ingest.http import make_client
from firstread_ingest.sources import gkg
from firstread_ingest.sources.gdelt_domains import DomainCountries, parse_lookup

FIX = next((Path(__file__).parent / "fixtures" / "gkg").glob("*.gkg.csv.zip"))
LATEST = gkg.parse_slot(FIX.name.split(".")[0])  # 2026-10-14 18:45


def lastupdate(slot: datetime, blob: bytes) -> str:
    name = gkg.slot_name(slot)
    md5 = hashlib.md5(blob).hexdigest()  # noqa: S324 (GDELT publishes MD5)
    base = "http://data.gdeltproject.org/gdeltv2"
    return (
        f"1 0 {base}/{name}.export.CSV.zip\n2 0 {base}/{name}.mentions.CSV.zip\n"
        f"{len(blob)} {md5} {base}/{name}.gkg.csv.zip\n"
    )


def mock_gdelt(router: respx.MockRouter, blob: bytes, md5_blob: bytes | None = None) -> respx.Route:
    route = router.get(gkg.LASTUPDATE).mock(
        return_value=httpx.Response(200, text=lastupdate(LATEST, md5_blob or blob))
    )
    router.get(gkg.slot_url(LATEST)).mock(return_value=httpx.Response(200, content=blob))
    router.get(url__regex=r".*\.gkg\.csv\.zip$").mock(return_value=httpx.Response(404))
    return route


def test_slot_math() -> None:
    t = datetime(2026, 10, 14, 18, 52, 31, tzinfo=UTC)
    assert gkg.slot_floor(t) == datetime(2026, 10, 14, 18, 45, tzinfo=UTC)
    assert gkg.slot_name(gkg.slot_floor(t)) == "20261014184500"
    assert gkg.slot_url(LATEST).startswith("https://data.gdeltproject.org/gdeltv2/")
    latest = datetime(2026, 10, 14, 18, 45, tzinfo=UTC)
    first_run = gkg.plan_slots(None, latest, 8)
    assert len(first_run) == 8
    assert first_run[0] == latest - timedelta(minutes=105)
    assert gkg.plan_slots(latest - timedelta(minutes=30), latest, 8) == [
        latest - timedelta(minutes=15),
        latest,
    ]
    assert gkg.plan_slots(latest, latest, 8) == []
    long_gap = gkg.plan_slots(latest - timedelta(days=1), latest, 8)
    assert len(long_gap) == 8  # catch-up is bounded to 2 h


def test_lastupdate_lists_the_gkg_file() -> None:
    f = gkg.parse_lastupdate(lastupdate(LATEST, b"abc"))
    assert f.slot == LATEST
    assert f.md5 == hashlib.md5(b"abc").hexdigest()  # noqa: S324
    with pytest.raises(ValueError, match=r"no \.gkg"):
        gkg.parse_lastupdate("1 x http://a/b.export.CSV.zip\n")


def test_parse_gkg_unescapes_titles() -> None:
    records = gkg.parse_gkg(FIX.read_bytes())
    assert len(records) == 11
    assert all("café prices & markets" in r.title for r in records)
    assert all(r.seen_at == LATEST for r in records)
    assert any(r.published_at for r in records)


def test_domain_placement(places: DomainCountries) -> None:
    assert places.fips("wisn.com") == "US"
    assert places.fips("edition.wisn.com") == "US"  # parent-domain fallback
    assert places.fips("jpost.com") == "IS"  # top country wins
    assert places.fips("aljazeera.com") == "QA"  # override beats the lookup
    assert places.fips("unknown.example") is None
    assert places.blocked("iheart.com")
    assert places.blocked("www.iheart.com")
    assert not places.blocked("wisn.com")


async def test_fetch_filters_places_and_requests_fresh_lastupdate(
    no_network: respx.MockRouter, places: DomainCountries
) -> None:
    route = mock_gdelt(no_network, FIX.read_bytes())
    now = LATEST + timedelta(minutes=20)
    async with make_client() as c:
        r = await gkg.fetch_gkg(c, None, now, places)
    assert route.calls.last.request.headers["cache-control"] == "no-cache"
    assert r.records == 11
    assert r.dropped == {"blocked_domain": 2, "unplaced_domain": 1}
    by_domain = {i.source_domain: i for i in r.items}
    assert len(by_domain) == 8
    assert by_domain["wisn.com"].geo == "us"
    assert (by_domain["jpost.com"].geo, by_domain["jpost.com"].source_country) == ("intl", "IL")
    assert by_domain["gazetteseries.co.uk"].source_country == "GB"
    assert all(i.origin == "gdelt" and i.description is None for i in r.items)
    assert not r.stale
    assert len(r.files) == 8  # first run: 7 older slots (404 here) + the newest
    # 404s older than 90 min count as skipped; younger ones are pending, so the cursor
    # holds at 17:30 and the next run retries 17:45 onwards.
    assert r.cursor == LATEST - timedelta(minutes=75)
    assert r.summary()["top_domains"]["iheart.com"] == 1


async def test_catch_up_from_cursor_and_staleness(
    no_network: respx.MockRouter, places: DomainCountries
) -> None:
    mock_gdelt(no_network, FIX.read_bytes())
    async with make_client() as c:
        r = await gkg.fetch_gkg(
            c, LATEST - timedelta(minutes=30), LATEST + timedelta(hours=3), places
        )
    assert [f["slot"] for f in r.files] == [
        gkg.slot_name(LATEST - timedelta(minutes=15)),
        gkg.slot_name(LATEST),
    ]
    assert r.stale  # the newest file is 3 h old


async def test_md5_mismatch_stops_and_keeps_the_cursor(
    no_network: respx.MockRouter, places: DomainCountries
) -> None:
    mock_gdelt(no_network, FIX.read_bytes(), md5_blob=b"other bytes")
    cursor = LATEST - timedelta(minutes=15)
    async with make_client() as c:
        r = await gkg.fetch_gkg(c, cursor, LATEST + timedelta(minutes=5), places)
    assert "MD5" in r.files[-1]["error"]
    assert r.cursor == cursor  # retried next run
    assert r.items == []


async def test_lastupdate_failure_is_an_error_not_a_crash(
    no_network: respx.MockRouter, places: DomainCountries, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    no_network.get(gkg.LASTUPDATE).mock(return_value=httpx.Response(503))
    async with make_client() as c:
        r = await gkg.fetch_gkg(c, None, LATEST, places)
    assert r.error is not None
    assert r.stale
    assert r.items == []


def test_parse_lookup_keeps_the_top_country() -> None:
    assert parse_lookup("domain,countrycode,cnt\na.com,US,5\na.com,UK,9\nb.com,FR,x\n") == {
        "a.com": "UK"
    }


async def test_recent_404s_are_pending_and_hold_the_cursor(
    no_network: respx.MockRouter, places: DomainCountries
) -> None:
    blob = FIX.read_bytes()
    newest = LATEST + timedelta(minutes=30)  # listed, but its file (and 19:00's) 404s
    no_network.get(gkg.LASTUPDATE).mock(
        return_value=httpx.Response(200, text=lastupdate(newest, blob))
    )
    no_network.get(gkg.slot_url(LATEST)).mock(return_value=httpx.Response(200, content=blob))
    no_network.get(url__regex=r".*\.gkg\.csv\.zip$").mock(return_value=httpx.Response(404))
    cursor = LATEST - timedelta(minutes=15)
    async with make_client() as c:
        r = await gkg.fetch_gkg(c, cursor, newest + timedelta(minutes=2), places)
    assert r.items  # 18:45 was processed ...
    assert r.cursor == LATEST  # ... but the cursor stops before the pending 19:00 slot
    assert [f.get("pending", False) for f in r.files] == [False, True, True]


async def test_old_404s_are_skipped_slots(
    no_network: respx.MockRouter, places: DomainCountries
) -> None:
    mock_gdelt(no_network, FIX.read_bytes())
    cursor = LATEST - timedelta(minutes=30)
    async with make_client() as c:  # three hours later, 18:30 is long overdue
        r = await gkg.fetch_gkg(c, cursor, LATEST + timedelta(hours=3), places)
    assert r.files[0].get("missing")
    assert r.cursor == LATEST
