import csv
import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from firstread_ingest.http import make_client
from firstread_ingest.sources import gkg
from firstread_ingest.sources.gdelt_domains import (
    DomainCountries,
    load_config,
    load_domain_countries,
    load_press_paths,
    load_press_wires,
    parse_lookup,
)

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
    # 404s beyond the 2 h catch-up horizon count as skipped; younger ones are pending,
    # so the cursor holds at 17:00 and the next run retries 17:15 onwards.
    assert r.cursor == LATEST - timedelta(minutes=105)
    top = r.summary()["top_domains"]
    assert "iheart.com" not in top  # blocked outlets don't crowd the list
    assert "no-such-outlet-firstread.example" not in top  # nor unplaceable ones
    assert top["wisn.com"] == 1
    assert sum(top.values()) == len(r.items)


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


def test_press_wire_domains_are_dropped_and_counted() -> None:
    places = DomainCountries(
        parse_lookup("domain,countrycode,cnt\nprnewswire.com,US,9\nwisn.com,US,9\n"),
        {},
        frozenset(),
        frozenset({"prnewswire.com"}),
    )
    assert places.press_wire("www.prnewswire.com")
    assert not places.press_wire("wisn.com")


def test_repo_press_wire_list_includes_the_spot_check_domains() -> None:
    wires = load_press_wires()
    assert {"finanznachrichten.de", "equitybulls.com", "prnewswire.com"} <= wires


def test_court_records_database_is_blocked() -> None:
    _, blocklist = load_config()
    assert "indiankanoon.org" in blocklist


MANILA = "https://www.manilatimes.net/2026/10/12"


def test_repo_review_of_2026_10_18(tmp_path: Path) -> None:
    """A few entries from the owner's review of the PR #12 list, via the real config."""
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("domain,countrycode,cnt\nmanilatimes.net,RP,9\naol.com,US,9\n")
    places = load_domain_countries(lookup)
    assert places.press_wire("www.miragenews.com")
    assert places.press_wire("investegate.co.uk")
    assert places.blocked("www.boredpanda.com")
    assert places.blocked("fool.com")
    assert places.blocked("aol.co.uk")
    assert places.blocked("www.aol.com")
    # The Big News Network family, including members found in the replay data.
    for d in ("bignewsnetwork.com", "heraldglobe.com", "greekherald.com", "bruneinews.net"):
        assert places.blocked(d), d
    # Kept: manilatimes.net (minus its newswire path) and businesstoday.in.
    assert not places.blocked("manilatimes.net")
    assert not places.press_wire("manilatimes.net")
    assert not places.blocked("businesstoday.in")
    assert not places.press_wire("businesstoday.in")


def test_press_release_paths_drop_only_the_newswire_section() -> None:
    assert load_press_paths()["manilatimes.net"] == {"tmt-newswire"}
    places = DomainCountries({}, {}, frozenset(), press_paths=load_press_paths())
    wire = f"{MANILA}/tmt-newswire/globenewswire/atk-phantom-gaming-mice/2443415"
    assert places.press_release_url("www.manilatimes.net", wire)
    assert places.press_release_url("manilatimes.net", wire.replace("globenewswire", "pr-newswire"))
    assert not places.press_release_url("manilatimes.net", f"{MANILA}/news/national/flood/2443400")
    # Whole segments only, and only on the listed domain.
    assert not places.press_release_url("manilatimes.net", f"{MANILA}/news/tmt-newswire-x/1")
    assert not places.press_release_url("example.com", "https://example.com/tmt-newswire/a/1")


def _with_manila_rows(blob: bytes) -> bytes:
    """The fixture with wisn.com and fox4news.com moved to manilatimes.net URLs."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        rows = list(csv.reader(io.StringIO(z.read(name).decode()), delimiter="\t",
                               quoting=csv.QUOTE_NONE))  # fmt: skip
    urls = {
        "wisn.com": f"{MANILA}/tmt-newswire/globenewswire/teladoc-to-announce-results/2446026",
        "fox4news.com": f"{MANILA}/news/national/senate-passes-budget/2443401",
    }
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE,
                   escapechar="\\")  # fmt: skip
    for r in rows:
        if r[3] in urls:
            r[3], r[4] = "manilatimes.net", urls[r[3]]
        w.writerow(r)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, out.getvalue())
    return buf.getvalue()


async def test_fetch_drops_and_counts_press_release_paths(
    no_network: respx.MockRouter, places: DomainCountries
) -> None:
    places.countries["manilatimes.net"] = "RP"
    places.press_paths = load_press_paths()
    mock_gdelt(no_network, _with_manila_rows(FIX.read_bytes()))
    async with make_client() as c:
        r = await gkg.fetch_gkg(c, None, LATEST + timedelta(minutes=20), places)
    assert r.dropped["press_wire_path"] == 1
    kept = [i.url for i in r.items if i.source_domain == "manilatimes.net"]
    assert kept == [f"{MANILA}/news/national/senate-passes-budget/2443401"]
