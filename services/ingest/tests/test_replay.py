"""Replay over generated GKG files: clock stepping, cursor, missing slots, retention, pools."""

import csv
import io
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_ingest import replay
from firstread_ingest.config import Settings
from firstread_ingest.models import TopicClassifier
from firstread_ingest.pipeline import EmbedFn
from firstread_ingest.sources import gkg
from firstread_ingest.sources.gdelt_domains import DomainCountries

FIX = next((Path(__file__).parent / "fixtures" / "gkg").glob("*.gkg.csv.zip"))
T = gkg.parse_slot(FIX.name.split(".")[0])  # 2026-10-14 18:45


def slot_zip(slot: Any, cache: Path) -> None:
    """The fixture's records re-dated to ``slot``, with slot-specific URLs and titles."""
    name = gkg.slot_name(slot)
    with zipfile.ZipFile(FIX) as z:
        rows = list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode()), delimiter="\t",
                               quoting=csv.QUOTE_NONE))  # fmt: skip
    out = io.StringIO()
    w = csv.writer(
        out, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\"
    )
    for r in rows:
        r[1] = name
        r[4] = f"{r[4]}-{name}"
        r[26] = r[26].replace("Synthetic GKG headline", f"Story {name} headline")
        r[26] = r[26].replace(
            "<PAGE_PRECISEPUBTIMESTAMP>20261014184500", f"<PAGE_PRECISEPUBTIMESTAMP>{name}"
        )
        w.writerow(r)
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cache / f"{name}.gkg.csv.zip", "w") as z:
        z.writestr(f"{name}.gkg.csv", out.getvalue())


def test_window_and_default_end() -> None:
    slots = replay.window(T, days=1)
    assert len(slots) == 96
    assert slots[-1] == T
    assert slots[0] == T - timedelta(hours=23, minutes=45)
    assert replay.default_end(T + timedelta(minutes=50)) == T


async def test_transport_serves_only_gdelt_as_of_the_slot(tmp_path: Path) -> None:
    slot_zip(T, tmp_path)
    slot_zip(T + gkg.SLOT, tmp_path)  # a future file must not be visible yet
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("domain,countrycode,cnt\n")
    async with httpx.AsyncClient(transport=replay.replay_transport(T, tmp_path, lookup)) as c:
        latest = gkg.parse_lastupdate((await c.get(gkg.LASTUPDATE)).text)
        assert latest.slot == T
        assert (await c.get(gkg.slot_url(T))).status_code == 200
        assert (await c.get(gkg.slot_url(T + gkg.SLOT))).status_code == 404
        with pytest.raises(RuntimeError, match="unexpected request"):
            await c.get("https://feeds.bbci.co.uk/news/rss.xml")


async def test_download_skips_cached_and_records_missing(tmp_path: Path) -> None:
    slot_zip(T, tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(404 if "184500" not in str(request.url) and "190000" in str(
            request.url) else 200, content=b"zip")  # fmt: skip

    async def no_sleep(_: float) -> None: ...

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        out = await replay.download(
            c, [T, T + gkg.SLOT, T + 2 * gkg.SLOT], tmp_path, sleep=no_sleep
        )
    assert len(seen) == 2  # T was cached
    assert out == {"missing": [gkg.slot_name(T + gkg.SLOT)], "failed": []}
    assert replay.cached(T + 2 * gkg.SLOT, tmp_path).is_file()


@pytest.mark.integration
async def test_replay_steps_the_clock_and_behaves_like_production(
    tmp_path: Path,
    classifier: TopicClassifier,
    fake_embed: EmbedFn,
    places: DomainCountries,
    redis_test: Any,
) -> None:
    await skip_or_fail_if_unreachable_async()
    old, missing, now_slot = T - timedelta(hours=50), T - gkg.SLOT, T
    slot_zip(old, tmp_path)
    slot_zip(now_slot, tmp_path)
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("domain,countrycode,cnt\n")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    settings.ingest_gkg_max_per_story = 100  # synthetic titles all look like one story
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            result = await replay.replay(settings, classifier, [old, missing, now_slot], conn,
                                         redis_test, places, lookup, cache=tmp_path,
                                         embed=fake_embed, progress=lambda _: None)  # fmt: skip
            assert [r["slot"] for r in result.runs] == [
                gkg.slot_name(s) for s in (old, missing, now_slot)
            ]
            assert result.runs[0]["inserted"] == 8
            assert result.runs[1]["inserted"] == 0
            assert result.runs[2]["inserted"] == 8
            # The 50 h old slot's coverage passes the 48 h retention once the clock moves on.
            assert result.runs[0]["retention"]["story_coverage"] == 0
            assert sum(r["retention"]["story_coverage"] for r in result.runs[1:]) > 0
            assert await conn.fetchval("select count(*) from public.articles") == 16
            state = await replay.final_state(conn, redis_test)
            assert state["zcard"]["pool:fresh"] == 8  # only the last slot is within 24 h
            assert set(state["zcard"]) >= {"pool:trending", "pool:topic:us", "pool:topic:world"}
            assert state["sizes"]["articles"] == 16
            assert result.trending_days  # a snapshot for the last replayed day
            again = await replay.replay(settings, classifier, [old, missing, now_slot], conn,
                                        redis_test, places, lookup, cache=tmp_path,
                                        embed=fake_embed, progress=lambda _: None)  # fmt: skip
            assert sum(r["inserted"] for r in again.runs) == 0
        finally:
            await conn.close()


def test_aggregates_strip_headlines_and_runs() -> None:
    report = {
        "trending_days": {"2026-10-14": [{"headline": "Secret", "topic": "us", "domains": 3}]},
        "runs": [{"slot": "x"}],
        "totals": {"inserted": 1},
    }
    agg = replay.aggregates(report)
    assert agg["trending_days"]["2026-10-14"] == [{"topic": "us", "domains": 3}]
    assert "runs" not in agg
    assert "Secret" not in str(agg)


def test_press_release_counts_and_size_estimate() -> None:
    runs = [
        {"dropped": {"press_release:financial_results": 2, "stale": 5},
         "gdelt": {"records": 100, "dropped": {"press_wire_domain": 3}}},
        {"dropped": {"press_release:company_prefix": 1},
         "gdelt": {"records": 100, "dropped": {}}},
    ]  # fmt: skip
    pr = replay.press_release_counts(runs)
    assert pr["press_wire_domain"] == 3
    assert pr["announcement_titles"] == 3
    assert pr["share_of_gdelt_records"] == 0.03
    state = {"sizes": {"articles": 1000, "total": 3_200_000, "heap": 1_600_000,
                       "indexes": 1_600_000, "hnsw": 1_200_000, "coverage": 1_000_000}}  # fmt: skip
    est = replay.size_estimate(state, gdelt_per_day=4800, rss_per_day=200)
    assert est["bytes_per_article"] == 3200
    assert est["with_rss"]["mb_14_days"] == pytest.approx((3200 * 5000 * 14 + 1e6) / 1e6, rel=1e-3)
