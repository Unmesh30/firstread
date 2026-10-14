"""GDELT 2.0 Global Knowledge Graph (GKG) 15-minute files (ADR 0007).

The DOC 2.0 API throttled every request we made (HTTP 429), so live GDELT data
comes from the static GKG files instead: one zip every 15 minutes, no API.

* **Catch-up, not just the latest file.** GitHub's scheduled runs are often late
  or skipped, so we keep a cursor (the last processed slot) and process every
  15-minute slot since then, bounded to ``max_files`` (8 = 2 h). File names are
  derived from the clock; ``lastupdate.txt`` (fetched with ``Cache-Control:
  no-cache``, since cached copies can be hours old) only tells us the newest slot.
* **Staleness:** if the newest available file is over ``stale_after`` old, the run
  summary marks GDELT as stale.
* **Records:** URL, outlet domain, page title (``<PAGE_TITLE>`` in the Extras
  field, HTML-unescaped), sharing image, crawl time and, when present, the precise
  publication time. No summary: GDELT articles are embedded from the title.
* **Placement:** the outlet's country comes from GDELT's domain lookup plus our
  overrides; blocked or unplaceable domains are dropped. The main GKG stream is
  English-language sources (translated content is a separate stream).

Credit: data from the GDELT Project, https://www.gdeltproject.org.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from firstread_core.countries import US, fips_to_iso
from firstread_ingest.http import fetch_with_retry
from firstread_ingest.items import RawItem
from firstread_ingest.sources.gdelt_domains import DomainCountries

BASE = "https://data.gdeltproject.org/gdeltv2/"
LASTUPDATE = BASE + "lastupdate.txt"
SLOT = timedelta(minutes=15)
SOURCE_ID = "gdelt-gkg"
N_COLUMNS = 27
COL_DATE, COL_SOURCE, COL_URL, COL_IMAGE, COL_EXTRAS = 1, 3, 4, 18, 26
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_PUBTIME = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>")


def slot_floor(t: datetime) -> datetime:
    t = t.astimezone(UTC)
    return t.replace(minute=t.minute - t.minute % 15, second=0, microsecond=0)


def slot_name(slot: datetime) -> str:
    return slot.strftime("%Y%m%d%H%M%S")


def parse_slot(name: str) -> datetime:
    return datetime.strptime(name, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def slot_url(slot: datetime) -> str:
    return f"{BASE}{slot_name(slot)}.gkg.csv.zip"


@dataclass(frozen=True)
class LatestFile:
    slot: datetime
    size: int
    md5: str


def parse_lastupdate(text: str) -> LatestFile:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].endswith(".gkg.csv.zip"):
            name = parts[2].rsplit("/", 1)[-1].split(".", 1)[0]
            return LatestFile(parse_slot(name), int(parts[0]), parts[1].lower())
    raise ValueError("lastupdate.txt lists no .gkg.csv.zip file")


def plan_slots(cursor: datetime | None, latest: datetime, max_files: int) -> list[datetime]:
    """Slots after ``cursor`` up to ``latest``, keeping only the newest ``max_files``."""
    first = latest - SLOT * (max_files - 1)
    if cursor is not None:
        first = max(first, cursor + SLOT)
    slots = []
    t = first
    while t <= latest:
        slots.append(t)
        t += SLOT
    return slots


@dataclass(frozen=True)
class GkgRecord:
    url: str
    domain: str
    title: str
    image_url: str | None
    seen_at: datetime
    published_at: datetime | None


def _ts(value: str) -> datetime | None:
    try:
        return parse_slot(value)
    except ValueError:
        return None


def parse_gkg(blob: bytes) -> list[GkgRecord]:
    """Records with a URL, domain and page title from one zipped GKG file."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    csv.field_size_limit(sys.maxsize)
    out = []
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) != N_COLUMNS:
            continue
        m = _TITLE.search(row[COL_EXTRAS])
        url, domain = row[COL_URL].strip(), row[COL_SOURCE].strip().lower()
        seen = _ts(row[COL_DATE].strip())
        if not (m and url.startswith("http") and domain and seen):
            continue
        pub = _PUBTIME.search(row[COL_EXTRAS])
        out.append(
            GkgRecord(
                url=url,
                domain=domain,
                title=html.unescape(m.group(1)).strip(),  # GDELT escapes non-ASCII
                image_url=row[COL_IMAGE].strip() or None,
                seen_at=seen,
                published_at=_ts(pub.group(1)) if pub else None,
            )
        )
    return out


@dataclass
class GkgResult:
    items: list[RawItem] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    cursor: datetime | None = None  # the newest slot processed (or attempted)
    newest_available: datetime | None = None
    stale: bool = False
    records: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    domains: Counter[str] = field(default_factory=Counter)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "cursor": slot_name(self.cursor) if self.cursor else None,
            "newest_available": slot_name(self.newest_available) if self.newest_available else None,
            "stale": self.stale,
            "records": self.records,
            "dropped": dict(self.dropped),
            "top_domains": dict(self.domains.most_common(20)),
            "error": self.error,
        }


def to_raw(rec: GkgRecord, fips: str) -> RawItem:
    return RawItem(
        origin="gdelt",
        source_id=SOURCE_ID,
        url=rec.url,
        title=rec.title,
        source_name=rec.domain,
        source_domain=rec.domain,
        geo="us" if fips == US else "intl",
        seen_at=rec.seen_at,
        published_at=rec.published_at,
        description=None,
        image_url=rec.image_url,
        language="en",
        source_country=fips_to_iso(fips),
        topic_hint=None,
    )


async def fetch_gkg(
    client: httpx.AsyncClient,
    cursor: datetime | None,
    now: datetime,
    places: DomainCountries,
    max_files: int = 8,
    stale_after: timedelta = timedelta(hours=1),
) -> GkgResult:
    """Fetch and filter every GKG file since ``cursor``. Never raises: errors are data."""
    result = GkgResult(cursor=cursor)
    try:
        resp = await fetch_with_retry(
            client, LASTUPDATE, headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}
        )
        resp.raise_for_status()
        latest = parse_lastupdate(resp.text)
    except Exception as exc:
        result.error = f"lastupdate: {type(exc).__name__}: {exc}"[:300]
        result.stale = True
        return result
    # Never trust a newest slot from the future (clock skew): bound it by our clock.
    newest = min(latest.slot, slot_floor(now))
    result.newest_available = newest
    result.stale = now - newest > stale_after + SLOT
    for slot in plan_slots(cursor, newest, max_files):
        entry: dict[str, Any] = {"slot": slot_name(slot)}
        result.files.append(entry)
        try:
            r = await fetch_with_retry(client, slot_url(slot))
            entry["status"] = r.status_code
            if r.status_code == 404:  # GDELT occasionally skips a slot
                result.cursor = slot
                continue
            r.raise_for_status()
            if slot == latest.slot and hashlib.md5(r.content).hexdigest() != latest.md5:  # noqa: S324
                raise ValueError("MD5 differs from lastupdate.txt")
            records = parse_gkg(r.content)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            break  # keep the cursor before this slot, so the next run retries it
        entry["records"] = len(records)
        result.records += len(records)
        for rec in records:
            result.domains[rec.domain] += 1
            if places.blocked(rec.domain):
                result.dropped["blocked_domain"] += 1
                continue
            fips = places.fips(rec.domain)
            if fips is None:
                result.dropped["unplaced_domain"] += 1
                continue
            result.items.append(to_raw(rec, fips))
        result.cursor = slot
    return result
