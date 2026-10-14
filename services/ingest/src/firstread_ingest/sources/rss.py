"""Curated RSS/Atom feeds (ADR 0007).

* Fetched with httpx (feedparser only parses bytes, it never fetches).
* **Conditional GET:** the previous run's ``ETag`` / ``Last-Modified`` (kept in
  ``feed_state``) are sent back; a 304 means "nothing new" and yields no items.
* **Bounded concurrency:** at most ``concurrency`` feeds at once, and feeds on the
  same host are fetched one after another, so no publisher sees parallel hits.
* **One failing feed never fails the run:** its error is recorded in its result.
* We keep the headline, link, time and thumbnail. The description is carried in
  memory for the embedding text only and never stored.
"""

from __future__ import annotations

import asyncio
import calendar
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import feedparser
import httpx

from firstread_ingest.config import Feed
from firstread_ingest.http import fetch_with_retry
from firstread_ingest.items import RawItem


@dataclass
class FeedState:
    etag: str | None = None
    last_modified: str | None = None
    last_status: int | None = None
    consecutive_failures: int = 0


@dataclass
class FeedResult:
    feed: Feed
    state: FeedState
    items: list[RawItem] = field(default_factory=list)
    status: int | None = None
    error: str | None = None
    not_modified: bool = False


def _when(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(calendar.timegm(t), tz=UTC)
    return None


def _image(entry: Any) -> str | None:
    for thumb in entry.get("media_thumbnail") or []:
        if thumb.get("url"):
            return str(thumb["url"])
    for media in entry.get("media_content") or []:
        # Untyped media:content (e.g. the Guardian's) is an image in practice.
        kind = media.get("medium") or media.get("type") or ""
        if media.get("url") and (not kind or kind == "image" or kind.startswith("image/")):
            return str(media["url"])
    for enc in entry.get("enclosures") or []:
        if str(enc.get("type", "")).startswith("image/") and enc.get("href"):
            return str(enc["href"])
    return None


def parse_feed(feed: Feed, body: bytes, fetched_at: datetime) -> list[RawItem]:
    """Turn a feed document into raw items. Entries without a link or title are skipped."""
    parsed = feedparser.parse(body)
    if parsed.get("bozo") and not parsed.entries:
        raise ValueError(f"unparseable feed: {type(parsed.get('bozo_exception')).__name__}")
    language = str(parsed.feed.get("language") or "en").lower()
    items = []
    for e in parsed.entries:
        link, title = e.get("link"), e.get("title")
        if not link or not title:
            continue
        items.append(
            RawItem(
                origin="rss",
                source_id=feed.id,
                url=str(link),
                title=str(title),
                source_name=feed.source_name,
                source_domain=feed.domain,
                geo=feed.geo,
                seen_at=fetched_at,
                published_at=_when(e),
                description=e.get("summary") or e.get("description"),
                image_url=_image(e),
                language=language,
                source_country=None,
                topic_hint=feed.topic_hint,
            )
        )
    return items


async def fetch_feed(
    client: httpx.AsyncClient, feed: Feed, state: FeedState, now: datetime | None = None
) -> FeedResult:
    now = now or datetime.now(UTC)
    headers = {}
    if state.etag:
        headers["If-None-Match"] = state.etag
    if state.last_modified:
        headers["If-Modified-Since"] = state.last_modified
    new = FeedState(state.etag, state.last_modified, None, state.consecutive_failures)
    result = FeedResult(feed=feed, state=new)
    try:
        resp = await fetch_with_retry(client, feed.url, headers=headers)
        result.status = new.last_status = resp.status_code
        if resp.status_code == 304:
            result.not_modified = True
        elif resp.status_code == 200:
            result.items = parse_feed(feed, resp.content, now)
            new.etag = resp.headers.get("etag") or None
            new.last_modified = resp.headers.get("last-modified") or None
        else:
            raise ValueError(f"HTTP {resp.status_code}")
    except Exception as exc:  # one feed's failure is data, not a crash
        result.error = f"{type(exc).__name__}: {exc}"[:300]
        new.consecutive_failures += 1
        return result
    new.consecutive_failures = 0
    return result


async def fetch_all(
    client: httpx.AsyncClient,
    feeds: Sequence[Feed],
    states: dict[str, FeedState],
    concurrency: int = 8,
) -> list[FeedResult]:
    """Fetch every feed; same-host feeds run sequentially, at most ``concurrency`` hosts at once."""
    by_host: dict[str, list[Feed]] = defaultdict(list)
    for f in feeds:
        by_host[urlsplit(f.url).hostname or f.url].append(f)
    sem = asyncio.Semaphore(concurrency)

    async def host_worker(group: list[Feed]) -> list[FeedResult]:
        out = []
        async with sem:
            for f in group:
                out.append(await fetch_feed(client, f, states.get(f.id, FeedState())))
        return out

    groups = await asyncio.gather(*(host_worker(g) for g in by_host.values()))
    return [r for g in groups for r in g]
