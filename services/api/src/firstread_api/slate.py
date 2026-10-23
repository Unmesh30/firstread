"""The feed slate and its cursor (ADR 0017).

The first page ranks a slate of up to ``slate_size`` items and stores it in
``u:{uid}:slate`` (30 min). Later pages are served from it:

* no item is ever repeated: the slate keeps every served id, and re-ranking excludes
  them;
* the slate remembers the history version it was ranked with; a newer version (a
  click) or a changed hidden set re-ranks the rest, so a click shows on the next page;
* the same page asked again within 60 s returns the cached page (a retry, or a
  client that lost the response), not the items after it.

The cursor is opaque to clients: base64url JSON ``{"s": slate_id, "o": offset,
"v": 1}``. A malformed one is a 400 ``invalid_cursor``; one for a slate this reader
doesn't have (expired, replaced, or another reader's) is a 400 ``cursor_expired``,
and the app starts over from the first page.

The slate is stored zlib-compressed (~6 KB for 100 items): Upstash bandwidth is a
budget too (ADR 0011).
"""

from __future__ import annotations

import base64
import binascii
import json
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

from firstread_api.errors import ApiError

CURSOR_VERSION = 1
MAX_OFFSET = 10_000
MAX_SERVED = 1_000  # served ids kept per slate (excluded from every re-rank)


@dataclass
class SlateItem:
    id: str
    final: float
    model: float
    fresh: float
    trend: float
    reasons: list[list[str]]  # [[code, text], ...]

    def components(self) -> list[float]:
        """What ``served_slates.components`` stores per item, in order."""
        return [self.model, self.fresh, self.trend, self.final]


@dataclass
class Slate:
    id: str
    hist_version: int
    hidden_sig: str
    items: list[SlateItem]  # in ranked order; served ones are skipped
    served: list[str] = field(default_factory=list)
    pages: dict[str, dict[str, Any]] = field(default_factory=dict)  # offset -> {at, ids}
    page: int = 0  # pages served so far (``served_slates.page`` of the next one)
    created: float = field(default_factory=time.time)

    def remaining(self) -> list[SlateItem]:
        served = set(self.served)
        return [i for i in self.items if i.id not in served]

    def by_id(self) -> dict[str, SlateItem]:
        return {i.id: i for i in self.items}

    def cached_page(self, offset: int, now: float, ttl_s: float) -> list[SlateItem] | None:
        """The page served at ``offset`` less than ``ttl_s`` ago, if any."""
        p = self.pages.get(str(offset))
        if p is None or now - float(p["at"]) >= ttl_s:
            return None
        items = self.by_id()
        return [items[i] for i in p["ids"] if i in items]

    def rerank(self, items: list[SlateItem], hist_version: int, hidden_sig: str) -> None:
        """Replace the unserved part with a new ranking; keep items a cached page names."""
        cached = {i for p in self.pages.values() for i in p["ids"]}
        self.items = [i for i in self.items if i.id in cached] + items
        self.hist_version = hist_version
        self.hidden_sig = hidden_sig

    def serve(self, offset: int, items: list[SlateItem], now: float, ttl_s: float) -> None:
        """Record a page: its ids join ``served`` and the page cache (old pages pruned)."""
        ids = [i.id for i in items]
        known = set(self.served)
        self.served = (self.served + [i for i in ids if i not in known])[-MAX_SERVED:]
        self.pages = {k: p for k, p in self.pages.items() if now - float(p["at"]) < ttl_s}
        self.pages[str(offset)] = {"at": now, "ids": ids}
        self.page += 1

    def dumps(self) -> bytes:
        raw = json.dumps(asdict(self), separators=(",", ":")).encode()
        return zlib.compress(raw, 6)

    @classmethod
    def loads(cls, blob: bytes) -> Slate:
        d = json.loads(zlib.decompress(blob))
        d["items"] = [SlateItem(**i) for i in d["items"]]
        return cls(**d)


@dataclass(frozen=True)
class Cursor:
    slate_id: str
    offset: int

    def encode(self) -> str:
        raw = json.dumps(
            {"s": self.slate_id, "o": self.offset, "v": CURSOR_VERSION}, separators=(",", ":")
        )
        return base64.urlsafe_b64encode(raw.encode()).rstrip(b"=").decode()

    @classmethod
    def decode(cls, token: str) -> Cursor:
        try:
            if len(token) > 200:
                raise ValueError("too long")
            d = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
            if not isinstance(d, dict) or set(d) != {"s", "o", "v"} or d["v"] != CURSOR_VERSION:
                raise ValueError("bad shape")
            if not isinstance(d["o"], int) or isinstance(d["o"], bool):
                raise ValueError("bad offset")
            slate_id, offset = str(uuid.UUID(d["s"])), d["o"]
            if not 0 < offset <= MAX_OFFSET:
                raise ValueError("bad offset")
        except (ValueError, KeyError, TypeError, binascii.Error, UnicodeDecodeError):
            raise ApiError(400, "invalid cursor", "invalid_cursor") from None
        return cls(slate_id, offset)


def cursor_expired() -> ApiError:
    return ApiError(400, "this feed has expired; start from the first page", "cursor_expired")


def new_slate_id() -> str:
    return str(uuid.uuid4())
