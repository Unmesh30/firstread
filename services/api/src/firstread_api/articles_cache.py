"""An in-process LRU of article rows with their vectors (ADR 0017).

Articles never change after insert, so a cached row is never stale; it only ages out
of the LRU (20,000 rows, about 25 MB with fp16 vectors). Misses are fetched in one
query. This keeps Supabase egress per feed request to the rows this instance hasn't
seen yet (ADR 0011).
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_api.repo import Q

_HYDRATE = """
select id, title, source_name, source_domain, url, image_url, category, published_at,
       first_seen_at, story_cluster_id, text_source, embedding
from public.articles where id = any($1::uuid[])
"""


@dataclass(frozen=True)
class Article:
    id: uuid.UUID
    title: str
    source_name: str
    source_domain: str
    url: str
    image_url: str | None
    category: str
    published_at: datetime | None
    first_seen_at: datetime
    story_cluster_id: uuid.UUID
    text_source: str
    vector: NDArray[np.float16]

    @classmethod
    def from_row(cls, r: Any) -> Article:
        return cls(
            r["id"], r["title"], r["source_name"], r["source_domain"], r["url"],
            r["image_url"], r["category"], r["published_at"], r["first_seen_at"],
            r["story_cluster_id"], r["text_source"],
            np.asarray(r["embedding"].to_numpy(), np.float16),
        )  # fmt: skip


class ArticlesCache:
    def __init__(self, capacity: int = 20_000) -> None:
        self.capacity = capacity
        self.rows: OrderedDict[uuid.UUID, Article] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _put(self, a: Article) -> None:
        self.rows[a.id] = a
        self.rows.move_to_end(a.id)
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    async def get_many(self, q: Q, ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, Article]:
        """Rows for ``ids`` that still exist (retention may have deleted some)."""
        out: dict[uuid.UUID, Article] = {}
        missing: list[uuid.UUID] = []
        for i in dict.fromkeys(ids):
            a = self.rows.get(i)
            if a is None:
                missing.append(i)
            else:
                self.rows.move_to_end(i)
                out[i] = a
        self.hits += len(out)
        self.misses += len(missing)
        if missing:
            for r in await q.fetch("articles_hydrate", _HYDRATE, missing):
                a = Article.from_row(r)
                self._put(a)
                out[a.id] = a
        return out
