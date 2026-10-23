"""Candidate generation in SQL: ANN by user vector, and dot products for pool members.

Both return ``(id, score, first_seen_at, story_cluster_id, source_domain)``, about
70 bytes a row, so the pre-hydrate blend has freshness, story and source for every
candidate, ANN hits outside the pools included (ADR 0017). Scores are inner products
of L2-normalized vectors, i.e. cosines (``<#>`` is the *negative* inner product).

**ANN** (pgvector 0.8): ``SET LOCAL`` inside the query's transaction, because session
settings don't survive Supabase's transaction pooler. ``hnsw`` mode uses the HNSW
index with iterative scans (``relaxed_order``) and re-sorts in a materialized CTE
(``+ 0`` forces the re-sort on Postgres 17); ``exact`` mode scans the 72 h window
through the ``first_seen_at`` index. ``docs/learn/data/03-ann.json`` decides which.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pgvector import HalfVector

from firstread_api.repo import Q

AnnMode = Literal["hnsw", "exact"]

_WHERE = (
    "first_seen_at > $2 and not (id = any($3::uuid[])) and not (source_domain = any($4::text[]))"
)
ANN_HNSW = f"""
with r as materialized (
  select id, embedding operator(extensions.<#>) $1::extensions.halfvec(384) as d,
         first_seen_at, story_cluster_id, source_domain
  from public.articles
  where {_WHERE}
  order by d
  limit $5
)
select id, -d as score, first_seen_at, story_cluster_id, source_domain from r order by d + 0
"""  # noqa: S608 (constant)
ANN_EXACT = f"""
select id, -(embedding operator(extensions.<#>) $1::extensions.halfvec(384)) as score,
       first_seen_at, story_cluster_id, source_domain
from public.articles
where {_WHERE}
order by (embedding operator(extensions.<#>) $1::extensions.halfvec(384)) + 0
limit $5
"""  # noqa: S608 (constant)
POOL_SCORES = """
select id, -(embedding operator(extensions.<#>) $1::extensions.halfvec(384)) as score,
       first_seen_at, story_cluster_id, source_domain
from public.articles
where id = any($2::uuid[]) and not (source_domain = any($3::text[]))
"""


@dataclass
class Candidate:
    id: uuid.UUID
    score: float  # cosine to the user vector
    first_seen_at: datetime
    cluster: uuid.UUID
    source_domain: str
    sources: set[str] = field(default_factory=set)  # ann, fresh, trending, topic:<t>


def _rows(rows: Sequence[Any], source: str) -> list[Candidate]:
    return [
        Candidate(
            r["id"],
            float(r["score"]),
            r["first_seen_at"],
            r["story_cluster_id"],
            r["source_domain"],
            {source},
        )
        for r in rows
    ]


async def ann(
    q: Q,
    user_vec: NDArray[np.float32],
    since: datetime,
    exclude_ids: Sequence[uuid.UUID],
    exclude_sources: Sequence[str],
    limit: int = 200,
    mode: AnnMode = "hnsw",
    ef_search: int = 200,
) -> list[Candidate]:
    vec = HalfVector(np.asarray(user_vec, np.float32))
    args = (vec, since, list(exclude_ids), list(exclude_sources), limit)
    async with q.transaction():
        if mode == "hnsw":
            await q.execute(
                "ann_settings", f"set local hnsw.ef_search = {int(max(ef_search, limit))}"
            )
            await q.execute("ann_settings", "set local hnsw.iterative_scan = relaxed_order")
            rows = await q.fetch("ann_hnsw", ANN_HNSW, *args)
        else:
            rows = await q.fetch("ann_exact", ANN_EXACT, *args)
    return _rows(rows, "ann")


async def pool_scores(
    q: Q,
    user_vec: NDArray[np.float32],
    ids: Sequence[uuid.UUID],
    exclude_sources: Sequence[str],
) -> list[Candidate]:
    if not ids:
        return []
    vec = HalfVector(np.asarray(user_vec, np.float32))
    rows = await q.fetch("pool_scores", POOL_SCORES, vec, list(ids), list(exclude_sources))
    return _rows(rows, "pool")
