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

**Exclusions run in SQL** (ids and clusters go in as parameters: ingress is free), so
no ANN slot is spent on an item the feed would drop: seen, clicked, hidden and
already-served ids; hidden sources; every story the reader has read (their history's
clusters) or hidden (the hidden articles' clusters, looked up server-side).
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


def _excluding(ids: int, sources: int, clusters: int, hidden: int) -> str:
    """The exclusion clause, with the given placeholder numbers (ints, never input)."""
    clause = (
        "not (id = any($I::uuid[])) and not (source_domain = any($S::text[])) "
        "and not (story_cluster_id = any($C::uuid[])) "
        "and not (story_cluster_id = any(array(select h.story_cluster_id "
        "from public.articles h where h.id = any($H::uuid[]))))"
    )
    for k, v in (("$I", ids), ("$S", sources), ("$C", clusters), ("$H", hidden)):
        clause = clause.replace(k, f"${int(v)}")
    return clause


# $1 user vector, $2 window start, $3 ids, $4 sources, $5 limit, $6 clusters, $7 hidden ids
_WHERE = "first_seen_at > $2 and " + _excluding(3, 4, 6, 7)
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
POOL_SCORES = f"""
select id, -(embedding operator(extensions.<#>) $1::extensions.halfvec(384)) as score,
       first_seen_at, story_cluster_id, source_domain
from public.articles
where id = any($2::uuid[]) and {_excluding(3, 4, 5, 6)}
"""  # noqa: S608 (constant)


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


@dataclass(frozen=True)
class Exclusions:
    """What no candidate may be: these ids, from these sources, or in these stories."""

    ids: Sequence[uuid.UUID] = ()  # seen, clicked, hidden, already served
    sources: Sequence[str] = ()  # hidden sources
    clusters: Sequence[uuid.UUID] = ()  # stories the reader read
    hidden: Sequence[uuid.UUID] = ()  # hidden articles: their stories go too

    def args(self) -> tuple[list[uuid.UUID], list[str], list[uuid.UUID], list[uuid.UUID]]:
        return list(self.ids), list(self.sources), list(self.clusters), list(self.hidden)


async def ann(
    q: Q,
    user_vec: NDArray[np.float32],
    since: datetime,
    exclude: Exclusions,
    limit: int = 200,
    mode: AnnMode = "hnsw",
    ef_search: int = 200,
) -> list[Candidate]:
    vec = HalfVector(np.asarray(user_vec, np.float32))
    ids, sources, clusters, hidden = exclude.args()
    args = (vec, since, ids, sources, limit, clusters, hidden)
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
    q: Q, user_vec: NDArray[np.float32], ids: Sequence[uuid.UUID], exclude: Exclusions
) -> list[Candidate]:
    """Cosines for pool members (ids from the in-process snapshot), minus exclusions."""
    if not ids:
        return []
    vec = HalfVector(np.asarray(user_vec, np.float32))
    rows = await q.fetch("pool_scores", POOL_SCORES, vec, list(ids), *exclude.args())
    return _rows(rows, "pool")
