"""Exact duplicates, near-duplicates and story clusters (ADR 0009). Pure numpy, no I/O.

For each new candidate, in order:

1. **Exact duplicate:** its canonical URL, or its title fingerprint within the
   fingerprint window, matches a recent article or an earlier candidate in the
   batch. Not inserted.
2. **Near-duplicate:** cosine to the nearest recent article (or earlier kept
   candidate) ≥ ``tau_dup`` for that pair type. Not inserted.
3. **Same story:** ``tau_story`` ≤ cosine < ``tau_dup``: inserted into the
   neighbour's ``story_cluster_id``.
4. Otherwise a new story: its own id is its cluster id.

Vectors are L2-normalized, so a dot product is the cosine. Thresholds depend on
the *pair type*, because a title-only vector and a title+summary vector of the same
story are less similar than two title-only vectors.

Every decision carries the cluster it belongs to, including dropped duplicates, so
``story_coverage`` can count every domain that ran the story (trending, ADR 0010).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TextSource = Literal["title", "title_summary"]
Action = Literal["insert", "duplicate", "near_duplicate"]
PairType = tuple[TextSource, TextSource]


def pair_type(a: TextSource, b: TextSource) -> PairType:
    return (a, b) if a <= b else (b, a)  # order-free


# Per pair type: (tau_dup, tau_story). A threshold is accepted when every cosine bin at or
# above it has >= 95% precision (score_pairs.py, ADR 0009). Values are only ever raised.
#  - GDELT-GDELT: validated on the 2b re-check (144 LLM-labeled pairs, 26 duplicates,
#    per-story-capped sample from 3 replayed days). tau_story rose 0.875 -> 0.925:
#    templated headlines ("<State> Midterm Elections 2026: Key Races...", court case
#    listings) sit at 0.85-0.92 without being the same story.
#  - GDELT-RSS and RSS-RSS: PROVISIONAL. Neither sample reached 100 pairs and 10
#    duplicates (183 and 93 pairs over 2a+2b, 2 and 0 duplicates); kept stricter than
#    the data allowed.
VALIDATED_BY_PAIR: dict[PairType, tuple[float, float]] = {
    ("title", "title"): (0.975, 0.925),
    ("title", "title_summary"): (0.95, 0.80),  # provisional
    ("title_summary", "title_summary"): (0.97, 0.875),  # provisional
}


@dataclass(frozen=True)
class Thresholds:
    tau_dup: float = 0.975
    tau_story: float = 0.925
    # Per-pair-type values: {("title", "title_summary"): (tau_dup, tau_story)}.
    by_pair: Mapping[PairType, tuple[float, float]] = field(default_factory=dict)

    def for_pair(self, a: TextSource, b: TextSource) -> tuple[float, float]:
        return self.by_pair.get(pair_type(a, b), (self.tau_dup, self.tau_story))


@dataclass
class Recent:
    """Articles already stored in the dedupe window (loaded by store.py)."""

    ids: list[uuid.UUID]
    clusters: list[uuid.UUID]
    text_sources: list[TextSource]
    embeddings: NDArray[np.float32]  # [n, d], L2-normalized
    url_cluster: dict[str, uuid.UUID]  # url_canonical -> cluster
    fingerprint_cluster: dict[str, uuid.UUID]  # title_fingerprint -> cluster (fingerprint window)

    @classmethod
    def empty(cls, dim: int = 384) -> Recent:
        return cls([], [], [], np.zeros((0, dim), np.float32), {}, {})


@dataclass(frozen=True)
class Item:
    """What dedupe needs to know about one candidate."""

    url_canonical: str
    title_fingerprint: str
    text_source: TextSource
    seen_at: datetime


@dataclass(frozen=True)
class Decision:
    action: Action
    article_id: uuid.UUID | None  # new id when inserted
    cluster_id: uuid.UUID
    cosine: float | None  # to the nearest neighbour, if any
    reason: str  # url | fingerprint | cosine | new_story | same_story


def decide(
    items: Sequence[Item],
    vectors: NDArray[np.float32],
    recent: Recent,
    thresholds: Thresholds,
    make_id: Callable[[], uuid.UUID] = uuid.uuid4,
) -> list[Decision]:
    """One decision per item, in order. ``vectors[i]`` is item i's normalized embedding.

    ``make_id`` mints new article ids (a seeded factory makes two runs comparable).
    """
    sims_recent = vectors @ recent.embeddings.T if len(recent.ids) else None
    kept_rows: list[int] = []  # indices into items that were inserted
    kept_clusters: list[uuid.UUID] = []
    urls = dict(recent.url_cluster)
    prints = dict(recent.fingerprint_cluster)
    out: list[Decision] = []
    for i, it in enumerate(items):
        if it.url_canonical in urls:
            out.append(Decision("duplicate", None, urls[it.url_canonical], None, "url"))
            continue
        if it.title_fingerprint in prints:
            cluster = prints[it.title_fingerprint]
            out.append(Decision("duplicate", None, cluster, None, "fingerprint"))
            urls[it.url_canonical] = cluster
            continue
        best, best_cluster, best_src = -2.0, None, None
        if sims_recent is not None:
            j = int(np.argmax(sims_recent[i]))
            best, best_cluster = float(sims_recent[i, j]), recent.clusters[j]
            best_src = recent.text_sources[j]
        if kept_rows:
            sims = vectors[kept_rows] @ vectors[i]
            k = int(np.argmax(sims))
            if float(sims[k]) > best:
                best, best_cluster = float(sims[k]), kept_clusters[k]
                best_src = items[kept_rows[k]].text_source
        if best_cluster is not None and best_src is not None:
            tau_dup, tau_story = thresholds.for_pair(it.text_source, best_src)
            if best >= tau_dup:
                out.append(Decision("near_duplicate", None, best_cluster, best, "cosine"))
                urls[it.url_canonical] = best_cluster
                prints.setdefault(it.title_fingerprint, best_cluster)
                continue
            if best >= tau_story:
                aid = make_id()
                out.append(Decision("insert", aid, best_cluster, best, "same_story"))
                kept_rows.append(i)
                kept_clusters.append(best_cluster)
                urls[it.url_canonical] = best_cluster
                prints[it.title_fingerprint] = best_cluster
                continue
        aid = make_id()
        out.append(Decision("insert", aid, aid, best if best_cluster else None, "new_story"))
        kept_rows.append(i)
        kept_clusters.append(aid)
        urls[it.url_canonical] = aid
        prints[it.title_fingerprint] = aid
    return out
