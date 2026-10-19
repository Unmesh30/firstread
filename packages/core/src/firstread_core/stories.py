"""Story thresholds and grouping, shared by ingestion and the feed (ADR 0009, 0013).

Two kinds of threshold, per pair type (a title-only vector and a title+description
vector of the same story sit further apart than two title-only vectors):

* ``VALIDATED_BY_PAIR`` → (τ_dup, τ_story): ingestion's duplicate and same-story
  thresholds, validated on hand-labelled live pairs (≥ 95% precision per bin). The
  feed also uses τ_story for "near-identical to something you already read".
* ``TAU_FEED_BY_PAIR`` → τ_feed: the looser threshold for *grouping* candidates
  that are the same story under different cluster ids (big stories split, ADR
  0009). Rule, fixed before computing it: on the pooled 2a + 2b labelled bins, the
  lowest bin edge from which every bin with ≥ 3 pairs is ≥ 90% same story or
  duplicate (:func:`threshold_from_bins`). 90%, not 95%: a feed-time false merge only
  hides one item from one slate, while an ingestion false merge corrupts data.

Caveats (ADR 0013): the labelled pairs were *nearest neighbours*, so their precision
doesn't carry over to arbitrary pairs in a candidate set, and the 2b sample found
different-story templated headlines at 0.85-0.92. Grouping is therefore complete
linkage (:func:`group`: an item joins a group only if it is ≥ τ_feed to every
member), never chained, and a pair type falls back to its τ_story if a reviewed
merge turns out to be two different stories.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

TextSource = Literal["title", "title_summary"]
PairType = tuple[TextSource, TextSource]


def pair_type(a: TextSource, b: TextSource) -> PairType:
    return (a, b) if a <= b else (b, a)  # order-free


# (tau_dup, tau_story). Values are only ever raised from what the data allowed.
#  - title-title: validated on the 2b re-check (144 LLM-labeled pairs, 26 duplicates).
#    tau_story rose 0.875 -> 0.925: templated headlines sit at 0.85-0.92.
#  - the two mixed/RSS types are PROVISIONAL (too few duplicates to validate tau_dup).
VALIDATED_BY_PAIR: dict[PairType, tuple[float, float]] = {
    ("title", "title"): (0.975, 0.925),
    ("title", "title_summary"): (0.95, 0.80),  # provisional
    ("title_summary", "title_summary"): (0.97, 0.875),  # provisional
}

FEED_PRECISION = 0.90
FEED_MIN_BIN = 3
# Computed by threshold_from_bins on docs/learn/data/02-thresholds{,-2b}.json pooled;
# test_stories.py recomputes them from the committed files.
TAU_FEED_BY_PAIR: dict[PairType, float] = {
    ("title", "title"): 0.875,
    ("title", "title_summary"): 0.75,
    ("title_summary", "title_summary"): 0.875,
}

BIN_KEYS: dict[str, PairType] = {
    "title-title": ("title", "title"),
    "title-title_summary": ("title", "title_summary"),
    "title_summary-title_summary": ("title_summary", "title_summary"),
}


def tau_story(a: TextSource, b: TextSource) -> float:
    return VALIDATED_BY_PAIR[pair_type(a, b)][1]


def tau_feed(a: TextSource, b: TextSource) -> float:
    return TAU_FEED_BY_PAIR[pair_type(a, b)]


def threshold_from_bins(
    bins: Mapping[str, Mapping[str, int]],
    precision: float = FEED_PRECISION,
    min_bin: int = FEED_MIN_BIN,
) -> float | None:
    """Lowest bin edge from which every bin with ≥ ``min_bin`` pairs has at least
    ``precision`` same-story-or-duplicate labels. None if no edge qualifies."""
    edges = sorted(bins, key=float)

    def good(edge: str) -> bool:
        counts = bins[edge]
        n = sum(counts.values())
        if n < min_bin:
            return True
        return (counts.get("same_story", 0) + counts.get("duplicate", 0)) / n >= precision

    for k, edge in enumerate(edges):
        if all(good(e) for e in edges[k:]):
            return float(edge)
    return None


def pool_bins(*docs: Mapping[str, Any]) -> dict[PairType, dict[str, dict[str, int]]]:
    """Sum the per-bin label counts of several threshold reports, per pair type."""
    out: dict[PairType, dict[str, dict[str, int]]] = {}
    for doc in docs:
        for key, t in doc["types"].items():
            pt = BIN_KEYS[key]
            for edge, counts in t["bins"].items():
                into = out.setdefault(pt, {}).setdefault(edge, {})
                for label, n in counts.items():
                    into[label] = into.get(label, 0) + int(n)
    return out


def group(
    vectors: NDArray[np.floating[Any]],
    text_sources: Sequence[TextSource],
    clusters: Sequence[uuid.UUID],
    thresholds: Mapping[PairType, float] = TAU_FEED_BY_PAIR,
) -> list[int]:
    """Group items that are the same story, without chaining. Returns a group id per item.

    Items are taken in the order given (callers pass relevance or score order, so the
    first member of a group is its leader). An item joins:

    1. the group that already holds its ``story_cluster_id`` (ingestion said so); else
    2. the first group in which its cosine to **every** member is ≥ the threshold for
       that pair type (complete linkage); else
    3. a new group.

    So A~B and B~C never put A and C together unless A~C as well.
    """
    v = np.asarray(vectors, np.float32)
    n = len(v)
    if n == 0:
        return []
    sims = v @ v.T
    out = [-1] * n
    members: list[list[int]] = []
    by_cluster: dict[uuid.UUID, int] = {}
    for i in range(n):
        g = by_cluster.get(clusters[i])
        if g is None:
            for gid, mem in enumerate(members):
                if all(
                    sims[i, j] >= thresholds[pair_type(text_sources[i], text_sources[j])]
                    for j in mem
                ):
                    g = gid
                    break
        if g is None:
            g = len(members)
            members.append([])
        members[g].append(i)
        out[i] = g
        by_cluster.setdefault(clusters[i], g)
    return out


def near_any(
    vectors: NDArray[np.floating[Any]],
    text_sources: Sequence[TextSource],
    others: NDArray[np.floating[Any]],
    other_sources: Sequence[TextSource],
) -> NDArray[np.bool_]:
    """For each item, whether it's ≥ τ_story (same-story, validated) to any of ``others``.

    The feed uses this to drop items near-identical to something already read.
    """
    v = np.asarray(vectors, np.float32)
    o = np.asarray(others, np.float32)
    if not len(v) or not len(o):
        return np.zeros(len(v), np.bool_)
    sims = v @ o.T
    tau = np.array([[tau_story(a, b) for b in other_sources] for a in text_sources], np.float32)
    out: NDArray[np.bool_] = (sims >= tau).any(axis=1)
    return out
