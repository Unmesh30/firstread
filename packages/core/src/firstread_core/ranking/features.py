"""Score components and the blend (ADR 0014).

Every function works on one request (1-D arrays) or, with ``offsets``, on many
requests at once (offline: one per MIND impression), with identical results.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Norm = Literal["raw", "zscore", "percentile"]
NORMS: tuple[Norm, ...] = ("raw", "zscore", "percentile")
Array = NDArray[np.float64]


def _groups(n: int, offsets: NDArray[np.int64] | None) -> NDArray[np.int64]:
    if offsets is None:
        return np.zeros(n, np.int64)
    sizes = np.diff(np.asarray(offsets, np.int64)).astype(np.int64)
    if sizes.sum() != n:
        raise ValueError(f"offsets cover {sizes.sum()} items, got {n}")
    out: NDArray[np.int64] = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    return out


def freshness(age_hours: NDArray[np.floating[Any]] | float, half_life_hours: float) -> Array:
    """``2^(-age / half_life)``; negative ages (clock skew) count as brand new."""
    if half_life_hours <= 0:
        raise ValueError("half_life_hours must be positive")
    age = np.maximum(np.asarray(age_hours, np.float64), 0.0)
    out: Array = np.power(2.0, -age / half_life_hours)
    return out


def _percentile(x: Array, g: NDArray[np.int64]) -> Array:
    """``(average rank - 0.5) / n`` within each group; ties share their average rank."""
    n_groups = int(g.max()) + 1 if len(g) else 0
    order = np.lexsort((x, g))
    xs, gs = x[order], g[order]
    sizes = np.bincount(g, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    pos = np.arange(len(x)) - starts[gs]  # 0-based position inside the group
    new = np.ones(len(x), np.bool_)
    new[1:] = (gs[1:] != gs[:-1]) | (xs[1:] != xs[:-1])
    tie_start = np.flatnonzero(new)
    tie_end = np.append(tie_start[1:], len(x)) - 1
    avg = (pos[tie_start] + pos[tie_end]) / 2.0 + 1.0
    ranks = np.empty(len(x), np.float64)
    ranks[order] = np.repeat(avg, tie_end - tie_start + 1)
    out: Array = (ranks - 0.5) / sizes[g]
    return out


def normalize(
    x: NDArray[np.floating[Any]], method: Norm, offsets: NDArray[np.int64] | None = None
) -> Array:
    """Put model scores on a per-request scale: ``raw``, ``zscore`` or ``percentile``."""
    v = np.asarray(x, np.float64)
    if method == "raw":
        return v.copy()
    g = _groups(len(v), offsets)
    if not len(v):
        return v.copy()
    if method == "zscore":
        n = np.bincount(g).astype(np.float64)
        mean = np.bincount(g, weights=v) / n
        var = np.bincount(g, weights=(v - mean[g]) ** 2) / n
        std = np.sqrt(var)
        out: Array = np.where(std[g] > 0, (v - mean[g]) / np.where(std[g] > 0, std[g], 1.0), 0.0)
        return out
    if method == "percentile":
        return _percentile(v, g)
    raise ValueError(f"unknown normalization {method!r}")


def trend_feature(
    raw: NDArray[np.floating[Any]], offsets: NDArray[np.int64] | None = None
) -> Array:
    """Per-request percentile of the trend score, 0 where it isn't trending (raw ≤ 0).

    Offline the raw score is decayed clicks, live it is decayed outlet breadth; the
    percentile makes only their ranks matter (ADR 0014).
    """
    v = np.asarray(raw, np.float64)
    if not len(v):
        return v.copy()
    out: Array = np.where(v > 0, _percentile(v, _groups(len(v), offsets)), 0.0)
    return out


def blend(
    model: NDArray[np.floating[Any]],
    fresh: NDArray[np.floating[Any]],
    trend: NDArray[np.floating[Any]],
    norm: Norm,
    w_fresh: float,
    w_trend: float,
    offsets: NDArray[np.int64] | None = None,
) -> Array:
    """``normalize(model) + w_fresh·fresh + w_trend·trend`` (w_model = 1)."""
    out: Array = (
        normalize(model, norm, offsets)
        + w_fresh * np.asarray(fresh, np.float64)
        + w_trend * np.asarray(trend, np.float64)
    )
    return out
