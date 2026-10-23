"""Maximal marginal relevance with a per-story cap (ADR 0014)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MMRResult:
    order: list[int]  # picked items, in display order
    capped: list[int]  # items dropped because their story group was full, best first
    diversity_picks: set[int]  # picked items that MMR moved ahead of a more relevant one


def mmr(
    relevance: NDArray[np.floating[Any]],
    vectors: NDArray[np.floating[Any]],
    lam: float,
    groups: Sequence[int] | None = None,
    cap: int = 2,
    k: int | None = None,
) -> MMRResult:
    """Greedy MMR: pick ``argmax λ·rel_i - (1 - λ)·max_{j picked} cos(i, j)``.

    * Relevance is min-max scaled to [0, 1] inside the request, so λ trades on a
      fixed scale whatever the blend produces.
    * An item whose story group already has ``cap`` picks is skipped (``capped``).
    * Ties go to the lower index, so λ = 1 is exactly a stable sort by relevance.
    """
    rel = np.asarray(relevance, np.float64)
    n = len(rel)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if n == 0:
        return MMRResult([], [], set())
    span = rel.max() - rel.min()
    scaled = (rel - rel.min()) / span if span > 0 else np.zeros(n)
    v = np.asarray(vectors, np.float32)
    sims = (v @ v.T).astype(np.float64)
    g = list(range(n)) if groups is None else list(groups)
    limit = n if k is None else min(k, n)
    remaining = np.ones(n, np.bool_)
    penalty = np.zeros(n)
    counts: dict[int, int] = {}
    order: list[int] = []
    capped: list[int] = []
    diverse: set[int] = set()
    while len(order) < limit and remaining.any():
        score = lam * scaled - (1.0 - lam) * penalty if order else scaled.copy()
        score[~remaining] = -np.inf
        i = int(np.argmax(score))
        remaining[i] = False
        if counts.get(g[i], 0) >= cap:
            capped.append(i)
            continue
        best_left = np.flatnonzero(remaining | (np.arange(n) == i))
        if order and scaled[i] < scaled[best_left].max():
            diverse.add(i)
        counts[g[i]] = counts.get(g[i], 0) + 1
        order.append(i)
        penalty = np.maximum(penalty, sims[i]) if len(order) > 1 else sims[i].copy()
    capped += [int(i) for i in np.flatnonzero(remaining) if counts.get(g[i], 0) >= cap]
    capped.sort(key=lambda i: (-rel[i], i))
    return MMRResult(order, capped, diverse)
