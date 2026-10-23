"""The cold-start user vector: chosen topics first, clicks as they arrive (SPEC §7.4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import l2_normalize


def topic_weight(n_clicks: int, k: float) -> float:
    """``k / (k + n)``: 1 with no clicks, half at k clicks, → 0 as history grows."""
    if k <= 0:
        return 0.0 if n_clicks > 0 else 1.0
    return float(k / (k + max(n_clicks, 0)))


def topics_vector(
    centroids: Mapping[str, NDArray[np.floating[Any]]],
    topics: Sequence[str],
    population: NDArray[np.floating[Any]],
) -> NDArray[np.float32]:
    """Normalized mean of the chosen topics' centroids; the population vector when
    there are no topics or no centroid for any of them."""
    vecs = [np.asarray(centroids[t], np.float32) for t in topics if t in centroids]
    if not vecs:
        return l2_normalize(np.asarray(population, np.float32))
    return l2_normalize(np.mean(vecs, axis=0))


def cold_start_vector(
    centroids: Mapping[str, NDArray[np.floating[Any]]],
    topics: Sequence[str],
    click_vec: NDArray[np.floating[Any]] | None,
    n_clicks: int,
    k: float,
    population: NDArray[np.floating[Any]],
) -> NDArray[np.float32]:
    """``normalize(w·c_topics + (1 - w)·u_click)`` with ``w = k / (k + n)``."""
    c = topics_vector(centroids, topics, population)
    if click_vec is None or n_clicks == 0:
        return c
    w = topic_weight(n_clicks, k)
    u = l2_normalize(np.asarray(click_vec, np.float32))
    return l2_normalize(w * c + (1.0 - w) * u)
