"""User-history features shared by training (ml) and serving (api).

A user is represented by their most recent clicked articles. Histories are
stored oldest → newest; we keep the newest ``HISTORY_LEN`` and right-align
them, so the newest click is always in the last slot and padding is on the left.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_DIM, l2_normalize

HISTORY_LEN = 50
PAD_INDEX = -1


def history_index_matrix(
    histories: Sequence[Sequence[int]], history_len: int = HISTORY_LEN
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Turn per-user lists of article indices into a padded batch.

    Returns ``(idx[B, L], mask[B, L])``: ``idx`` holds ``PAD_INDEX`` where
    ``mask`` is False.
    """
    idx = np.full((len(histories), history_len), PAD_INDEX, dtype=np.int64)
    mask = np.zeros((len(histories), history_len), dtype=np.bool_)
    for row, hist in enumerate(histories):
        recent = list(hist)[-history_len:] if history_len > 0 else []
        if recent:
            idx[row, history_len - len(recent) :] = recent
            mask[row, history_len - len(recent) :] = True
    return idx, mask


def pad_history(
    embs: NDArray[Any], history_len: int = HISTORY_LEN
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Pad one user's clicked-article embeddings (``[n, d]``, oldest first).

    Returns ``(hist[L, d], mask[L])`` with padding rows set to zero.
    """
    arr = np.asarray(embs, dtype=np.float32).reshape(-1, EMBED_DIM)
    hist = np.zeros((history_len, EMBED_DIM), dtype=np.float32)
    mask = np.zeros(history_len, dtype=np.bool_)
    recent = arr[-history_len:] if history_len > 0 else arr[:0]
    if len(recent):
        hist[history_len - len(recent) :] = recent
        mask[history_len - len(recent) :] = True
    return hist, mask


def gather_history(
    table: NDArray[np.float32], idx: NDArray[np.int64], mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Look up ``idx[B, L]`` in an embedding ``table[N, d]``; padded slots become zeros."""
    out = table[np.where(mask, idx, 0)]
    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def mean_history_vector(
    hist: NDArray[Any], mask: NDArray[np.bool_], fallback: NDArray[Any] | None = None
) -> NDArray[np.float32]:
    """Normalized mean of the unmasked history rows.

    Works on one user (``hist[L, d]``, ``mask[L]``) or a batch
    (``hist[B, L, d]``, ``mask[B, L]``). Users with no history get
    ``fallback`` (for example the population vector), or zeros.
    """
    h = np.asarray(hist, dtype=np.float32)
    m = np.asarray(mask, dtype=np.bool_)
    counts = m.sum(axis=-1, keepdims=True)
    summed = (h * m[..., None]).sum(axis=-2)
    vec = l2_normalize(summed / np.maximum(counts, 1))
    if fallback is not None:
        empty = counts[..., 0] == 0
        vec = np.where(empty[..., None], l2_normalize(fallback), vec).astype(np.float32)
    return vec
