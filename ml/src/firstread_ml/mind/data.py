"""Compact in-memory impressions (CSR layout) used by splits, baselines and eval.

An impression is one feed shown to one user: their click history plus a list of
candidate articles, each labelled clicked (1) or not (0). We store all
impressions' candidates in one flat array with offsets (like a sparse CSR
matrix), so slicing and scoring are vectorized numpy instead of Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _offsets(sizes: NDArray[np.int64]) -> NDArray[np.int64]:
    out = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=out[1:])
    return out


@dataclass(frozen=True)
class Impressions:
    imp_id: NDArray[np.int64]  # [n]
    time: NDArray[np.int64]  # [n] unix seconds
    hist_offsets: NDArray[np.int64]  # [n + 1]
    hist: NDArray[np.int32]  # [total history clicks], article index, oldest first
    cand_offsets: NDArray[np.int64]  # [n + 1]
    cand: NDArray[np.int32]  # [total candidates], article index
    label: NDArray[np.int8]  # [total candidates], 1 = clicked

    @property
    def n(self) -> int:
        return len(self.imp_id)

    @property
    def hist_len(self) -> NDArray[np.int64]:
        return np.diff(self.hist_offsets)

    @property
    def cand_len(self) -> NDArray[np.int64]:
        return np.diff(self.cand_offsets)

    @property
    def cand_imp(self) -> NDArray[np.int64]:
        """Impression row of each flat candidate."""
        return np.repeat(np.arange(self.n, dtype=np.int64), self.cand_len)

    def histories(self, rows: slice | None = None) -> list[NDArray[np.int32]]:
        """Per-impression history arrays for a contiguous range of rows."""
        start, stop, _ = (rows or slice(None)).indices(self.n)
        offs = self.hist_offsets[start : stop + 1]
        if len(offs) < 2:
            return []
        return np.split(self.hist[offs[0] : offs[-1]], offs[1:-1] - offs[0])

    def take(self, rows: NDArray[np.bool_] | NDArray[np.int64]) -> Impressions:
        """Keep a subset of impressions (boolean mask or sorted row indices)."""
        idx = np.flatnonzero(rows) if rows.dtype == np.bool_ else np.asarray(rows)
        h_sizes, c_sizes = self.hist_len[idx], self.cand_len[idx]
        h_sel = _ranges(self.hist_offsets[idx], h_sizes)
        c_sel = _ranges(self.cand_offsets[idx], c_sizes)
        return Impressions(
            imp_id=self.imp_id[idx],
            time=self.time[idx],
            hist_offsets=_offsets(h_sizes),
            hist=self.hist[h_sel],
            cand_offsets=_offsets(c_sizes),
            cand=self.cand[c_sel],
            label=self.label[c_sel],
        )

    @staticmethod
    def concat(parts: list[Impressions]) -> Impressions:
        return Impressions(
            imp_id=np.concatenate([p.imp_id for p in parts]),
            time=np.concatenate([p.time for p in parts]),
            hist_offsets=_offsets(np.concatenate([p.hist_len for p in parts])),
            hist=np.concatenate([p.hist for p in parts]),
            cand_offsets=_offsets(np.concatenate([p.cand_len for p in parts])),
            cand=np.concatenate([p.cand for p in parts]),
            label=np.concatenate([p.label for p in parts]),
        )


def _ranges(starts: NDArray[np.int64], sizes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Concatenate ``arange(s, s + k)`` for each (s, k) without a Python loop."""
    run_start = np.cumsum(sizes) - sizes  # where each run begins in the output
    total = int(sizes.sum())
    return np.arange(total, dtype=np.int64) - np.repeat(run_start - starts, sizes)
