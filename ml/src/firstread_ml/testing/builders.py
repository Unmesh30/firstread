"""Build tiny ``Impressions`` by hand, for tests."""

import numpy as np

from firstread_ml.mind.data import Impressions


def make_impressions(
    hists: list[list[int]], cands: list[list[int]], labels: list[list[int]]
) -> Impressions:
    def csr(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        sizes = np.array([len(r) for r in rows], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        flat = np.array([x for r in rows for x in r])
        return off, flat

    ho, h = csr(hists)
    co, c = csr(cands)
    _, lab = csr(labels)
    n = len(cands)
    return Impressions(
        imp_id=np.arange(n, dtype=np.int64),
        time=np.arange(n, dtype=np.int64) * 100,
        hist_offsets=ho,
        hist=h.astype(np.int32),
        cand_offsets=co,
        cand=c.astype(np.int32),
        label=lab.astype(np.int8),
    )
