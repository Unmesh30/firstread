"""Per-impression ranking metrics, vectorized over all impressions at once.

Every impression must have at least one clicked and one non-clicked candidate
(callers filter; see ``slices.restrict``). Definitions follow MIND's evaluation:

* AUC: Mann-Whitney U / (pos * neg). Tied scores share their average rank,
  which matches ``sklearn.metrics.roc_auc_score``.
* MRR: sum(label / rank) / sum(label) over the ranked list.
* nDCG@k: gain 2^label - 1, discount log2(rank + 1).

MRR and nDCG need a strict order. MIND's script uses numpy's argsort, so ties
fall in input order; we break ties with a seeded random key instead, so models
with many tied scores (popularity, random) are neither flattered nor punished
by the order candidates were logged in.
"""

import numpy as np
from numpy.typing import NDArray

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def per_impression(
    scores: NDArray[np.floating],
    labels: NDArray[np.integer],
    offsets: NDArray[np.int64],
    seed: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Return ``{metric: values[n_impressions]}``."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    y = np.asarray(labels, dtype=np.float64)
    n = len(offsets) - 1
    sizes = np.diff(offsets)
    imp = np.repeat(np.arange(n), sizes)
    npos = np.bincount(imp, weights=y, minlength=n)
    nneg = sizes - npos
    if (npos == 0).any() or (nneg == 0).any():
        raise ValueError("every impression needs at least one positive and one negative")
    within = np.arange(len(y)) - offsets[imp]  # 0-based position inside its impression

    # Descending by score, random tie-break; candidates stay grouped by impression.
    tiebreak = np.random.default_rng(seed).random(len(y))
    desc = np.lexsort((tiebreak, -scores, imp))
    rank = within + 1.0
    y_desc = y[desc]
    out = {"mrr": np.bincount(imp, weights=y_desc / rank, minlength=n) / npos}
    gain = (2.0**y_desc - 1.0) / np.log2(rank + 1.0)
    for k in (5, 10):
        dcg = np.bincount(imp, weights=gain * (rank <= k), minlength=n)
        ideal_cum = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
        idcg = ideal_cum[np.minimum(npos, k).astype(np.int64)]
        out[f"ndcg@{k}"] = dcg / idcg

    # AUC from average ranks (ascending), ties grouped per impression.
    asc = np.lexsort((scores, imp))
    s_asc = scores[asc]
    new_group = np.ones(len(y), dtype=np.bool_)
    new_group[1:] = (imp[1:] != imp[:-1]) | (s_asc[1:] != s_asc[:-1])
    starts = np.flatnonzero(new_group)
    ends = np.append(starts[1:], len(y)) - 1
    group_rank = (within[starts] + within[ends]) / 2.0 + 1.0
    avg_rank = np.repeat(group_rank, ends - starts + 1)
    pos_rank_sum = np.bincount(imp, weights=y[asc] * avg_rank, minlength=n)
    out["auc"] = (pos_rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)
    return {m: out[m] for m in METRICS}


def mean_metrics(
    scores: NDArray[np.floating],
    labels: NDArray[np.integer],
    offsets: NDArray[np.int64],
    seed: int = 0,
) -> dict[str, float]:
    """Average of each per-impression metric."""
    return {m: float(v.mean()) for m, v in per_impression(scores, labels, offsets, seed).items()}
