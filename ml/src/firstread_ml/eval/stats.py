"""Seed summaries and paired bootstrap confidence intervals.

* Seed summary: NRMS-lite is trained once per seed; we report each seed and the
  mean ± sample standard deviation across seeds.
* Paired bootstrap: per test impression, gain_i = mean over seeds of NRMS-lite's
  nDCG@10 minus the baseline's nDCG@10 on the *same* impression (paired, so
  impression difficulty cancels). Resample impressions with replacement 1,000
  times and take the 2.5th and 97.5th percentiles of the mean gain.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_ml.eval.metrics import METRICS

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 12345


def seed_summary(
    results: dict[str, dict[str, dict[str, float]]], seed_models: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """``{slice: {metric: {"mean", "std", "min", "max"}}}`` over the seed models."""
    slices = set.intersection(*(set(results[m]) for m in seed_models))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for sl in sorted(slices):
        out[sl] = {}
        for metric in METRICS:
            vals = np.array([results[m][sl][metric] for m in seed_models])
            out[sl][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
    return out


def paired_bootstrap(
    candidate: list[NDArray[np.float64]],
    baseline: NDArray[np.float64],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """CI for mean(candidate - baseline); ``candidate`` holds one array per seed."""
    gain = np.mean(np.stack(candidate), axis=0) - baseline
    rng = np.random.default_rng(seed)
    n = len(gain)
    means = np.empty(resamples)
    for b in range(resamples):
        means[b] = gain[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "gain_mean": float(gain.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "share_resamples_gain_le_0": float((means <= 0).mean()),
        "per_seed_gain": [float((c - baseline).mean()) for c in candidate],
        "n_impressions": int(n),
        "resamples": resamples,
        "bootstrap_seed": seed,
    }
