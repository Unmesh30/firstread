import numpy as np
import pytest

from firstread_ml.eval import report
from firstread_ml.eval.stats import paired_bootstrap, seed_summary

METRIC_KEYS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def fake_results(nrms: dict[str, tuple[float, float]]) -> dict:  # type: ignore[type-arg]
    def row(overall: float, cold: float) -> dict:  # type: ignore[type-arg]
        return {
            "overall": dict.fromkeys(METRIC_KEYS, overall),
            "cold_candidates": dict.fromkeys(METRIC_KEYS, cold),
        }

    res = {
        "random": row(0.28, 0.32),
        "popularity": row(0.30, 0.32),
        "recency": row(0.29, 0.35),
        "mean_history": row(0.39, 0.43),
    }
    res.update({m: row(*v) for m, v in nrms.items()})
    return {"results": res}


def test_acceptance_requires_every_seed_to_beat_best_baseline() -> None:
    ok = fake_results({"nrms_lite_s0": (0.40, 0.45), "nrms_lite_s1": (0.41, 0.44)})
    assert all(line.startswith("ACCEPT PASS") for line in report.acceptance(ok))
    one_bad = fake_results({"nrms_lite_s0": (0.40, 0.45), "nrms_lite_s1": (0.385, 0.44)})
    lines = report.acceptance(one_bad)
    assert lines[0].startswith("ACCEPT FAIL [overall]")  # seed 1 is below mean_history
    assert lines[1].startswith("ACCEPT PASS [cold_candidates]")
    assert "s0 0.4000, s1 0.3850" in lines[0]


def test_seed_summary_mean_and_sample_std() -> None:
    res = fake_results({"nrms_lite_s0": (0.40, 0.45), "nrms_lite_s1": (0.42, 0.47)})["results"]
    s = seed_summary(res, ["nrms_lite_s0", "nrms_lite_s1"])
    assert s["overall"]["ndcg@10"]["mean"] == pytest.approx(0.41)
    assert s["overall"]["ndcg@10"]["std"] == pytest.approx(np.std([0.40, 0.42], ddof=1))
    single = seed_summary(res, ["nrms_lite_s0"])
    assert single["overall"]["ndcg@10"]["std"] == 0.0


def test_paired_bootstrap_covers_true_gain_and_is_paired() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    difficulty = rng.uniform(0.0, 1.0, n)  # shared per-impression difficulty
    base = difficulty
    seeds = [difficulty + 0.02 + rng.normal(0, 0.01, n) for _ in range(3)]
    r = paired_bootstrap(seeds, base, resamples=500)
    assert r["ci95_low"] < 0.02 < r["ci95_high"]
    # Pairing cancels the (large) difficulty variance: the CI is narrow.
    assert r["ci95_high"] - r["ci95_low"] < 0.002
    assert r["share_resamples_gain_le_0"] == 0.0
    assert len(r["per_seed_gain"]) == 3
    assert r == paired_bootstrap(seeds, base, resamples=500)  # deterministic


def test_paired_bootstrap_straddles_zero_without_real_gain() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, 4000)
    r = paired_bootstrap([base + rng.normal(0, 0.05, 4000)], base, resamples=500)
    assert r["ci95_low"] < 0 < r["ci95_high"]
