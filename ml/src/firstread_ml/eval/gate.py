"""Eval-regression gate on synthetic data (SPEC section 13), runnable in CI.

CI cannot hold MIND (no redistribution), so the gate runs the full pipeline
(prepare -> embed -> fit -> score -> metrics -> slices) on the seeded synthetic
dataset with a hash embedder and fails if any model's nDCG@10 drops by more
than 1% (relative) versus the committed expected values. It catches
regressions in code (metrics, slices, features, training), not in MIND quality;
the real-data gate is ``make eval-gate`` locally (added with the model in 1b/1c).
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from firstread_core.embedding import Embedder
from firstread_ml.eval import harness
from firstread_ml.paths import REPO_ROOT
from firstread_ml.testing import synthetic

EXPECTED = REPO_ROOT / "ml" / "tests" / "data" / "eval_gate_expected.json"
MAX_RELATIVE_DROP = 0.01
GATE_MODELS = ["random", "popularity", "recency", "mean_history"]
GATED_SLICES = ("overall", "cold_candidates", "cold_users")


def run_synthetic(root: Path, models: list[str] | None = None) -> dict[str, Any]:
    raw = root / "mind" / "raw"
    synthetic.generate(raw, seed=0)
    return harness.run(
        synthetic.VARIANT,
        "dev",
        models or GATE_MODELS,
        embedder=Embedder(backend=synthetic.hash_backend),
        data_root=root,
        raw_dir=raw,
    )


def current_values(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        model: {sl: round(res[sl]["ndcg@10"], 6) for sl in GATED_SLICES}
        for model, res in report["results"].items()
    }


def check(values: dict[str, dict[str, float]], expected: dict[str, Any]) -> list[str]:
    """Return a list of failures (empty = pass)."""
    failures = []
    for model, slices in expected["ndcg@10"].items():
        for sl, exp in slices.items():
            got = values.get(model, {}).get(sl)
            if got is None:
                failures.append(f"{model}/{sl}: missing")
            elif got < exp * (1 - MAX_RELATIVE_DROP):
                failures.append(f"{model}/{sl}: nDCG@10 {got:.6f} < expected {exp:.6f} - 1%")
    return failures


def main(update: bool = False) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        values = current_values(run_synthetic(Path(tmp)))
    if update:
        EXPECTED.write_text(json.dumps({"ndcg@10": values}, indent=1, sort_keys=True) + "\n")
        print(f"updated {EXPECTED}")
        return 0
    failures = check(values, json.loads(EXPECTED.read_text()))
    for f in failures:
        print(f"GATE FAIL {f}")
    if not failures:
        print("GATE PASS: no nDCG@10 drop > 1% on the synthetic eval set")
    return 1 if failures else 0
