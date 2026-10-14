"""Real-data eval gate: the released ONNX user encoder on a fixed MIND-small dev subset.

Runs locally only (MIND can't live in CI; ADR 0003) and ``release build``
requires it. Two rules, both on nDCG@10:

1. ``onnx_release`` beats ``mean_history`` on the overall and cold-candidate slices;
2. no model drops by more than 1% (relative) versus the committed reference
   ``ml/reports/eval-gate-real.json`` (``gate.check``).

The reference is rewritten only by ``eval-gate --real --update``, deliberately,
in a PR. A normal gate run writes its result under gitignored ``data/``, so a
release build never modifies a tracked file.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import Embedder
from firstread_ml.baselines import FitContext, MeanHistoryScorer
from firstread_ml.embed_cache import load_or_build
from firstread_ml.eval import gate
from firstread_ml.eval.harness import evaluate, load_eval_data
from firstread_ml.mind.load import load_news, prepare
from firstread_ml.mind.manifest import HF_REVISION
from firstread_ml.models.onnx_scorer import ORTUserEncoderScorer
from firstread_ml.paths import REPO_ROOT, mind_raw_dir, processed_dir

REFERENCE = REPO_ROOT / "ml" / "reports" / "eval-gate-real.json"
VARIANT = "small"
SUBSET_SIZE = 5_000
SUBSET_SEED = 0
MODEL = "onnx_release"
BASELINE = "mean_history"
MUST_BEAT_ON = ("overall", "cold_candidates")


def subset_rows(n: int, size: int = SUBSET_SIZE, seed: int = SUBSET_SEED) -> NDArray[np.int64]:
    """A fixed, sorted sample of impression rows (all rows if there are fewer)."""
    if n <= size:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False)).astype(np.int64)


def run(
    onnx_path: Path,
    variant: str = VARIANT,
    data_root: Path | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Evaluate ``mean_history`` and the ONNX user encoder on the dev subset."""
    proc = prepare(
        variant, raw_dir=mind_raw_dir(data_root), out_dir=processed_dir(variant, data_root)
    )
    data = load_eval_data(variant, "dev", data_root)
    rows = subset_rows(data.test.n)
    data.test = data.test.take(rows)
    emb = load_or_build(variant, load_news(variant, proc), embedder, data_root)
    ctx = FitContext(
        train=data.fit, n_news=data.n_news, first_seen=data.first_seen, emb=emb, tag="gate-real"
    )
    print(f"eval gate (real): {MODEL} vs {BASELINE}, {len(rows):,} MIND-{variant} dev impressions")
    out = evaluate([MeanHistoryScorer(), ORTUserEncoderScorer(onnx_path)], ctx, data)
    meta = {
        "variant": variant,
        "split": "dev",
        "subset_size": len(rows),
        "subset_seed": SUBSET_SEED,
        "dataset_revision": HF_REVISION if variant in ("small", "large") else variant,
    }
    return {"meta": meta, **out}


def check(values: dict[str, dict[str, float]], reference: dict[str, Any] | None) -> list[str]:
    """Failures (empty = pass). ``reference=None`` skips the regression rule."""
    failures = []
    for sl in MUST_BEAT_ON:
        got, base = values.get(MODEL, {}).get(sl), values.get(BASELINE, {}).get(sl)
        if got is None or base is None:
            failures.append(f"{MODEL}/{sl}: missing")
        elif got <= base:
            failures.append(f"{MODEL}/{sl}: nDCG@10 {got:.6f} does not beat {BASELINE} {base:.6f}")
    if reference is not None:
        failures += gate.check(values, reference)
    return failures


def main(
    onnx_path: Path,
    out: Path,
    update: bool = False,
    reference_path: Path = REFERENCE,
    variant: str = VARIANT,
    data_root: Path | None = None,
    embedder: Embedder | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the gate, write the result to ``out``; return (exit code, result)."""
    report = run(onnx_path, variant, data_root, embedder)
    values = gate.current_values(report)
    reference = (
        json.loads(reference_path.read_text()) if reference_path.is_file() and not update else None
    )
    failures = check(values, reference)
    if reference is None and not update:
        failures.append(f"no reference at {reference_path} (run once with --update)")
    result = {
        "ndcg@10": values,
        "meta": report["meta"],
        "slices": {k: v["n_impressions"] for k, v in report["slices"].items()},
        "reference": str(reference_path.relative_to(REPO_ROOT))
        if reference_path.is_relative_to(REPO_ROOT)
        else str(reference_path),
        "compared_to_reference": reference is not None,
        "failures": failures,
        "passed": not failures,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for sl in MUST_BEAT_ON:
        print(f"  {sl}: {MODEL} {values[MODEL][sl]:.4f} vs {BASELINE} {values[BASELINE][sl]:.4f}")
    for f in failures:
        print(f"GATE FAIL {f}")
    if failures:
        return 1, result
    if update:
        keep = {"ndcg@10": values, "meta": report["meta"]}
        reference_path.write_text(json.dumps(keep, indent=1, sort_keys=True) + "\n")
        print(f"GATE PASS; wrote reference {reference_path}")
    else:
        print("GATE PASS: beats mean_history, no nDCG@10 drop > 1% vs the reference")
    return 0, result
