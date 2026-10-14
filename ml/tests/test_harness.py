import json
from pathlib import Path

import numpy as np
import pytest

from firstread_core.embedding import Embedder
from firstread_ml.baselines import BASELINES
from firstread_ml.eval import gate, harness, report
from firstread_ml.mind.load import load_impressions
from firstread_ml.paths import processed_dir
from firstread_ml.testing import synthetic


def test_synthetic_prepare_shapes(synthetic_root: Path) -> None:
    proc = processed_dir(synthetic.VARIANT, synthetic_root)
    train = load_impressions(synthetic.VARIANT, "train", proc)
    dev = load_impressions(synthetic.VARIANT, "dev", proc)
    assert train.n == 2500
    assert dev.n == 800
    assert (train.cand >= 0).all()
    # Every impression has at least one click and one non-click.
    pos = np.bincount(dev.cand_imp, weights=dev.label, minlength=dev.n)
    assert ((pos > 0) & (pos < dev.cand_len)).all()


@pytest.fixture(scope="module")
def dev_report(synthetic_root: Path) -> dict:  # type: ignore[type-arg]
    return harness.run(
        synthetic.VARIANT,
        "dev",
        list(BASELINES),
        embedder=Embedder(backend=synthetic.hash_backend),
        data_root=synthetic_root,
        raw_dir=synthetic_root / "mind" / "raw",
    )


def test_harness_reports_all_models_and_slices(dev_report: dict) -> None:  # type: ignore[type-arg]
    res = dev_report["results"]
    assert set(res) == set(BASELINES)
    for sl in ("overall", "cold_candidates", "cold_users"):
        assert dev_report["slices"][sl]["n_impressions"] > 0
        for m in BASELINES:
            assert 0.0 <= res[m][sl]["auc"] <= 1.0
    assert dev_report["slices"]["cold_users"]["empty_history"] > 0


def test_planted_structure_is_recovered(dev_report: dict) -> None:  # type: ignore[type-arg]
    res = dev_report["results"]
    # Random is ~0.5 AUC; content beats random; popularity is uninformative on
    # cold candidates (all zero clicks -> pure ties).
    assert abs(res["random"]["overall"]["auc"] - 0.5) < 0.05
    assert res["mean_history"]["overall"]["ndcg@10"] > res["random"]["overall"]["ndcg@10"]
    assert res["popularity"]["cold_candidates"]["auc"] == pytest.approx(0.5)


def test_holdout_uses_only_train(synthetic_root: Path) -> None:
    rep = harness.run(
        synthetic.VARIANT,
        "holdout",
        ["popularity"],
        data_root=synthetic_root,
        raw_dir=synthetic_root / "mind" / "raw",
    )
    assert rep["meta"]["fit_impressions"] + rep["meta"]["test_impressions"] == 2500


def test_unknown_split_and_model(synthetic_root: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        harness.load_eval_data(synthetic.VARIANT, "test", synthetic_root)
    with pytest.raises(ValueError, match="unknown models"):
        harness.run(synthetic.VARIANT, "dev", ["nope"], data_root=synthetic_root)


def test_report_writes_markdown_and_json(dev_report: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    j, m = report.write(dev_report, tmp_path, list(BASELINES))
    assert json.loads(j.read_text())["meta"]["variant"] == synthetic.VARIANT
    md = m.read_text()
    assert "## Cold-start candidates" in md
    assert "Bar to beat [overall]" in md
    lines = report.acceptance(dev_report, "mean_history", ["random", "popularity"])
    assert lines[0].startswith("ACCEPT PASS [overall] mean_history")


def test_eval_gate_passes_on_committed_expectations(tmp_path: Path) -> None:
    values = gate.current_values(gate.run_synthetic(tmp_path))
    assert gate.check(values, json.loads(gate.EXPECTED.read_text())) == []


def test_eval_gate_detects_a_drop() -> None:
    expected = {"ndcg@10": {"m": {"overall": 0.5}}}
    assert gate.check({"m": {"overall": 0.4951}}, expected) == []
    assert gate.check({"m": {"overall": 0.4949}}, expected)
    assert gate.check({}, expected) == ["m/overall: missing"]


def test_eda_report_has_aggregates_only(synthetic_root: Path, tmp_path: Path) -> None:
    from firstread_ml.eda import run

    path = run(synthetic.VARIANT, data_root=synthetic_root, out_dir=tmp_path)
    md = path.read_text()
    assert "## Cold-start candidates" in md
    assert "topic" in md  # category names are aggregates
    # No article text or ids leak into the report.
    assert "Synthetic abstract" not in md
    assert " N1" not in md
    assert len(list((tmp_path / "figures").glob("*.svg"))) == 3
