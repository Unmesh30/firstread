import json
from pathlib import Path
from typing import Any

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
    j, m = report.write(dev_report, tmp_path)
    assert json.loads(j.read_text())["meta"]["variant"] == synthetic.VARIANT
    md = m.read_text()
    assert "## Cold-start candidates" in md
    assert "Bar to beat [overall]" in md  # no NRMS-lite in this report


def test_report_refuses_missing_baselines(dev_report: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    partial = {**dev_report, "results": {"mean_history": dev_report["results"]["mean_history"]}}
    with pytest.raises(report.IncompleteReportError, match="random"):
        report.write(partial, tmp_path)
    assert not list(tmp_path.iterdir())


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


def test_nrms_lite_trains_scores_and_reuses_checkpoint(synthetic_root: Path) -> None:
    from firstread_ml.paths import REPO_ROOT

    kwargs: dict[str, Any] = {
        "embedder": Embedder(backend=synthetic.hash_backend),
        "data_root": synthetic_root,
        "raw_dir": synthetic_root / "mind" / "raw",
        "nrms_config": REPO_ROOT / "ml" / "configs" / "nrms_lite_ci.yaml",
    }
    first = harness.run(synthetic.VARIANT, "holdout", ["mean_history", "nrms_lite"], **kwargs)
    res = first["results"]["nrms_lite_s0"]["overall"]
    assert 0.0 <= res["auc"] <= 1.0
    ckpts = list((synthetic_root / "models" / synthetic.VARIANT / "holdout").glob("nrms_lite/*"))
    assert len(ckpts) == 1
    second = harness.run(synthetic.VARIANT, "holdout", ["nrms_lite"], **kwargs)
    assert second["results"]["nrms_lite_s0"] == first["results"]["nrms_lite_s0"]  # reused
    assert "bootstrap" in first  # mean_history + NRMS-lite present -> paired CI
    assert set(first["bootstrap"]["slices"]) == {"overall", "cold_candidates"}


def test_category_classifier_on_synthetic(
    synthetic_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from firstread_ml import embed_cache
    from firstread_ml.models import classifier

    # Use the hash embedder (no model download) for any cache miss.
    real = embed_cache.load_or_build
    monkeypatch.setattr(
        classifier,
        "load_or_build",
        lambda v, news, emb, root: real(v, news, Embedder(backend=synthetic.hash_backend), root),
    )
    r = classifier.run(synthetic.VARIANT, data_root=synthetic_root, out_dir=tmp_path)
    assert r["labels"] == [f"topic{i}" for i in range(synthetic.N_TOPICS)]
    assert r["n_test_dev_only"] > 0  # articles that appear only in dev
    assert r["test"]["accuracy"] > 0.9  # the topic word dominates the synthetic embedding
    assert 0.0 <= r["test"]["macro_f1"] <= 1.0
    assert (tmp_path / "classifier-mind-synthetic.md").is_file()
    assert (synthetic_root / "models" / synthetic.VARIANT / "classifier" / "model.joblib").is_file()
