"""Title-only skew study: decision rule, split-table scoring, and a synthetic end-to-end run."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from firstread_core.embedding import Embedder
from firstread_ml.baselines import FitContext, MeanHistoryScorer
from firstread_ml.config import ModelConfig
from firstread_ml.eval import gate_real, skew
from firstread_ml.eval.harness import load_eval_data
from firstread_ml.export.user_encoder import export_user_encoder
from firstread_ml.models.nrms_lite import NRMSLite
from firstread_ml.testing import synthetic


def results(a: float, c: float, base_c: float) -> dict[str, Any]:
    def cell(x: float) -> dict[str, dict[str, float]]:
        return {"overall": {"ndcg@10": x}}

    return {
        "A": {"onnx_release": cell(a), "mean_history": cell(a - 0.02)},
        "B": {"onnx_release": cell((a + c) / 2), "mean_history": cell(base_c)},
        "C": {"onnx_release": cell(c), "mean_history": cell(base_c)},
    }


def test_decision_accepts_small_loss_that_still_beats_the_baseline() -> None:
    d = skew.decide(results(0.42, 0.41, 0.37))
    assert d["title_only_ok"]
    assert d["relative_loss_C"] == pytest.approx(0.0238, abs=1e-4)


@pytest.mark.parametrize(("c", "base_c"), [(0.39, 0.37), (0.41, 0.415)])
def test_decision_rejects_big_loss_or_losing_to_the_baseline(c: float, base_c: float) -> None:
    assert not skew.decide(results(0.42, c, base_c))["title_only_ok"]


def test_split_text_scorer_uses_history_and_candidate_tables(synthetic_root: Path) -> None:
    data = load_eval_data(synthetic.VARIANT, "dev", synthetic_root)
    rng = np.random.default_rng(0)
    hist = rng.normal(size=(data.n_news, 384)).astype(np.float32)
    cand = rng.normal(size=(data.n_news, 384)).astype(np.float32)

    def ctx() -> FitContext:
        return FitContext(train=data.fit, n_news=data.n_news, first_seen=data.first_seen)

    same = skew.SplitTextScorer(MeanHistoryScorer(), hist, hist)
    same.fit(ctx())
    plain = MeanHistoryScorer()
    c = ctx()
    c.emb = hist
    plain.fit(c)
    np.testing.assert_allclose(same.score(data.test), plain.score(data.test), rtol=1e-6)

    split = skew.SplitTextScorer(MeanHistoryScorer(), hist, cand)
    split.fit(ctx())
    assert not np.allclose(split.score(data.test), plain.score(data.test))


def test_skew_study_on_synthetic(
    synthetic_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate_real, "VARIANT", synthetic.VARIANT)
    torch.manual_seed(0)
    model = NRMSLite(ModelConfig(heads=4, query_dim=16), torch.zeros(384))
    onnx_path = export_user_encoder(model, tmp_path / "u.onnx")
    r = skew.run(onnx_path, synthetic_root, Embedder(backend=synthetic.hash_backend))
    assert set(r["results"]) == {"A", "B", "C"}
    assert set(r["results"]["A"]) == {"mean_history", "onnx_release"}
    # Synthetic abstracts carry no topic signal, so dropping them barely matters.
    assert abs(r["decision"]["relative_loss_C"]) < 0.2
    js, md = skew.write_report(r, tmp_path)
    assert "Decision" in md.read_text()
    assert js.is_file()
