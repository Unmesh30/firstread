"""ONNX export parity on tiny random models (no MIND needed; runs in CI)."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from firstread_core.embedding import EMBED_DIM
from firstread_core.features import HISTORY_LEN
from firstread_ml.config import ModelConfig
from firstread_ml.export import parity
from firstread_ml.export.classifier import export_classifier
from firstread_ml.export.user_encoder import export_user_encoder
from firstread_ml.models.nrms_lite import NRMSLite


@pytest.fixture(scope="module")
def model() -> NRMSLite:
    torch.manual_seed(0)
    m = NRMSLite(ModelConfig(heads=4, query_dim=16), torch.randn(EMBED_DIM))
    with torch.no_grad():
        m.alpha.fill_(0.5)  # non-zero, so the attention path contributes to the output
    return m.eval()


@pytest.fixture(scope="module")
def encoder_onnx(model: NRMSLite, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_user_encoder(model, tmp_path_factory.mktemp("onnx") / "user_encoder.onnx")


def test_user_encoder_parity_on_empty_partial_and_full_histories(
    model: NRMSLite, encoder_onnx: Path
) -> None:
    result = parity.check_user_encoder(model, encoder_onnx)
    assert set(result.max_abs_diff) >= {"len0", "len1", "len7", "len49", "len50", "mixed"}
    assert result.ok, result.as_dict()


def test_attention_path_is_exercised(model: NRMSLite) -> None:
    hist, mask = parity.default_cases()["len7"]
    with torch.no_grad():
        u = model(torch.from_numpy(hist), torch.from_numpy(mask))
        model.alpha.fill_(0.0)
        base = model(torch.from_numpy(hist), torch.from_numpy(mask))
        model.alpha.fill_(0.5)
    assert (u - base).abs().max() > 1e-2


def test_empty_history_is_the_population_vector(model: NRMSLite, encoder_onnx: Path) -> None:
    hist, mask = parity.default_cases()["len0"]
    got = parity.session(encoder_onnx).run(None, {"hist": hist, "mask": mask})[0]
    np.testing.assert_allclose(got[0], model.population.numpy(), atol=1e-6)


def test_user_encoder_signature(encoder_onnx: Path) -> None:
    sess = parity.session(encoder_onnx)
    ins = {i.name: (i.type, i.shape) for i in sess.get_inputs()}
    assert ins == {
        "hist": ("tensor(float)", ["batch", HISTORY_LEN, EMBED_DIM]),
        "mask": ("tensor(bool)", ["batch", HISTORY_LEN]),
    }
    (out,) = sess.get_outputs()
    assert (out.name, out.shape) == ("user_vec", ["batch", EMBED_DIM])
    props = {p.key: p.value for p in onnx.load(str(encoder_onnx)).metadata_props}
    assert props["history_len"] == str(HISTORY_LEN)
    assert float(props["score_scale"]) == pytest.approx(10.0, rel=1e-5)


def test_dynamic_batch_runs_large_batches(encoder_onnx: Path) -> None:
    hist, mask = parity.histories([3] * 100, np.random.default_rng(1))
    out = parity.session(encoder_onnx).run(None, {"hist": hist, "mask": mask})[0]
    assert out.shape == (100, EMBED_DIM)
    assert np.isfinite(out).all()


def test_classifier_parity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, EMBED_DIM))
    y = rng.integers(0, 5, 600)
    x = (centers[y] + rng.standard_normal((600, EMBED_DIM))).astype(np.float32)
    clf = LogisticRegression(C=30.0, max_iter=2000).fit(x[:500], y[:500])
    path = export_classifier(clf, tmp_path / "clf.onnx")
    result = parity.check_classifier(clf, path, x[500:])
    assert result["ok"], result
    sess = parity.session(path)
    assert [(i.name, i.type, i.shape) for i in sess.get_inputs()] == [
        ("embedding", "tensor(float)", [None, EMBED_DIM])
    ]
    assert [(o.name, o.type) for o in sess.get_outputs()] == [
        ("label", "tensor(int64)"),
        ("probabilities", "tensor(float)"),
    ]
