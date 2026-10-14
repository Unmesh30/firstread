import math
from pathlib import Path

import numpy as np
import pydantic
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from firstread_core.embedding import EMBED_DIM
from firstread_core.features import mean_history_vector
from firstread_ml.config import ModelConfig, NRMSLiteConfig, load_config
from firstread_ml.models.nrms_lite import MAX_SCALE, MIN_SCALE, NRMSLite
from firstread_ml.models.train import build_samples, sample_negatives
from firstread_ml.paths import REPO_ROOT
from firstread_ml.testing.builders import make_impressions

L = 6


def make_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows: empty history, one click, partial, full."""
    g = torch.Generator().manual_seed(seed)
    hist = F.normalize(torch.randn(4, L, EMBED_DIM, generator=g), dim=-1)
    lengths = [0, 1, 3, L]
    mask = torch.zeros(4, L, dtype=torch.bool)
    for i, n in enumerate(lengths):
        if n:
            mask[i, L - n :] = True
    return hist * mask.unsqueeze(-1), mask


@pytest.fixture
def model() -> NRMSLite:
    torch.manual_seed(0)
    pop = torch.randn(EMBED_DIM)
    return NRMSLite(ModelConfig(heads=4, query_dim=16), pop)


def test_untrained_model_is_the_mean_history_baseline(model: NRMSLite) -> None:
    hist, mask = make_batch()
    model.eval()
    out = model(hist, mask).detach().numpy()
    pop = model.population.numpy()
    expected = mean_history_vector(hist.numpy(), mask.numpy(), fallback=pop)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_empty_history_returns_population_exactly(model: NRMSLite) -> None:
    hist, mask = make_batch()
    for mode in (model.train, model.eval):
        mode()
        with torch.no_grad():
            model.alpha.fill_(0.7)  # make the model branch matter
        out = model(hist, mask)
        assert torch.equal(out[0], model.population)
        assert not torch.equal(out[1], model.population)


def test_mixed_batch_has_finite_loss_and_gradients(model: NRMSLite) -> None:
    hist, mask = make_batch()
    with torch.no_grad():
        model.alpha.fill_(0.5)
    model.train()
    cands = F.normalize(torch.randn(4, 5, EMBED_DIM), dim=-1)
    logits = model.score(model(hist, mask), cands)
    loss = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_all_empty_batch_is_finite(model: NRMSLite) -> None:
    hist = torch.zeros(3, L, EMBED_DIM, requires_grad=True)
    mask = torch.zeros(3, L, dtype=torch.bool)
    out = model(hist, mask)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert hist.grad is not None
    assert torch.isfinite(hist.grad).all()


def test_scale_stays_positive_and_bounded(model: NRMSLite) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=1e6)
    hist, mask = make_batch()
    cands = F.normalize(torch.randn(4, 5, EMBED_DIM), dim=-1)
    for direction in (1.0, -1.0):  # push log_scale hard both ways
        opt.zero_grad()
        (direction * model.score(model(hist, mask), cands).sum()).backward()
        opt.step()
        scale = float(model.scale.detach())
        assert MIN_SCALE * (1 - 1e-6) <= scale <= MAX_SCALE * (1 + 1e-6)  # float32 exp/log
    with torch.no_grad():
        model.log_scale.fill_(1e9)
    assert float(model.scale.detach()) == pytest.approx(MAX_SCALE)
    with torch.no_grad():
        model.log_scale.fill_(-1e9)
    assert float(model.scale.detach()) == pytest.approx(MIN_SCALE)
    assert math.isclose(math.log(MIN_SCALE), 0.0)


def test_heads_must_divide_embedding_dim() -> None:
    with pytest.raises(ValueError, match="heads"):
        NRMSLite(ModelConfig(heads=5), torch.ones(EMBED_DIM))


def test_negatives_come_from_the_same_impression() -> None:
    imps = make_impressions(
        [[], []],
        [[10, 11, 12], [20, 21, 22, 23, 24]],
        [[1, 0, 0], [0, 1, 1, 0, 0]],
    )
    s = build_samples(imps)
    assert s.pos.tolist() == [10, 21, 22]
    negs = sample_negatives(s, np.arange(s.n), k=4, rng=np.random.default_rng(0))
    assert negs.shape == (3, 4)
    assert set(negs[0]) <= {11, 12}  # only 2 negatives: drawn with replacement
    assert set(negs[1:].ravel()) <= {20, 23, 24}


def test_configs_load_and_reject_unknown_keys(tmp_path: Path) -> None:
    for name in ("nrms_lite.yaml", "nrms_lite_ci.yaml"):
        cfg = load_config(REPO_ROOT / "ml" / "configs" / name)
        assert cfg.train.negatives == 4  # SPEC: K = 4
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  hedas: 8\n")
    with pytest.raises(pydantic.ValidationError):
        load_config(bad)
    assert NRMSLiteConfig().model.heads == 8


def test_tuning_overrides_are_validated() -> None:
    from firstread_ml.models.tune import apply_overrides

    cfg = apply_overrides(NRMSLiteConfig(), ["train.lr=1e-3", "model.heads=4", "train.epochs=5"])
    assert cfg.train.lr == pytest.approx(0.001)
    assert (cfg.model.heads, cfg.train.epochs) == (4, 5)
    with pytest.raises(pydantic.ValidationError):
        apply_overrides(NRMSLiteConfig(), ["train.epochs=0"])
    with pytest.raises(ValueError, match="override"):
        apply_overrides(NRMSLiteConfig(), ["lr=0.1"])
