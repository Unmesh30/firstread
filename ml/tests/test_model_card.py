"""The model card's numbers must match the committed reports, so the card can't drift."""

import json
from typing import Any

import pytest
import yaml

from firstread_ml.eval import gate_real
from firstread_ml.mind.manifest import LICENSE_URL
from firstread_ml.paths import REPO_ROOT, reports_dir
from firstread_ml.release import HF_REPO

CARD = (REPO_ROOT / "docs" / "model-card.md").read_text()


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((reports_dir() / name).read_text())
    return data


RESULTS = load("results-mind-large-dev.json")
CLASSIFIER = load("classifier-mind-large.json")


def ndcg(model: str, sl: str) -> str:
    return f"{RESULTS['results'][model][sl]['ndcg@10']:.4f}"


def test_front_matter_declares_the_mind_license() -> None:
    meta = yaml.safe_load(CARD.split("---")[1])
    assert meta["license"] == "other"
    assert meta["license_link"] == LICENSE_URL
    assert meta["library_name"] == "onnx"
    assert "MIT" in CARD
    assert "non-commercial" in CARD


@pytest.mark.parametrize(
    "model", ["random", "popularity", "recency", "mean_history", "nrms_lite_s0"]
)
@pytest.mark.parametrize("sl", ["overall", "cold_candidates", "cold_users"])
def test_dev_table_matches_report(model: str, sl: str) -> None:
    assert ndcg(model, sl) in CARD


def test_seed_summary_and_bootstrap_match_report() -> None:
    for sl in ("overall", "cold_candidates", "cold_users"):
        s = RESULTS["nrms_lite"]["summary"][sl]["ndcg@10"]
        assert f"{s['mean']:.4f} ± {s['std']:.4f}" in CARD
    for sl in ("overall", "cold_candidates"):
        b = RESULTS["bootstrap"]["slices"][sl]
        assert f"+{b['gain_mean']:.4f}" in CARD
        assert f"[+{b['ci95_low']:.4f}, +{b['ci95_high']:.4f}]" in CARD
    for sl in ("overall", "cold_candidates", "cold_users"):
        assert f"{RESULTS['slices'][sl]['n_impressions']:,}" in CARD


def test_category_losses_match_report() -> None:
    for cat in ("tv", "weather", "video", "lifestyle"):
        sl = f"category:{cat}"
        assert float(ndcg("nrms_lite_s0", sl)) < float(ndcg("mean_history", sl))
        assert f"| {cat} | {ndcg('nrms_lite_s0', sl)} | {ndcg('mean_history', sl)} |" in CARD


def test_classifier_and_gate_numbers_match() -> None:
    t = CLASSIFIER["test"]
    assert f"accuracy {t['accuracy']:.4f}, macro-F1 {t['macro_f1']:.4f}" in CARD
    gate = json.loads(gate_real.REFERENCE.read_text())["ndcg@10"]
    for sl in ("overall", "cold_candidates"):
        assert f"{gate['onnx_release'][sl]:.4f} vs {gate['mean_history'][sl]:.4f}" in CARD


def test_card_links_the_hub_repo_and_tag() -> None:
    assert f"https://huggingface.co/{HF_REPO}" in CARD
    assert f"https://huggingface.co/{HF_REPO}/tree/v0.1.0" in CARD
