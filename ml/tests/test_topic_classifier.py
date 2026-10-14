import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from firstread_core.embedding import Embedder
from firstread_core.topics import TOPIC_IDS
from firstread_ml.models import topic_classifier as tc
from firstread_ml.testing import synthetic

SYNTHETIC_MAP = """
version: 1
categories:
  topic0: {default: us}
  topic1: {default: politics}
  topic2: {default: sports}
  topic3: {subcategories: {sub3: health}}
  topic4: {default: exclude}
  topic5: {default: world}
"""


@pytest.fixture
def map_path(tmp_path: Path) -> Path:
    p = tmp_path / "topic_map.yaml"
    p.write_text(SYNTHETIC_MAP)
    return p


def test_real_topic_map_loads_and_uses_only_app_topics() -> None:
    m = tc.load_topic_map()
    assert len(m.sha256) == 64
    assert m.topic("news", "newsus") == "us"
    assert m.topic("news", "newsworld") == "world"
    assert m.topic("finance", "markets") == "business"
    assert m.topic("finance", "finance-technology") == "tech_science"
    assert m.topic("video", "news") == tc.EXCLUDE
    # Same subcategory name, different category, different topic.
    assert m.topic("video", "sports") == "sports"
    assert m.topic("health", "recipes") == "food_drink"
    targets = {
        t
        for e in m.categories.values()
        for t in [*(e.get("subcategories") or {}).values(), e.get("default")]
        if t
    }
    assert targets == {*TOPIC_IDS, tc.EXCLUDE}  # every topic has training data


def test_unmapped_pairs_fail_loudly(map_path: Path) -> None:
    m = tc.load_topic_map(map_path)
    with pytest.raises(tc.TopicMapError, match="no default"):
        m.topic("topic3", "other")
    with pytest.raises(tc.TopicMapError, match="not in the topic map"):
        m.topic("crypto", "bitcoin")
    news = pl.DataFrame({"category": ["topic0", "topic9"], "subcategory": ["sub0", "sub9"]})
    with pytest.raises(tc.TopicMapError):
        tc.map_topics(news, m)


def test_unknown_topic_in_map_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\ncategories:\n  news: {default: crypto}\n")
    with pytest.raises(tc.TopicMapError, match="crypto"):
        tc.load_topic_map(p)


def test_confidence_threshold_sits_above_the_last_inaccurate_bin() -> None:
    rng = np.random.default_rng(0)
    top = rng.uniform(0.2, 1.0, 5000)
    # The rest of the mass is spread over 4 classes, so `top` is always the max.
    proba = np.column_stack([top, *[(1 - top) / 4] * 4])
    correct = rng.random(5000) < np.where(top < 0.4, 0.3, 0.9)  # bad below 0.4
    threshold, bins = tc.confidence_threshold(proba, correct)
    assert threshold == pytest.approx(0.4)
    assert sum(b["n"] for b in bins) == 5000


def test_confidence_threshold_is_zero_when_every_bin_is_accurate() -> None:
    proba = np.array([[0.9, 0.1]] * 100)
    assert tc.confidence_threshold(proba, np.ones(100, dtype=bool))[0] == 0.0


def test_gate_flags_weak_topics_and_large_title_gap() -> None:
    def mode(macro: float, f1s: list[float]) -> dict:  # type: ignore[type-arg]
        return {
            "macro_f1": macro,
            "per_class": [{"label": f"t{i}", "f1": f} for i, f in enumerate(f1s)],
        }

    assert tc.gate({"title_abstract": mode(0.7, [0.8]), "title_only": mode(0.65, [0.7])}) == []
    failures = tc.gate({"title_abstract": mode(0.8, [0.9]), "title_only": mode(0.6, [0.2])})
    assert len(failures) == 2


def test_topic_classifier_on_synthetic(
    synthetic_root: Path, tmp_path: Path, map_path: Path
) -> None:
    r = tc.run(
        synthetic.VARIANT,
        data_root=synthetic_root,
        out_dir=tmp_path,
        topic_map_path=map_path,
        embedder=Embedder(backend=synthetic.hash_backend),
    )
    # Labels follow TOPIC_IDS order; topic4 was excluded.
    assert r["labels"] == ["us", "world", "politics", "health", "sports"]
    assert r["n_excluded"] > 0
    assert r["n_train_rows"] == 2 * r["n_train_articles"]
    for m in tc.MODES:
        assert r["test"][m]["accuracy"] > 0.9  # the topic word dominates the hash embedding
    assert 0.0 <= r["threshold"] < 1.0
    assert (tmp_path / "topic-classifier-mind-synthetic.md").is_file()
    model_dir = synthetic_root / "models" / synthetic.VARIANT / "topic_classifier"
    assert json.loads((model_dir / "labels.json").read_text()) == r["labels"]
    policy = json.loads((model_dir / "policy.json").read_text())
    assert policy == {"threshold": r["threshold"], "topic_map_sha256": r["topic_map_sha256"]}
