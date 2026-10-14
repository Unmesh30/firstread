import numpy as np
import pytest

from firstread_core.topics import TOPIC_IDS, TOPICS, resolve_topic

LABELS = list(TOPIC_IDS)


def probs(**weights: float) -> np.ndarray:
    """A probability vector over LABELS with the given topics' weights."""
    p = np.full(len(LABELS), 1e-3)
    for topic, w in weights.items():
        p[LABELS.index(topic)] = w
    out: np.ndarray = p / p.sum()
    return out


def test_thirteen_topics_with_display_names() -> None:
    assert len(TOPIC_IDS) == 13
    assert all(TOPICS[t] for t in TOPIC_IDS)
    assert {"us", "world", "politics", "tech_science"} <= set(TOPIC_IDS)


def test_confident_model_wins_over_hint() -> None:
    p = probs(sports=0.9, health=0.05)
    d = resolve_topic(p, LABELS, "us", threshold=0.5, hint="health")
    assert (d.topic, d.source) == ("sports", "model")
    assert d.confidence == pytest.approx(p.max())


def test_low_confidence_hint_in_top_k_wins() -> None:
    p = probs(business=0.35, tech_science=0.30, us=0.2)
    d = resolve_topic(p, LABELS, "us", threshold=0.5, hint="tech_science")
    assert (d.topic, d.source) == ("tech_science", "feed_hint")


def test_low_confidence_hint_outside_top_k_is_ignored() -> None:
    p = probs(business=0.35, tech_science=0.30, us=0.2, politics=0.1)
    d = resolve_topic(p, LABELS, "us", threshold=0.5, hint="weather")
    assert (d.topic, d.source) == ("business", "model")


def test_no_hint_keeps_model() -> None:
    d = resolve_topic(probs(health=0.3, lifestyle=0.29), LABELS, "us", threshold=0.5)
    assert (d.topic, d.source) == ("health", "model")


@pytest.mark.parametrize("topic", ["us", "politics"])
def test_geo_rule_moves_intl_domestic_topics_to_world(topic: str) -> None:
    d = resolve_topic(probs(**{topic: 0.9}), LABELS, "intl", threshold=0.5)
    assert (d.topic, d.source) == ("world", "geo_rule")


def test_geo_rule_applies_after_the_tie_breaker() -> None:
    p = probs(health=0.35, us=0.3, lifestyle=0.3)
    d = resolve_topic(p, LABELS, "intl", threshold=0.5, hint="us")
    assert (d.topic, d.source) == ("world", "geo_rule")


def test_geo_rule_leaves_global_subjects_and_us_items_alone() -> None:
    assert resolve_topic(probs(sports=0.9), LABELS, "intl", 0.5).topic == "sports"
    assert resolve_topic(probs(politics=0.9), LABELS, "us", 0.5).topic == "politics"


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        resolve_topic([0.5, 0.5], LABELS, "us", 0.5)
    with pytest.raises(ValueError, match="hint"):
        resolve_topic(probs(us=0.9), LABELS, "us", 0.5, hint="crypto")
