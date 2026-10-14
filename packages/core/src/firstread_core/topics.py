"""App topics and the one rule that turns classifier output into a topic.

The topic classifier (released from v0.2.0) predicts one of :data:`TOPIC_IDS`
directly; its training labels come from MIND (category, subcategory) pairs via
``ml/configs/topic_map.yaml``. Ingestion, the API and offline analysis all call
:func:`resolve_topic`, so the low-confidence tie-breaker and the geo rule can't
drift apart (ADR 0006).

Meaning of the topics:

* ``us`` and ``politics`` are **US domestic** news;
* ``world`` is news about places outside the US;
* the other ten are global subjects.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Geo = Literal["us", "intl"]
CategorySource = Literal["model", "feed_hint", "geo_rule"]

TOPICS: dict[str, str] = {
    "us": "U.S.",
    "world": "World",
    "politics": "Politics",
    "business": "Business",
    "tech_science": "Tech & Science",
    "health": "Health",
    "sports": "Sports",
    "entertainment": "Entertainment",
    "lifestyle": "Lifestyle",
    "food_drink": "Food & Drink",
    "travel": "Travel",
    "autos": "Autos",
    "weather": "Weather",
}
TOPIC_IDS: tuple[str, ...] = tuple(TOPICS)
US_DOMESTIC = frozenset({"us", "politics"})
# A feed hint only overrides the model when it is among the model's top-k topics.
HINT_TOP_K = 3


@dataclass(frozen=True)
class TopicDecision:
    topic: str
    confidence: float  # the model's probability for its own top-1 topic
    source: CategorySource


def resolve_topic(
    probs: NDArray[Any] | Sequence[float],
    labels: Sequence[str],
    geo: Geo,
    threshold: float,
    hint: str | None = None,
) -> TopicDecision:
    """Pick the topic for one article.

    1. The model's top-1 topic.
    2. Tie-breaker: if its probability is below ``threshold`` and the feed's
       ``hint`` is among the model's top ``HINT_TOP_K``, use the hint.
    3. Geo rule: an ``intl`` item labelled ``us`` or ``politics`` becomes
       ``world`` (MIND's US-domestic labels would otherwise file a Nairobi crime
       story under ``us``).
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (len(labels),):
        raise ValueError(f"expected {len(labels)} probabilities, got shape {p.shape}")
    order = np.argsort(-p, kind="stable")
    topic, confidence = labels[order[0]], float(p[order[0]])
    source: CategorySource = "model"
    if hint is not None and hint not in TOPICS:
        raise ValueError(f"unknown topic hint {hint!r}")
    top_k = {labels[i] for i in order[:HINT_TOP_K]}
    if hint is not None and confidence < threshold and hint != topic and hint in top_k:
        topic, source = hint, "feed_hint"
    if geo == "intl" and topic in US_DOMESTIC:
        topic, source = "world", "geo_rule"
    return TopicDecision(topic=topic, confidence=confidence, source=source)
