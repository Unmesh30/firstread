"""Human-readable "why this?" reasons, derived from the blend's actual components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from firstread_core.topics import TOPICS

ReasonCode = Literal[
    "similar_to_read", "topic_match", "followed_topic", "trending", "new", "diversity"
]
MAX_REASONS = 2
NEW_HOURS = 1.0


@dataclass(frozen=True)
class Reason:
    code: ReasonCode
    text: str


@dataclass(frozen=True)
class ReasonInputs:
    model: float  # the model component's contribution to the final score
    fresh: float  # w_fresh · freshness
    trend: float  # w_trend · trend feature
    age_hours: float
    outlets: int  # distinct outlets covering the story in 24 h (0 if not trending)
    topic: str
    followed: Sequence[str] = field(default_factory=tuple)
    topic_weight: float = 0.0  # share of the user vector that came from chosen topics
    nearest_read: str | None = None  # title of the most similar history item
    diversity_pick: bool = False


def reasons(x: ReasonInputs, topic_names: Mapping[str, str] = TOPICS) -> list[Reason]:
    """At most two reasons, ordered by how much each component contributed."""
    topic_name = topic_names.get(x.topic, x.topic)
    model = max(x.model, 0.0)
    cands: list[tuple[float, Reason]] = []
    if x.nearest_read and x.topic_weight < 0.5:
        text = f"Because you read “{x.nearest_read}”"
        cands.append((model * (1.0 - x.topic_weight), Reason("similar_to_read", text)))
    if x.topic_weight >= 0.5 and x.topic in x.followed:
        text = f"Matches topics you picked: {topic_name}"
        cands.append((model * x.topic_weight, Reason("topic_match", text)))
    if x.outlets >= 2 and x.trend > 0:
        text = f"Covered by {x.outlets} outlets in the last 24 h"
        cands.append((x.trend, Reason("trending", text)))
    if x.age_hours < NEW_HOURS:
        cands.append((x.fresh, Reason("new", "New in the last hour")))
    if x.diversity_pick:
        cands.append((0.0, Reason("diversity", "A different story from the rest of your feed")))
    codes = {r.code for _, r in cands}
    if x.topic in x.followed and "topic_match" not in codes:
        cands.append((-1.0, Reason("followed_topic", f"From {topic_name}, a topic you follow")))
    cands.sort(key=lambda c: -c[0])  # stable: equal weights keep the order above
    return [r for _, r in cands[:MAX_REASONS]]
