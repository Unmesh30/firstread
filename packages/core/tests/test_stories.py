"""Story thresholds and complete-linkage grouping (ADR 0009, 0013)."""

import json
import uuid
from pathlib import Path

import numpy as np

from firstread_core import stories

DATA = Path(__file__).resolve().parents[3] / "docs" / "learn" / "data"


def test_tau_feed_is_what_the_rule_gives_on_the_committed_labels() -> None:
    docs = [
        json.loads((DATA / f).read_text()) for f in ("02-thresholds.json", "02-thresholds-2b.json")
    ]
    pooled = stories.pool_bins(*docs)
    computed = {pt: stories.threshold_from_bins(b) for pt, b in pooled.items()}
    assert computed == stories.COMPUTED_TAU_FEED


def test_the_fallback_raises_only_the_reviewed_pair_types_to_tau_story() -> None:
    assert stories.TAU_FEED_BY_PAIR == {
        ("title", "title"): 0.925,  # fallback (P3e persona replay, ADR 0013)
        ("title", "title_summary"): 0.75,
        ("title_summary", "title_summary"): 0.875,
    }
    for pt in stories.FALLBACK_TO_TAU_STORY:
        assert stories.TAU_FEED_BY_PAIR[pt] == stories.VALIDATED_BY_PAIR[pt][1]


def test_tau_feed_never_exceeds_tau_story() -> None:
    for pt, feed in stories.TAU_FEED_BY_PAIR.items():
        assert feed <= stories.VALIDATED_BY_PAIR[pt][1]


def test_rule_ignores_small_bins_and_needs_every_bin_above() -> None:
    bins = {
        "0.80": {"same_story": 9, "different": 1},  # 90%: ok
        "0.85": {"same_story": 1, "different": 1},  # n=2 < 3: ignored
        "0.90": {"duplicate": 3},
    }
    assert stories.threshold_from_bins(bins) == 0.80
    bins["0.90"] = {"duplicate": 2, "different": 2}
    assert stories.threshold_from_bins(bins) is None


def unit(*xs: float) -> np.ndarray:
    v = np.zeros(384, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def angle(cos: float) -> np.ndarray:
    return unit(cos, float(np.sqrt(1 - cos**2)))


def test_grouping_does_not_chain() -> None:
    # A~B and B~C at 0.95 (>= 0.925), but A~C at ~0.81: three items, two groups.
    a = unit(1.0)
    b = angle(0.95)
    c_dir = np.arccos(0.95) * 2  # twice the angle from A
    c = unit(np.cos(c_dir), np.sin(c_dir))
    assert float(a @ c) < stories.TAU_FEED_BY_PAIR[("title", "title")]
    ids = [uuid.uuid4() for _ in range(3)]
    groups = stories.group(np.stack([a, b, c]), ["title"] * 3, ids)
    assert groups == [0, 0, 1]  # B joins A's group; C would need to be close to A too


def test_same_cluster_always_groups_and_pair_types_have_their_own_threshold() -> None:
    a, far = unit(1.0), unit(0.0, 1.0)
    cluster = uuid.uuid4()
    assert stories.group(np.stack([a, far]), ["title", "title"], [cluster, cluster]) == [0, 0]
    mixed = angle(0.78)  # >= 0.75 for title~title_summary, < 0.925 for title~title
    src_a = ["title", "title_summary"]
    src_b = ["title", "title"]
    ids = [uuid.uuid4(), uuid.uuid4()]
    assert stories.group(np.stack([a, mixed]), src_a, ids) == [0, 0]  # type: ignore[arg-type]
    assert stories.group(np.stack([a, mixed]), src_b, ids) == [0, 1]  # type: ignore[arg-type]


def test_near_any_uses_the_stricter_tau_story() -> None:
    read = np.stack([unit(1.0)])
    close, related = angle(0.93), angle(0.90)  # tau_story(title, title) = 0.925
    hit = stories.near_any(np.stack([close, related]), ["title", "title"], read, ["title"])
    assert hit.tolist() == [True, False]
    assert not stories.near_any(np.zeros((0, 384)), [], read, ["title"]).any()
