import uuid
from datetime import UTC, datetime

import numpy as np
import pytest

from firstread_ingest.dedupe import Item, Recent, Thresholds, decide, pair_type

NOW = datetime(2026, 10, 14, tzinfo=UTC)
D = 384


def unit(i: int) -> np.ndarray:
    v: np.ndarray = np.zeros(D, np.float32)
    v[i] = 1.0
    return v


def at_cos(base: np.ndarray, other_axis: int, cos: float) -> np.ndarray:
    """A unit vector with exactly ``cos`` cosine to ``base`` (an axis vector)."""
    v: np.ndarray = (cos * base + np.sqrt(1 - cos**2) * unit(other_axis)).astype(np.float32)
    return v


def item(n: int, src: str = "title_summary") -> Item:
    return Item(f"https://example.com/{n}", f"fp{n}", src, NOW)  # type: ignore[arg-type]


def recent_with(vec: np.ndarray, src: str = "title_summary") -> tuple[Recent, uuid.UUID]:
    aid = uuid.uuid4()
    r = Recent([aid], [aid], [src], vec[None, :], {"https://example.com/old": aid}, {"fpold": aid})  # type: ignore[list-item]
    return r, aid


@pytest.mark.parametrize(
    ("cos", "action", "reason"),
    [
        (0.98, "near_duplicate", "cosine"),
        (0.9751, "near_duplicate", "cosine"),  # float32: exactly 0.975 can round below
        (0.9749, "insert", "same_story"),
        (0.8751, "insert", "same_story"),
        (0.8749, "insert", "new_story"),
    ],
)
def test_threshold_boundaries(cos: float, action: str, reason: str) -> None:
    recent, old = recent_with(unit(0))
    (d,) = decide([item(1)], at_cos(unit(0), 1, cos)[None, :], recent, Thresholds())
    assert (d.action, d.reason) == (action, reason)
    if reason == "new_story":
        assert d.cluster_id == d.article_id
    else:
        assert d.cluster_id == old


def test_exact_duplicates_by_url_and_fingerprint() -> None:
    recent, old = recent_with(unit(0))
    url_dup = Item("https://example.com/old", "fpX", "title", NOW)
    fp_dup = Item("https://example.com/new", "fpold", "title", NOW)
    vecs = np.stack([unit(5), unit(6)])
    a, b = decide([url_dup, fp_dup], vecs, recent, Thresholds())
    assert (a.action, a.reason, a.cluster_id) == ("duplicate", "url", old)
    assert (b.action, b.reason, b.cluster_id) == ("duplicate", "fingerprint", old)


def test_duplicates_inside_one_batch() -> None:
    items = [item(1), item(1), Item("https://example.com/2", "fp1", "title", NOW), item(3)]
    vecs = np.stack([unit(0), unit(0), unit(1), at_cos(unit(0), 2, 0.98)])
    d = decide(items, vecs, Recent.empty(), Thresholds())
    assert [x.action for x in d] == ["insert", "duplicate", "duplicate", "near_duplicate"]
    assert len({x.cluster_id for x in d}) == 1  # all the same story, for coverage


def test_cluster_continues_across_runs() -> None:
    first = decide([item(1)], unit(0)[None, :], Recent.empty(), Thresholds())[0]
    assert first.article_id is not None
    recent = Recent(
        [first.article_id], [first.cluster_id], ["title_summary"], unit(0)[None, :], {}, {}
    )
    second = decide([item(2)], at_cos(unit(0), 3, 0.92)[None, :], recent, Thresholds())[0]
    assert second.action == "insert"
    assert second.cluster_id == first.cluster_id


def test_per_pair_type_thresholds() -> None:
    cross = ("title", "title_summary")
    t = Thresholds(by_pair={cross: (0.9, 0.75)})  # type: ignore[dict-item]
    recent, _ = recent_with(unit(0), "title_summary")
    vec = at_cos(unit(0), 1, 0.8)[None, :]
    assert decide([item(1, "title")], vec, recent, t)[0].reason == "same_story"
    assert decide([item(1, "title_summary")], vec, recent, t)[0].reason == "new_story"
    assert pair_type("title_summary", "title") == pair_type("title", "title_summary")


def test_validated_thresholds_are_stricter_where_the_sample_was_thin() -> None:
    from firstread_ingest.dedupe import VALIDATED_BY_PAIR

    gdelt = VALIDATED_BY_PAIR[("title", "title")]
    cross = VALIDATED_BY_PAIR[("title", "title_summary")]
    rss = VALIDATED_BY_PAIR[("title_summary", "title_summary")]
    assert gdelt == (0.975, 0.875)
    assert cross[1] < gdelt[1]  # title-only vs title+summary vectors sit further apart
    assert all(dup > story for dup, story in (gdelt, cross, rss))
