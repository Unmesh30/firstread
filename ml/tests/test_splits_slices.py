import numpy as np

from firstread_ml.eval.slices import build_slices, restrict
from firstread_ml.mind.splits import DAY, click_counts, clicked_articles, first_seen, holdout_split
from firstread_ml.testing.builders import make_impressions as make


def test_holdout_is_last_calendar_day() -> None:
    imps = make([[], [], []], [[1, 2], [3, 4], [5, 6]], [[1, 0]] * 3)
    imps = imps.__class__(**{**imps.__dict__, "time": np.array([0, DAY + 5, 2 * DAY + 7])})
    fit, hold = holdout_split(imps)
    assert fit.time.tolist() == [0, DAY + 5]
    assert hold.time.tolist() == [2 * DAY + 7]


def test_clicked_articles_and_counts() -> None:
    imps = make([[7], []], [[1, 2], [1, 3]], [[1, 0], [1, 0]])
    clicked = clicked_articles(imps, 10)
    assert np.flatnonzero(clicked).tolist() == [1, 7]  # positives + histories
    assert click_counts(imps, 10)[1] == 2


def test_first_seen_is_earliest_candidate_time() -> None:
    a = make([[]], [[1, 2]], [[1, 0]])  # time 0
    b = make([[], []], [[2, 3], [3, 4]], [[1, 0], [1, 0]])  # times 0, 100
    b = b.__class__(**{**b.__dict__, "time": np.array([50, 100])})
    fs = first_seen([a, b], 6)
    assert fs[[1, 2, 3, 4]].tolist() == [0, 0, 50, 100]
    assert fs[0] == fs[5] == -DAY  # never a candidate: earliest minus a day


def test_restrict_drops_impressions_without_both_labels() -> None:
    imps = make([[], []], [[1, 2, 3], [4, 5]], [[1, 0, 0], [1, 0]])
    keep = np.array([True, True, False, True, False])
    s = restrict(imps, keep, "x")
    assert s.n_impressions == 1  # second impression keeps only its positive
    assert s.idx.tolist() == [0, 1]


def test_build_slices_definitions() -> None:
    test = make(
        [[], [9], [1, 2, 3]],
        [[1, 2], [3, 4], [2, 5]],
        [[1, 0], [0, 1], [1, 0]],
    )
    clicked = np.zeros(10, dtype=bool)
    clicked[[1, 2]] = True
    category = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    slices = {s.name: s for s in build_slices(test, clicked, category, ["a", "b"])}
    assert slices["overall"].n_impressions == 3
    assert slices["cold_candidates"].n_impressions == 1  # only impression 1 has 2 cold cands
    assert slices["cold_users"].n_impressions == 2
    assert slices["cold_users"].extra == {"empty_history": 1}
    assert slices["category:b"].n_impressions == 1  # impression 2: cands 2 (b) and 5 (b)
