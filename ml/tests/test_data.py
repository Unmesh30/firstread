import numpy as np

from firstread_ml.mind.data import Impressions, _ranges
from firstread_ml.testing.builders import make_impressions as make


def test_ranges() -> None:
    out = _ranges(np.array([5, 0, 10]), np.array([2, 0, 3]))
    assert out.tolist() == [5, 6, 10, 11, 12]
    assert _ranges(np.array([], dtype=np.int64), np.array([], dtype=np.int64)).size == 0


def test_take_and_concat_roundtrip() -> None:
    imps = make([[1, 2], [], [3]], [[4, 5], [6, 7, 8], [9, 1]], [[1, 0], [0, 1, 0], [0, 1]])
    sub = imps.take(np.array([True, False, True]))
    assert sub.n == 2
    assert sub.cand.tolist() == [4, 5, 9, 1]
    assert sub.hist.tolist() == [1, 2, 3]
    assert [h.tolist() for h in sub.histories()] == [[1, 2], [3]]
    both = Impressions.concat([imps.take(np.array([0])), imps.take(np.array([1, 2]))])
    np.testing.assert_array_equal(both.cand, imps.cand)
    np.testing.assert_array_equal(both.cand_offsets, imps.cand_offsets)


def test_histories_row_range() -> None:
    imps = make([[1, 2], [], [3]], [[4, 5], [6, 7], [9, 1]], [[1, 0], [0, 1], [0, 1]])
    assert [h.tolist() for h in imps.histories(slice(1, 3))] == [[], [3]]
    assert imps.cand_imp.tolist() == [0, 0, 1, 1, 2, 2]
