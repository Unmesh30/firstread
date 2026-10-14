import numpy as np

from firstread_core.embedding import EMBED_DIM, l2_normalize
from firstread_core.features import (
    PAD_INDEX,
    gather_history,
    history_index_matrix,
    mean_history_vector,
    pad_history,
)


def test_history_index_matrix_right_aligns_most_recent() -> None:
    idx, mask = history_index_matrix([[1, 2, 3], [], [4, 5, 6, 7, 8]], history_len=4)
    assert idx.tolist() == [
        [PAD_INDEX, 1, 2, 3],
        [PAD_INDEX] * 4,
        [5, 6, 7, 8],  # oldest (4) dropped, newest last
    ]
    assert mask.tolist() == [
        [False, True, True, True],
        [False] * 4,
        [True] * 4,
    ]


def test_pad_history_matches_index_version() -> None:
    rng = np.random.default_rng(0)
    table = l2_normalize(rng.normal(size=(10, EMBED_DIM)))
    hist_ids = [3, 1, 4, 1, 5]
    hist, mask = pad_history(table[hist_ids], history_len=8)
    idx, mask2 = history_index_matrix([hist_ids], history_len=8)
    np.testing.assert_array_equal(mask, mask2[0])
    np.testing.assert_array_equal(hist, gather_history(table, idx, mask2)[0])
    assert (hist[~mask] == 0).all()


def test_mean_history_vector_single_and_batch() -> None:
    hist = np.zeros((3, EMBED_DIM), dtype=np.float32)
    hist[1, 0] = 1.0
    hist[2, 1] = 1.0
    mask = np.array([False, True, True])
    vec = mean_history_vector(hist, mask)
    expected = np.zeros(EMBED_DIM, dtype=np.float32)
    expected[:2] = 1 / np.sqrt(2)
    np.testing.assert_allclose(vec, expected, atol=1e-6)

    batch = mean_history_vector(np.stack([hist, hist]), np.stack([mask, mask]))
    np.testing.assert_allclose(batch, [expected, expected], atol=1e-6)


def test_mean_history_vector_empty_uses_fallback() -> None:
    hist = np.zeros((2, 4, EMBED_DIM), dtype=np.float32)
    hist[1, 3, 5] = 2.0
    mask = np.array([[False] * 4, [False, False, False, True]])
    fallback = np.zeros(EMBED_DIM, dtype=np.float32)
    fallback[0] = 3.0
    out = mean_history_vector(hist, mask, fallback=fallback)
    assert out[0, 0] == 1.0  # normalized fallback
    assert out[1, 5] == 1.0
    assert np.isfinite(out).all()
    # Without a fallback, an empty history is a zero vector (never NaN).
    assert not mean_history_vector(hist[0], mask[0]).any()
