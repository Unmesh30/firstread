import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from firstread_ml.eval.metrics import mean_metrics, per_impression


def off(*sizes: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def test_perfect_and_worst_ranking() -> None:
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    good = mean_metrics(np.array([4.0, 3, 2, 1, 4, 3, 2, 1]), labels, off(4, 4))
    assert good == pytest.approx({"auc": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "ndcg@10": 1.0})
    bad = per_impression(np.array([1.0, 2, 3, 4]), np.array([1, 0, 0, 0]), off(4))
    assert bad["auc"][0] == 0.0
    assert bad["mrr"][0] == pytest.approx(1 / 4)
    assert bad["ndcg@10"][0] == pytest.approx(1 / np.log2(5))
    assert bad["ndcg@5"][0] == pytest.approx(1 / np.log2(5))


def test_hand_computed_two_positives() -> None:
    # Ranked order: [0, 1, 1, 0, 0] -> positives at ranks 2 and 3.
    scores = np.array([5.0, 4, 3, 2, 1])
    labels = np.array([0, 1, 1, 0, 0])
    m = per_impression(scores, labels, off(5))
    assert m["mrr"][0] == pytest.approx((1 / 2 + 1 / 3) / 2)
    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg@5"][0] == pytest.approx(dcg / idcg)
    assert m["auc"][0] == pytest.approx(4 / 6)  # 4 of 6 (pos, neg) pairs ordered correctly


def test_ndcg_cutoff() -> None:
    scores = -np.arange(12, dtype=float)
    labels = np.zeros(12, dtype=int)
    labels[10] = 1  # rank 11
    m = per_impression(scores, labels, off(12))
    assert m["ndcg@5"][0] == 0.0
    assert m["ndcg@10"][0] == 0.0
    assert m["mrr"][0] == pytest.approx(1 / 11)


def test_auc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    sizes = rng.integers(2, 30, 200)
    labels_list, scores_list = [], []
    for k in sizes:
        lab = rng.integers(0, 2, k)
        lab[0], lab[1] = 1, 0
        labels_list.append(lab)
        scores_list.append(rng.integers(0, 4, k).astype(float))  # many ties
    m = per_impression(np.concatenate(scores_list), np.concatenate(labels_list), off(*sizes))
    expected = [roc_auc_score(y, s) for y, s in zip(labels_list, scores_list, strict=True)]
    np.testing.assert_allclose(m["auc"], expected, atol=1e-12)


def test_ties_are_broken_randomly_not_by_input_order() -> None:
    # All-equal scores with the positive always first: input-order tie-breaking
    # would give MRR = 1. Random tie-breaking averages ~ H(n)/n.
    n_imp, k = 2000, 10
    labels = np.tile(np.r_[1, np.zeros(k - 1, dtype=int)], n_imp)
    m = mean_metrics(np.zeros(n_imp * k), labels, off(*[k] * n_imp))
    assert m["auc"] == 0.5
    assert m["mrr"] == pytest.approx(np.mean(1 / np.arange(1, k + 1)), abs=0.02)


def test_rejects_invalid_impressions() -> None:
    with pytest.raises(ValueError, match="positive"):
        per_impression(np.array([1.0, 2.0]), np.array([1, 1]), off(2))
    with pytest.raises(ValueError, match="finite"):
        per_impression(np.array([np.nan, 2.0]), np.array([1, 0]), off(2))
