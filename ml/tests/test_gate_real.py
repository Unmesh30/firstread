"""Real-data eval gate rules (pure logic; the gate run itself is covered by test_release)."""

import numpy as np

from firstread_ml.eval import gate_real


def test_gate_real_rules() -> None:
    vals = {
        "onnx_release": {"overall": 0.40, "cold_candidates": 0.55, "cold_users": 0.36},
        "mean_history": {"overall": 0.39, "cold_candidates": 0.56, "cold_users": 0.35},
    }
    assert gate_real.check(vals, None) == [
        "onnx_release/cold_candidates: nDCG@10 0.550000 does not beat mean_history 0.560000"
    ]
    vals["onnx_release"]["cold_candidates"] = 0.57
    assert gate_real.check(vals, None) == []


def test_gate_real_regression_rule() -> None:
    vals = {
        "onnx_release": {"overall": 0.40, "cold_candidates": 0.57},
        "mean_history": {"overall": 0.39, "cold_candidates": 0.56},
    }
    assert gate_real.check(vals, {"ndcg@10": {"onnx_release": {"overall": 0.403}}}) == []
    (fail,) = gate_real.check(vals, {"ndcg@10": {"onnx_release": {"overall": 0.41}}})
    assert fail.startswith("onnx_release/overall")


def test_gate_real_subset_is_fixed() -> None:
    a, b = gate_real.subset_rows(100_000), gate_real.subset_rows(100_000)
    assert len(a) == gate_real.SUBSET_SIZE
    np.testing.assert_array_equal(a, b)
    assert (np.diff(a) > 0).all()
    np.testing.assert_array_equal(gate_real.subset_rows(10), np.arange(10))
