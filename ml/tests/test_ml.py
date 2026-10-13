from firstread_ml import METRICS, __version__


def test_metrics_declared() -> None:
    assert "ndcg@10" in METRICS
    assert __version__
