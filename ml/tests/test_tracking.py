from pathlib import Path

import pytest

from firstread_ml import tracking


def test_tracking_writes_to_sqlite_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRSTREAD_TRACKING", "1")
    with tracking.run("unit", experiment="test-exp", root=tmp_path) as run:
        assert run.active
        run.params({"a": 1})
        run.metrics({"m": 0.5}, step=1)
        f = tmp_path / "note.txt"
        f.write_text("hi")
        run.artifact(f)
    assert (tmp_path / "mlruns" / "mlflow.db").is_file()
    assert list((tmp_path / "mlruns" / "artifacts" / "test-exp").rglob("note.txt"))

    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking.tracking_uri(tmp_path))
    exp = client.get_experiment_by_name("test-exp")
    assert exp is not None
    (r,) = client.search_runs([exp.experiment_id])
    assert r.data.params == {"a": "1"}
    assert r.data.metrics == {"m": 0.5}


def test_tracking_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FIRSTREAD_TRACKING", "0")
    with tracking.run("off", root=tmp_path) as run:
        assert not run.active
        run.metrics({"m": 1.0})  # no-op
    assert not (tmp_path / "mlruns").exists()
