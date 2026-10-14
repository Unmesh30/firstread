"""Local experiment tracking with MLflow, sqlite backend.

Runs are stored in ``data/mlruns/mlflow.db`` (sqlite) with artifacts in
``data/mlruns/artifacts``, all gitignored. Full mlflow is an ``ml``-only
dependency; the API image never installs it. Browse runs with ``make mlflow-ui``.
Tracking is best-effort context: ``FIRSTREAD_TRACKING=0`` disables it (tests do).
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from firstread_ml.paths import data_dir

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def store_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "mlruns"


def tracking_uri(root: Path | None = None) -> str:
    return f"sqlite:///{(store_dir(root) / 'mlflow.db').resolve()}"


class Run:
    def __init__(self, active: bool) -> None:
        self.active = active

    def params(self, params: dict[str, Any]) -> None:
        if self.active:
            import mlflow

            mlflow.log_params(params)

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.active:
            import mlflow

            mlflow.log_metrics(metrics, step=step)

    def artifact(self, path: Path) -> None:
        if self.active:
            import mlflow

            mlflow.log_artifact(str(path))


def _experiment_id(name: str, root: Path | None) -> str:
    import mlflow

    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return str(exp.experiment_id)
    artifacts = (store_dir(root) / "artifacts" / name).resolve()
    return str(mlflow.create_experiment(name, artifact_location=artifacts.as_uri()))


@contextlib.contextmanager
def run(
    name: str, experiment: str = "firstread", enabled: bool = True, root: Path | None = None
) -> Iterator[Run]:
    if not enabled or os.environ.get("FIRSTREAD_TRACKING", "1") == "0":
        yield Run(active=False)
        return
    import mlflow

    store_dir(root).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(root))
    with mlflow.start_run(run_name=name, experiment_id=_experiment_id(experiment, root)):
        yield Run(active=True)
