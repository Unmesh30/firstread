"""Local experiment tracking with MLflow (file store in ``data/mlruns``).

We use mlflow-skinny (no server, no SQL deps). MLflow 3 puts its file store in
"maintenance mode" and raises unless MLFLOW_ALLOW_FILE_STORE is set; the
file store is enough for a single-machine project. Browse runs with
``uvx --from mlflow mlflow ui --backend-store-uri data/mlruns``.
Tracking is best-effort: if MLflow fails, training continues.
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from firstread_ml.paths import data_dir

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


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


@contextlib.contextmanager
def run(name: str, experiment: str = "firstread", enabled: bool = True) -> Iterator[Run]:
    if not enabled or os.environ.get("FIRSTREAD_TRACKING", "1") == "0":
        yield Run(active=False)
        return
    import mlflow

    store = data_dir() / "mlruns"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name):
        yield Run(active=True)
