"""The ingest workflow's window cache: least privilege and a safe step order (ADR 0012)."""

from pathlib import Path
from typing import Any

import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ingest.yml"


def load() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text())
    return data


def test_actions_write_is_granted_to_the_ingest_job_only() -> None:
    wf = load()
    assert wf["permissions"] == {"contents": "read"}  # workflow default stays read-only
    assert wf["jobs"]["ingest"]["permissions"] == {"contents": "read", "actions": "write"}


def test_old_snapshots_are_deleted_only_after_the_new_one_is_saved() -> None:
    steps = [s.get("name", "") for s in load()["jobs"]["ingest"]["steps"]]
    restore = steps.index("Restore the window snapshot")
    run = steps.index("Run ingestion")
    save = steps.index("Save the window snapshot")
    delete = steps.index("Delete superseded window snapshots")
    assert restore < run < save < delete
    step = load()["jobs"]["ingest"]["steps"][delete]
    assert step["if"].startswith("success()")  # a failed save skips the delete
    assert "'.[2:][].id'" in step["run"]  # the newest 2 entries are always kept


def test_the_run_uses_the_restored_window_file() -> None:
    env = load()["jobs"]["ingest"]["env"]
    assert env["INGEST_WINDOW_FILE"].endswith(".window/window.npz")
