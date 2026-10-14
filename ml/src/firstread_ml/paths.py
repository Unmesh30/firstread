"""Filesystem locations. Everything data-like lives under the gitignored ``data/``."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """``$FIRSTREAD_DATA_DIR`` or ``<repo>/data``."""
    env = os.environ.get("FIRSTREAD_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data"


def mind_raw_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "mind" / "raw"


def processed_dir(variant: str, root: Path | None = None) -> Path:
    return (root or data_dir()) / "mind" / "processed" / variant


def emb_cache_dir(variant: str, root: Path | None = None) -> Path:
    return (root or data_dir()) / "cache" / "emb" / variant


def reports_dir() -> Path:
    """Committed reports (aggregates only, never MIND rows or text)."""
    return REPO_ROOT / "ml" / "reports"
