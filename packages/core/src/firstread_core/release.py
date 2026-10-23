"""Download and verify a pinned model release (ADR 0005). Shared by ingest and the API.

A Hub tag can move, so a release is pinned in code by the SHA256 of its
``manifest.json`` and by its Hub commit. Verifying a folder:

1. ``manifest.json`` hashes to the pinned value;
2. every requested file hashes to the manifest's value;
3. the manifest's ``embedding`` and ``features`` blocks match ``packages/core``:
   a mismatch is a startup error, never a silent skew.

``huggingface_hub`` is imported only to download (the ``core[release]`` extra). The
API image bakes the verified files in at build time and only verifies at runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from firstread_core.embedding import (
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_MODEL_FILE,
    EMBED_MODEL_REVISION,
    EMBED_MODEL_SOURCE,
    FASTEMBED_VERSION,
    default_model_cache_dir,
)
from firstread_core.features import HISTORY_LEN
from firstread_core.text import TEXT_PREP_VERSION

HF_REPO = "Unmesh30/firstread-models"
# tag -> {"manifest_sha256", "revision"}. Files are downloaded at the Hub commit, never
# the (movable) tag, and the manifest must hash to the pinned value.
PINNED: dict[str, dict[str, str]] = {
    "v0.2.0": {
        "manifest_sha256": "2f8ebe6385b6fd2813e39968777e4a4ab80170196e95dc70d33502fa7a84fb35",
        "revision": "099201fabb57bde64af046d0c3be2f690adfff40",  # Hub commit of tag v0.2.0
    },
}
CLASSIFIER_FILES = ("category_classifier.onnx", "labels.json")
ENCODER_FILES = ("user_encoder.onnx", "population.npy")


class ModelReleaseError(RuntimeError):
    """The release isn't the pinned one or doesn't match packages/core."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_blocks() -> dict[str, dict[str, Any]]:
    return {
        "embedding": {
            "model": EMBED_MODEL,
            "source": EMBED_MODEL_SOURCE,
            "revision": EMBED_MODEL_REVISION,
            "file": EMBED_MODEL_FILE,
            "fastembed": FASTEMBED_VERSION,
            "text_prep_version": TEXT_PREP_VERSION,
            "l2_normalized": True,
        },
        "features": {"history_len": HISTORY_LEN, "embed_dim": EMBED_DIM, "right_aligned": True},
    }


def pin(tag: str) -> dict[str, str]:
    p = PINNED.get(tag)
    if p is None:
        raise ModelReleaseError(f"{tag} is not pinned in firstread_core.release.PINNED")
    return p


def verify_release(folder: Path, manifest_sha256: str, files: Sequence[str]) -> dict[str, Any]:
    """Check ``files`` in a release folder against the pin and packages/core."""
    if not (folder / "manifest.json").is_file():
        raise ModelReleaseError(f"{folder}: no manifest.json")
    if sha256(folder / "manifest.json") != manifest_sha256:
        raise ModelReleaseError("manifest.json SHA256 differs from the pinned value")
    manifest: dict[str, Any] = json.loads((folder / "manifest.json").read_text())
    for name in files:
        want = manifest["files"][name]["sha256"]
        if not (folder / name).is_file() or sha256(folder / name) != want:
            raise ModelReleaseError(f"{name}: SHA256 differs from the manifest")
    for key, want_block in expected_blocks().items():
        if manifest.get(key) != want_block:
            raise ModelReleaseError(f"manifest {key} block doesn't match packages/core")
    return manifest


def download_release(
    tag: str, files: Sequence[str], cache_dir: Path | None = None, api: Any = None
) -> Path:
    """Download ``files`` of the pinned ``tag`` at its Hub commit; return the verified folder."""
    p = pin(tag)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    folder = (cache_dir or default_model_cache_dir()) / "firstread-models" / tag
    for name in ("manifest.json", *files):
        api.hf_hub_download(HF_REPO, name, revision=p["revision"], local_dir=folder)
    verify_release(folder, p["manifest_sha256"], files)
    return folder


def main() -> None:
    """``python -m firstread_core.release <tag> <out_dir> <file>...``: used by the API's
    Docker build stage to fetch and verify the encoder files once, at build time."""
    import shutil
    import sys

    tag, out, *names = sys.argv[1:]
    folder = download_release(tag, names)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", *names):
        shutil.copy2(folder / name, dest / name)
    verify_release(dest, pin(tag)["manifest_sha256"], names)
    print(f"verified {tag}: {', '.join(names)} -> {dest}")


if __name__ == "__main__":
    main()
