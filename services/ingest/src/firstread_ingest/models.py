"""Download the pinned model release and load the topic classifier (ADR 0005/0006).

A Hub tag can move, so the release is pinned in code by the SHA256 of its
``manifest.json`` (and, once uploaded, its Hub commit). Loading:

1. download ``manifest.json`` at the pinned revision and check its SHA256;
2. download the classifier and labels and check each against the manifest;
3. check the manifest's ``embedding`` / ``features`` blocks and topic ids match
   ``packages/core``: a mismatch is a startup error, never a silent skew.

Needs only ``onnxruntime`` and ``huggingface_hub``: no torch, no scikit-learn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
from firstread_core.topics import TOPIC_IDS

HF_REPO = "Unmesh30/firstread-models"
# tag -> {"manifest_sha256", "revision"}; revision is the Hub commit once published.
PINNED: dict[str, dict[str, str]] = {
    "v0.2.0": {
        "manifest_sha256": "2f8ebe6385b6fd2813e39968777e4a4ab80170196e95dc70d33502fa7a84fb35",
        "revision": "v0.2.0",  # replaced by the Hub commit after upload
    },
}
CLASSIFIER_FILES = ("category_classifier.onnx", "labels.json")


class ModelReleaseError(RuntimeError):
    """The downloaded release isn't the pinned one or doesn't match packages/core."""


def _sha256(path: Path) -> str:
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


def verify_release(folder: Path, manifest_sha256: str) -> dict[str, Any]:
    """Check a downloaded folder against the pin and packages/core; return the manifest."""
    if _sha256(folder / "manifest.json") != manifest_sha256:
        raise ModelReleaseError("manifest.json SHA256 differs from the pinned value")
    manifest: dict[str, Any] = json.loads((folder / "manifest.json").read_text())
    for name in CLASSIFIER_FILES:
        want = manifest["files"][name]["sha256"]
        if _sha256(folder / name) != want:
            raise ModelReleaseError(f"{name}: SHA256 differs from the manifest")
    for key, want_block in expected_blocks().items():
        if manifest.get(key) != want_block:
            raise ModelReleaseError(f"manifest {key} block doesn't match packages/core")
    cc = manifest.get("category_classifier", {})
    labels = json.loads((folder / "labels.json").read_text())
    if cc.get("kind") != "topics" or labels != list(TOPIC_IDS):
        raise ModelReleaseError("the release's classifier doesn't predict TOPIC_IDS")
    return manifest


def download_release(tag: str, cache_dir: Path | None = None, api: Any = None) -> Path:
    """Download the pinned classifier files for ``tag``; return the verified folder."""
    pin = PINNED.get(tag)
    if pin is None:
        raise ModelReleaseError(f"{tag} is not pinned in firstread_ingest.models.PINNED")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    folder = (cache_dir or default_model_cache_dir()) / "firstread-models" / tag
    for name in ("manifest.json", *CLASSIFIER_FILES):
        api.hf_hub_download(HF_REPO, name, revision=pin["revision"], local_dir=folder)
    verify_release(folder, pin["manifest_sha256"])
    return folder


@dataclass
class TopicClassifier:
    """ONNX Runtime session over the release's topic classifier."""

    session: Any
    labels: list[str]
    threshold: float
    version: str

    @classmethod
    def load(cls, folder: Path, threads: int = 1) -> TopicClassifier:
        import onnxruntime as ort

        manifest = json.loads((folder / "manifest.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        session = ort.InferenceSession(
            str(folder / "category_classifier.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        return cls(
            session=session,
            labels=json.loads((folder / "labels.json").read_text()),
            threshold=float(manifest["category_classifier"]["low_confidence_threshold"]),
            version=str(manifest["tag"]),
        )

    def probabilities(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(embeddings) == 0:
            return np.zeros((0, len(self.labels)), np.float32)
        _, probs = self.session.run(None, {"embedding": embeddings.astype(np.float32)})
        out: NDArray[np.float32] = np.asarray(probs, dtype=np.float32)
        return out
