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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core import release as core_release
from firstread_core.topics import TOPIC_IDS

# The pins and the generic checks live in core (the API verifies the same release).
HF_REPO = core_release.HF_REPO
PINNED = core_release.PINNED
CLASSIFIER_FILES = core_release.CLASSIFIER_FILES
ModelReleaseError = core_release.ModelReleaseError
expected_blocks = core_release.expected_blocks


def verify_release(folder: Path, manifest_sha256: str) -> dict[str, Any]:
    """The classifier files against the pin and packages/core, and its labels."""
    manifest = core_release.verify_release(folder, manifest_sha256, CLASSIFIER_FILES)
    cc = manifest.get("category_classifier", {})
    labels = json.loads((folder / "labels.json").read_text())
    if cc.get("kind") != "topics" or labels != list(TOPIC_IDS):
        raise ModelReleaseError("the release's classifier doesn't predict TOPIC_IDS")
    return manifest


def download_release(tag: str, cache_dir: Path | None = None, api: Any = None) -> Path:
    """Download the pinned classifier files for ``tag``; return the verified folder."""
    folder = core_release.download_release(tag, CLASSIFIER_FILES, cache_dir, api)
    verify_release(folder, PINNED[tag]["manifest_sha256"])
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
