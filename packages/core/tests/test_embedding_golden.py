"""Golden-vector test: fails if the embedding model, library or text prep changes.

Downloads the pinned model (~67 MB, cached). The golden file was generated on
the platform recorded in it; see docs/learn/01-offline-ml.md (gotchas) for how
the tolerance relates to cross-platform differences.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from firstread_core.embedding import (
    EMBED_MODEL_FILE,
    EMBED_MODEL_REVISION,
    EMBED_MODEL_SHA256,
    FASTEMBED_VERSION,
    Embedder,
    fetch_model,
    sha256_file,
)
from firstread_core.text import article_text

pytestmark = pytest.mark.model

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_bge_small_en_v1_5.json").read_text())

# Tolerances. A one-word change to the text must move the vector by at least
# NEGATIVE_CONTROL_MARGIN times these, otherwise the test could not tell models apart.
MAX_ABS_DIFF = 1e-4
MIN_COSINE = 0.9999
NEGATIVE_CONTROL_MARGIN = 10.0


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def _golden() -> tuple[list[str], np.ndarray[Any, Any]]:
    texts = [article_text(i["title"], i["summary"]) for i in GOLDEN["items"]]
    return texts, np.array([i["vector"] for i in GOLDEN["items"]], dtype=np.float32)


def test_golden_metadata_matches_pins() -> None:
    assert GOLDEN["revision"] == EMBED_MODEL_REVISION
    assert GOLDEN["fastembed_version"] == FASTEMBED_VERSION
    # Text prep must still produce the exact strings that were embedded.
    for item in GOLDEN["items"]:
        assert article_text(item["title"], item["summary"]) == item["text"]


def test_model_file_checksum() -> None:
    assert sha256_file(fetch_model() / EMBED_MODEL_FILE) == EMBED_MODEL_SHA256


def test_golden_vectors(embedder: Embedder) -> None:
    texts, expected = _golden()
    actual = embedder.embed(texts)
    max_abs = float(np.abs(actual - expected).max())
    min_cos = float((actual * expected).sum(axis=1).min())
    print(f"golden: max_abs_diff={max_abs:.3e} min_cosine={min_cos:.8f} on {GOLDEN['platform']}")
    assert max_abs <= MAX_ABS_DIFF
    assert min_cos >= MIN_COSINE


def test_negative_control_is_far_outside_tolerance(embedder: Embedder) -> None:
    texts, expected = _golden()
    perturbed = [t.replace(t.split()[1], "banana", 1) for t in texts]
    actual = embedder.embed(perturbed)
    max_abs = np.abs(actual - expected).max(axis=1)
    one_minus_cos = 1.0 - (actual * expected).sum(axis=1)
    assert (max_abs >= NEGATIVE_CONTROL_MARGIN * MAX_ABS_DIFF).all()
    assert (one_minus_cos >= NEGATIVE_CONTROL_MARGIN * (1.0 - MIN_COSINE)).all()
