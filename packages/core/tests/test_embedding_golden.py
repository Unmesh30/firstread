"""Golden-vector test: fails if the embedding model, library or text prep changes.

Downloads the pinned model (~67 MB, cached). The golden file was generated on
the platform recorded in it; see docs/learn/01-offline-ml.md (gotchas) for how
the tolerance relates to cross-platform differences.
"""

import json
import platform
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

# Cross-platform difference measured in CI (2026-10-14): golden vectors from macOS
# arm64 vs. the same pinned int8 model on GitHub's Linux x86-64 runner. The int8
# kernels differ by CPU, so vectors are close but not bit-identical.
MEASURED_MAX_ABS_DIFF = 2.30e-4
MEASURED_ONE_MINUS_COS = 3.6e-7

# Tolerances: ~3x the measured cross-platform difference. A one-word change to the
# text must move the vector by at least NEGATIVE_CONTROL_MARGIN times these
# (measured: >= 0.083 max abs diff, >= 0.109 in 1 - cosine), so the test still
# tells different inputs or models apart. Never loosen these without re-measuring.
MAX_ABS_DIFF = 7e-4
MIN_COSINE = 1.0 - 1.1e-6
NEGATIVE_CONTROL_MARGIN = 10.0


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def _golden() -> tuple[list[str], np.ndarray[Any, Any]]:
    texts = [article_text(i["title"], i["summary"]) for i in GOLDEN["items"]]
    return texts, np.array([i["vector"] for i in GOLDEN["items"]], dtype=np.float32)


def test_tolerances_stay_tied_to_the_measurement() -> None:
    # Guards against quietly loosening the gate: tolerances must stay within
    # 2-4x of the measured cross-platform difference.
    assert 2 * MEASURED_MAX_ABS_DIFF <= MAX_ABS_DIFF <= 4 * MEASURED_MAX_ABS_DIFF
    assert 2 * MEASURED_ONE_MINUS_COS <= 1 - MIN_COSINE <= 4 * MEASURED_ONE_MINUS_COS


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
    # The measurement is in the message so a CI-only failure reports the
    # cross-platform difference (see the learn chapter before changing tolerances).
    measured = (
        f"max_abs_diff={max_abs:.3e} min_cosine={min_cos:.8f} "
        f"(golden from {GOLDEN['platform']}, running on {platform.system()} {platform.machine()})"
    )
    assert max_abs <= MAX_ABS_DIFF, measured
    assert min_cos >= MIN_COSINE, measured


def test_negative_control_is_far_outside_tolerance(embedder: Embedder) -> None:
    texts, expected = _golden()
    perturbed = [t.replace(t.split()[1], "banana", 1) for t in texts]
    actual = embedder.embed(perturbed)
    max_abs = np.abs(actual - expected).max(axis=1)
    one_minus_cos = 1.0 - (actual * expected).sum(axis=1)
    assert (max_abs >= NEGATIVE_CONTROL_MARGIN * MAX_ABS_DIFF).all()
    assert (one_minus_cos >= NEGATIVE_CONTROL_MARGIN * (1.0 - MIN_COSINE)).all()


def test_vectors_do_not_depend_on_batch_composition(embedder: Embedder) -> None:
    """Ingestion embeds different batches than training did, so a text's vector
    must not depend on what it is batched with (padding length, batch size)."""
    texts, _ = _golden()
    alone = np.stack([embedder.embed([t])[0] for t in texts])
    long_text = article_text("Long report", " ".join(["markets rallied after the data"] * 60))
    fillers = ["a", "Hi", long_text, "Rates", long_text[:400], "x y", long_text[:700]] * 3
    mixed = embedder.embed(fillers[:9] + texts + fillers[9:])
    together = mixed[9 : 9 + len(texts)]
    max_abs = float(np.abs(together - alone).max())
    min_cos = float((together * alone).sum(axis=1).min())
    print(f"batch invariance: max_abs_diff={max_abs:.3e} min_cosine={min_cos:.8f}")
    assert max_abs <= MAX_ABS_DIFF, f"max_abs_diff={max_abs:.3e}"
    assert min_cos >= MIN_COSINE, f"min_cosine={min_cos:.8f}"
