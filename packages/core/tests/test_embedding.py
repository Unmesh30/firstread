from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from firstread_core import embedding
from firstread_core.embedding import (
    EMBED_DIM,
    Embedder,
    EmbeddingModelError,
    embedding_cache_key,
    l2_normalize,
)


def fake_backend(texts: list[str]) -> Iterable[NDArray[Any]]:
    # Deterministic, unnormalized vectors that depend on the text.
    for t in texts:
        rng = np.random.default_rng(sum(map(ord, t)))
        yield rng.normal(size=EMBED_DIM) * 3.0


def test_embedder_normalizes_and_shapes() -> None:
    out = Embedder(backend=fake_backend).embed(["a", "bb", "ccc"])
    assert out.shape == (3, EMBED_DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_embedder_empty_input() -> None:
    assert Embedder(backend=fake_backend).embed([]).shape == (0, EMBED_DIM)


def test_embedder_rejects_wrong_dimension() -> None:
    def bad(texts: list[str]) -> Iterable[NDArray[Any]]:
        return [np.ones(10) for _ in texts]

    with pytest.raises(EmbeddingModelError, match="shape"):
        Embedder(backend=bad).embed(["x"])


def test_l2_normalize_keeps_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_cache_key_is_stable_and_complete() -> None:
    key1, comps = embedding_cache_key()
    key2, _ = embedding_cache_key()
    assert key1 == key2
    assert len(key1) == 64
    for name in (
        "embed_model_revision",
        "fastembed_version",
        "text_prep_version",
        "max_chars",
    ):
        assert name in comps


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("TEXT_PREP_VERSION", 999),
        ("MAX_CHARS", 7),
        ("EMBED_MODEL_REVISION", "0" * 40),
        ("FASTEMBED_VERSION", "0.0.1"),
    ],
)
def test_cache_key_changes_with_each_component(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    before, _ = embedding_cache_key()
    monkeypatch.setattr(embedding, attr, value)
    after, _ = embedding_cache_key()
    assert before != after


def test_fetch_model_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / embedding.EMBED_MODEL_FILE).write_bytes(b"not the model")

    def fake_snapshot_download(**kwargs: Any) -> str:
        assert kwargs["revision"] == embedding.EMBED_MODEL_REVISION
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    with pytest.raises(EmbeddingModelError, match="SHA256"):
        embedding.fetch_model(tmp_path)


def test_fastembed_version_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding.check_fastembed_version()  # the installed version is the pinned one
    monkeypatch.setattr(embedding, "FASTEMBED_VERSION", "0.0.1")
    with pytest.raises(EmbeddingModelError, match="pinned"):
        embedding.check_fastembed_version()
