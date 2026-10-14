"""The one article embedding function (bge-small-en-v1.5 via fastembed).

Training (ml), ingestion and serving all embed through :class:`Embedder`, so
they can never disagree about what a vector means (train/serve skew).

fastembed always fetches the *latest* revision of its model repo, which would
let the model change under us. Instead we download a pinned revision ourselves,
verify the ONNX file's SHA256 and hand fastembed the local directory.

The fastembed and huggingface_hub imports are lazy: they come with the
``firstread-core[embed]`` extra, which the API image does not install.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.text import MAX_CHARS, TEXT_PREP_VERSION

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# fastembed's registry maps EMBED_MODEL to this int8-quantized ONNX export.
EMBED_MODEL_SOURCE = "Qdrant/bge-small-en-v1.5-onnx-Q"
EMBED_MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
EMBED_MODEL_FILE = "model_optimized.onnx"
EMBED_MODEL_SHA256 = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
EMBED_MODEL_FILES = (
    EMBED_MODEL_FILE,
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EMBED_DIM = 384
FASTEMBED_VERSION = "0.9.0"

# Measured on an M-series Mac (docs/learn/01-offline-ml.md): with length-sorted
# input, batch 16 and 4 ONNX threads were fastest (~100 texts/s); bigger batches
# waste compute on padding and more threads contend. Thread count doesn't change
# vectors; padding does, slightly (<= ~4.4e-4 abs, 1 - cos <= ~5e-7), which
# test_vectors_do_not_depend_on_batch_composition bounds.
DEFAULT_BATCH_SIZE = 16
DEFAULT_THREADS = 4

Vectors = NDArray[np.float32]
Backend = Callable[[list[str]], Iterable[NDArray[Any]]]


class EmbeddingModelError(RuntimeError):
    """The embedding model or library is not the pinned one."""


def l2_normalize(x: NDArray[Any], eps: float = 1e-12) -> Vectors:
    """Scale each row (or a single vector) to unit L2 norm; zero rows stay zero."""
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    out: Vectors = (arr / np.maximum(norms, eps)).astype(np.float32, copy=False)
    return out


def embedding_cache_key() -> tuple[str, dict[str, str | int]]:
    """Identify everything that determines an article's vector.

    Returns ``(sha256_hex, components)``. Embedding caches store the
    components next to the vectors and are rebuilt when the key differs.
    """
    components: dict[str, str | int] = {
        "embed_model": EMBED_MODEL,
        "embed_model_source": EMBED_MODEL_SOURCE,
        "embed_model_revision": EMBED_MODEL_REVISION,
        "embed_model_sha256": EMBED_MODEL_SHA256,
        "embed_dim": EMBED_DIM,
        "fastembed_version": FASTEMBED_VERSION,
        "text_prep_version": TEXT_PREP_VERSION,
        "max_chars": MAX_CHARS,
    }
    blob = json.dumps(components, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest(), components


def default_model_cache_dir() -> Path:
    """Where model files live: ``$FIRSTREAD_MODEL_CACHE`` or ``~/.cache/firstread/models``."""
    env = os.environ.get("FIRSTREAD_MODEL_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "firstread" / "models"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_fastembed_version() -> None:
    """Fail fast if the installed fastembed is not the pinned version."""
    installed = metadata.version("fastembed")
    if installed != FASTEMBED_VERSION:
        raise EmbeddingModelError(
            f"fastembed {installed} is installed but {FASTEMBED_VERSION} is pinned; "
            "a different version can change vectors. Run `uv sync`."
        )


def fetch_model(cache_dir: Path | None = None) -> Path:
    """Download the pinned model revision (if needed) and verify its checksum.

    Returns the directory to pass to fastembed as ``specific_model_path``.
    """
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=EMBED_MODEL_SOURCE,
            revision=EMBED_MODEL_REVISION,
            cache_dir=cache_dir or default_model_cache_dir(),
            allow_patterns=list(EMBED_MODEL_FILES),
        )
    )
    actual = sha256_file(path / EMBED_MODEL_FILE)
    if actual != EMBED_MODEL_SHA256:
        raise EmbeddingModelError(
            f"{EMBED_MODEL_FILE} has SHA256 {actual}, expected {EMBED_MODEL_SHA256}"
        )
    return path


def _fastembed_backend(cache_dir: Path | None, threads: int | None, batch_size: int) -> Backend:
    from fastembed import TextEmbedding

    check_fastembed_version()
    model_dir = fetch_model(cache_dir)
    model = TextEmbedding(
        model_name=EMBED_MODEL,
        cache_dir=str(cache_dir or default_model_cache_dir()),
        threads=threads,
        specific_model_path=str(model_dir),
    )

    def run(texts: list[str]) -> Iterable[NDArray[Any]]:
        return model.embed(texts, batch_size=batch_size)

    return run


class Embedder:
    """Embed already-prepared texts (see :func:`firstread_core.text.article_text`).

    Output is ``float32[n, EMBED_DIM]``, always L2-normalized, so a dot product
    between two vectors is their cosine similarity.

    ``backend`` replaces fastembed (tests use a fake); by default the pinned
    model is loaded on first use.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        threads: int | None = DEFAULT_THREADS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        backend: Backend | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._threads = threads
        self._batch_size = batch_size
        self._backend = backend

    def _get_backend(self) -> Backend:
        if self._backend is None:
            self._backend = _fastembed_backend(self._cache_dir, self._threads, self._batch_size)
        return self._backend

    def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        # Embed shortest-first so each batch pads to similar lengths (much less
        # wasted compute), then restore the caller's order. A text's vector can
        # shift very slightly with how much padding its batch has (see the
        # batch-invariance test), far below anything that changes a ranking.
        order = np.argsort([len(t) for t in texts], kind="stable")
        rows = list(self._get_backend()([texts[i] for i in order]))
        sorted_out = np.asarray(rows, dtype=np.float32)
        if sorted_out.shape != (len(texts), EMBED_DIM):
            raise EmbeddingModelError(
                f"expected shape {(len(texts), EMBED_DIM)}, model returned {sorted_out.shape}"
            )
        out = np.empty_like(sorted_out)
        out[order] = sorted_out
        return l2_normalize(out)
