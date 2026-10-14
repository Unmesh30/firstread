"""Cached article embeddings for a MIND variant.

Vectors live in ``data/cache/emb/<variant>/<key[:16]>/`` with a ``key.json``
sidecar. The key (``firstread_core.embedding.embedding_cache_key``) covers the
model, its revision, the fastembed version and the text-prep settings, and the
sidecar also records a hash of the article ids. If anything differs, the cache
is rebuilt; stale vectors are never reused (that would be train/serve skew).
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_DIM, Embedder, embedding_cache_key
from firstread_core.text import article_text
from firstread_ml.paths import emb_cache_dir

CHUNK = 4096


def _expected_sidecar(news: pl.DataFrame) -> dict[str, Any]:
    key, components = embedding_cache_key()
    ids = "\n".join(news["nid"].to_list()).encode()
    return {
        "key": key,
        "components": components,
        "n_articles": news.height,
        "article_ids_sha256": hashlib.sha256(ids).hexdigest(),
    }


def _explain_miss(variant_dir: Path, expected: dict[str, Any]) -> str:
    """Say which key component changed vs. any older cache for this variant."""
    for old in sorted(variant_dir.glob("*/key.json")):
        try:
            prev = json.loads(old.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        diffs = [
            f"{k}: {prev.get('components', {}).get(k)!r} -> {v!r}"
            for k, v in expected["components"].items()
            if prev.get("components", {}).get(k) != v
        ]
        if prev.get("article_ids_sha256") != expected["article_ids_sha256"]:
            diffs.append("article ids changed")
        if diffs:
            return f"cache {old.parent.name} is stale ({'; '.join(diffs)})"
    return "no matching cache"


def load_or_build(
    variant: str,
    news: pl.DataFrame,
    embedder: Embedder | None = None,
    root: Path | None = None,
) -> NDArray[np.float32]:
    """Return ``float32[n_articles, EMBED_DIM]`` aligned with ``news['nidx']``."""
    variant_dir = emb_cache_dir(variant, root)
    expected = _expected_sidecar(news)
    cache = variant_dir / expected["key"][:16]
    sidecar, vectors = cache / "key.json", cache / "emb.npy"
    if sidecar.is_file() and vectors.is_file():
        try:
            current = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            current = None
        if current == expected:
            emb: NDArray[np.float32] = np.load(vectors)
            if emb.shape == (news.height, EMBED_DIM):
                return emb
        print(f"embedding cache {cache.name}: sidecar mismatch, rebuilding")
    else:
        print(f"embedding cache miss: {_explain_miss(variant_dir, expected)}")

    embedder = embedder or Embedder()
    texts = [
        article_text(t, a)
        for t, a in zip(news["title"].to_list(), news["abstract"].to_list(), strict=True)
    ]
    parts = []
    for i in range(0, len(texts), CHUNK):
        parts.append(embedder.embed(texts[i : i + CHUNK]))
        print(f"  embedded {min(i + CHUNK, len(texts))}/{len(texts)}", flush=True)
    emb = np.concatenate(parts) if parts else np.zeros((0, EMBED_DIM), np.float32)

    tmp = variant_dir / f".{cache.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "emb.npy", emb)
    (tmp / "key.json").write_text(json.dumps(expected, indent=1) + "\n")  # written last
    shutil.rmtree(cache, ignore_errors=True)
    tmp.rename(cache)
    return emb
