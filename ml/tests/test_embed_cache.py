import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from firstread_core import embedding, text
from firstread_core.embedding import Embedder
from firstread_ml import embed_cache
from firstread_ml.testing.synthetic import hash_backend


class CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: list[str]):  # type: ignore[no-untyped-def]
        self.calls += 1
        return hash_backend(texts)


@pytest.fixture
def news() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nid": ["N1", "N2", "N3"],
            "title": ["topic1 a b", "topic2 c d", "topic1 e f"],
            "abstract": ["x", "", "y"],
        }
    )


def build(news: pl.DataFrame, root: Path, backend: CountingBackend) -> np.ndarray:
    return embed_cache.load_or_build("t", news, Embedder(backend=backend), root)


def test_builds_then_reuses(news: pl.DataFrame, tmp_path: Path) -> None:
    b = CountingBackend()
    first = build(news, tmp_path, b)
    assert first.shape == (3, embedding.EMBED_DIM)
    second = build(news, tmp_path, b)
    assert b.calls == 1
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("module", "attr", "value"),
    [
        (text, "TEXT_PREP_VERSION", 999),
        (text, "MAX_CHARS", 5),
        (embedding, "EMBED_MODEL_REVISION", "0" * 40),
        (embedding, "FASTEMBED_VERSION", "9.9.9"),
    ],
)
def test_key_component_change_rebuilds(
    news: pl.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: object,
    attr: str,
    value: object,
) -> None:
    b = CountingBackend()
    build(news, tmp_path, b)
    monkeypatch.setattr(module, attr, value)
    # embedding_cache_key reads the constants it imported, so patch both places.
    if module is text:
        monkeypatch.setattr(embedding, attr, value)
    build(news, tmp_path, b)
    assert b.calls == 2
    assert "stale" in capsys.readouterr().out


def test_changed_article_ids_rebuild(news: pl.DataFrame, tmp_path: Path) -> None:
    b = CountingBackend()
    build(news, tmp_path, b)
    build(news.with_columns(pl.Series("nid", ["N1", "N2", "N9"])), tmp_path, b)
    assert b.calls == 2


@pytest.mark.parametrize("tamper", ["edit", "garbage", "delete"])
def test_tampered_sidecar_rebuilds(news: pl.DataFrame, tmp_path: Path, tamper: str) -> None:
    b = CountingBackend()
    build(news, tmp_path, b)
    (sidecar,) = tmp_path.glob("cache/emb/t/*/key.json")
    if tamper == "edit":
        data = json.loads(sidecar.read_text())
        data["components"]["text_prep_version"] = -1
        sidecar.write_text(json.dumps(data))
    elif tamper == "garbage":
        sidecar.write_text("{not json")
    else:
        sidecar.unlink()
    build(news, tmp_path, b)
    assert b.calls == 2
    assert json.loads(sidecar.read_text())["key"] == embedding.embedding_cache_key()[0]
