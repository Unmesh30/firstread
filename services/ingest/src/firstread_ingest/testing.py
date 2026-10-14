"""Test support for ingest (imported only by tests; needs the dev dependencies).

No network, a deterministic fake embedder, and a tiny ONNX topic classifier with a
correct release manifest. ``services/ingest/tests/conftest.py`` re-exports the
fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
import respx

from firstread_core.embedding import EMBED_DIM, l2_normalize
from firstread_core.topics import TOPIC_IDS
from firstread_ingest.models import TopicClassifier, expected_blocks
from firstread_ingest.sources.gdelt_domains import DomainCountries, load_config, parse_lookup


@pytest.fixture
def no_network() -> Iterator[respx.MockRouter]:
    """Every HTTP request must be mocked; an unmocked one raises (CI never calls out)."""
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        yield router


def _word_vec(word: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=EMBED_DIM)


def _fake_embed(texts: Sequence[str]) -> np.ndarray:
    """Deterministic bag-of-words embedding: shared words make texts similar."""
    out = np.zeros((len(texts), EMBED_DIM))
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i] += _word_vec(w)
    return l2_normalize(out)


@pytest.fixture
def fake_embed() -> Callable[[Sequence[str]], np.ndarray]:
    return _fake_embed


def build_classifier_onnx(path: Path, weights: np.ndarray, bias: np.ndarray) -> None:
    """embedding [N, 384] -> label [N] int64, probabilities [N, k] (like the release)."""
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    k = weights.shape[1]
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["embedding", "W"], ["logits0"]),
            helper.make_node("Add", ["logits0", "b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
            helper.make_node("ArgMax", ["logits"], ["label"], axis=1, keepdims=0),
        ],
        "tiny_topics",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, EMBED_DIM])],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, k]),
        ],
        initializer=[
            numpy_helper.from_array(weights.astype(np.float32), "W"),
            numpy_helper.from_array(bias.astype(np.float32), "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    path.write_bytes(model.SerializeToString())


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_release(folder: Path, threshold: float = 0.5, seed: int = 0) -> str:
    """A fake v-release folder with a valid manifest; returns the manifest's SHA256."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    build_classifier_onnx(
        folder / "category_classifier.onnx",
        rng.normal(size=(EMBED_DIM, len(TOPIC_IDS))),
        np.zeros(len(TOPIC_IDS)),
    )
    (folder / "labels.json").write_text(json.dumps(list(TOPIC_IDS)))
    manifest = {
        "tag": "v9.9.9",
        "files": {
            n: {"sha256": _sha(folder / n)} for n in ("category_classifier.onnx", "labels.json")
        },
        "category_classifier": {"kind": "topics", "low_confidence_threshold": threshold},
        **expected_blocks(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return _sha(folder / "manifest.json")


@pytest.fixture
def release_dir(tmp_path: Path) -> tuple[Path, str]:
    folder = tmp_path / "release"
    return folder, make_release(folder)


@pytest.fixture
def classifier(release_dir: tuple[Path, str]) -> TopicClassifier:
    return TopicClassifier.load(release_dir[0])


# GDELT outlet placement for the GKG fixture (FIPS codes, like the real lookup).
LOOKUP = """domain,countrycode,cnt
wisn.com,US,900
wisn.com,UK,10
fox4news.com,US,500
mashable.com,US,300
gazetteseries.co.uk,UK,400
dunmowbroadcast.co.uk,UK,90
nzherald.co.nz,NZ,800
thehindu.com,IN,700
jpost.com,IS,600
jpost.com,US,200
iheart.com,US,999
themarketsdaily.com,US,999
"""


@pytest.fixture
def places() -> DomainCountries:
    """GDELT placement from a tiny lookup plus the real overrides/blocklist."""
    overrides, blocklist = load_config()
    return DomainCountries(parse_lookup(LOOKUP), overrides, blocklist)
