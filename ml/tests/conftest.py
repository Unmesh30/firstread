from pathlib import Path

import pytest

from firstread_core.embedding import Embedder
from firstread_ml.mind.load import prepare
from firstread_ml.paths import processed_dir
from firstread_ml.testing import synthetic


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data root with prepared synthetic MIND-format data."""
    root = tmp_path_factory.mktemp("data")
    raw = root / "mind" / "raw"
    synthetic.generate(raw, seed=0)
    prepare(synthetic.VARIANT, raw_dir=raw, out_dir=processed_dir(synthetic.VARIANT, root))
    return root


@pytest.fixture
def hash_embedder() -> Embedder:
    return Embedder(backend=synthetic.hash_backend)
