import json
from pathlib import Path

import numpy as np
import pytest

from firstread_core.topics import TOPIC_IDS
from firstread_ingest import models
from firstread_ingest.models import ModelReleaseError, TopicClassifier, verify_release


def test_valid_release_loads_and_scores(release_dir: tuple[Path, str]) -> None:
    folder, sha = release_dir
    verify_release(folder, sha)
    clf = TopicClassifier.load(folder)
    assert clf.labels == list(TOPIC_IDS)
    assert clf.threshold == 0.5
    p = clf.probabilities(np.random.default_rng(0).normal(size=(4, 384)).astype(np.float32))
    assert p.shape == (4, 13)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)
    assert clf.probabilities(np.zeros((0, 384), np.float32)).shape == (0, 13)


def test_manifest_must_match_the_pin(release_dir: tuple[Path, str]) -> None:
    folder, _ = release_dir
    with pytest.raises(ModelReleaseError, match="pinned"):
        verify_release(folder, "0" * 64)


def test_tampered_classifier_is_refused(release_dir: tuple[Path, str]) -> None:
    folder, sha = release_dir
    (folder / "category_classifier.onnx").write_bytes(b"not the model")
    with pytest.raises(ModelReleaseError, match=r"category_classifier\.onnx"):
        verify_release(folder, sha)


@pytest.mark.parametrize("change", ["embedding", "features", "kind"])
def test_manifest_must_match_packages_core(release_dir: tuple[Path, str], change: str) -> None:
    folder, _ = release_dir
    m = json.loads((folder / "manifest.json").read_text())
    if change == "embedding":
        m["embedding"]["text_prep_version"] = 99
    elif change == "features":
        m["features"]["embed_dim"] = 768
    else:
        m["category_classifier"]["kind"] = "mind_categories"
    (folder / "manifest.json").write_text(json.dumps(m))
    sha = models._sha256(folder / "manifest.json")
    with pytest.raises(ModelReleaseError):
        verify_release(folder, sha)


def test_unpinned_tags_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelReleaseError, match="not pinned"):
        models.download_release("v9.9.9", tmp_path, api=object())


def test_download_uses_the_pinned_revision(
    release_dir: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, sha = release_dir
    monkeypatch.setitem(models.PINNED, "v9.9.9", {"manifest_sha256": sha, "revision": "c" * 40})

    class Hub:
        def __init__(self) -> None:
            self.revisions: set[str] = set()

        def hf_hub_download(self, repo: str, name: str, revision: str, local_dir: Path) -> str:
            self.revisions.add(revision)
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / name).write_bytes((folder / name).read_bytes())
            return str(Path(local_dir) / name)

    hub = Hub()
    out = models.download_release("v9.9.9", tmp_path, api=hub)
    assert hub.revisions == {"c" * 40}
    assert (out / "category_classifier.onnx").is_file()


def test_v0_2_0_is_pinned_by_manifest_hash_and_hub_commit() -> None:
    pin = models.PINNED["v0.2.0"]
    assert pin["manifest_sha256"] == (
        "2f8ebe6385b6fd2813e39968777e4a4ab80170196e95dc70d33502fa7a84fb35"
    )
    assert pin["revision"] == "099201fabb57bde64af046d0c3be2f690adfff40"


@pytest.mark.model
def test_real_v0_2_0_downloads_verifies_and_scores(tmp_path: Path) -> None:
    """The published release, at its pinned Hub commit (cached in CI like the embedder)."""
    folder = models.download_release("v0.2.0")
    clf = TopicClassifier.load(folder)
    assert clf.labels == list(TOPIC_IDS)
    assert clf.threshold == 0.5
    assert clf.version == "v0.2.0"
    p = clf.probabilities(np.random.default_rng(0).normal(size=(2, 384)).astype(np.float32))
    assert p.shape == (2, 13)
