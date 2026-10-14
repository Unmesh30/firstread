import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

from firstread_ml import cli
from firstread_ml.mind import download as dl
from firstread_ml.mind.manifest import FILES, HF_REVISION, LICENSE_URL, MindFile


def _zip_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


GOOD_ZIP = _zip_bytes({"news.tsv": "N1\tnews\n", "behaviors.tsv": "1\tU1\n"})


class FakeHub:
    def __init__(self) -> None:
        self.payloads = {"MINDsmall_train": GOOD_ZIP, "MINDsmall_dev": GOOD_ZIP}
        self.manifest = {
            name: MindFile(name, hashlib.sha256(data).hexdigest(), len(data))
            for name, data in self.payloads.items()
        }
        self.calls: list[str] = []

    def hf_hub_download(self, **kwargs: Any) -> str:
        assert kwargs["revision"] == HF_REVISION
        assert kwargs["repo_type"] == "dataset"
        name = kwargs["filename"].removesuffix(".zip")
        self.calls.append(name)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(self.payloads[name])
        return str(path)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """Point the manifest at tiny fake zips and serve them from a fake hub."""
    import huggingface_hub

    fake = FakeHub()
    monkeypatch.setattr(dl, "FILES", fake.manifest)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    return fake


def test_refuses_without_license_and_prints_terms(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["download", "--variant", "small"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert LICENSE_URL in err
    assert "--accept-license" in err
    assert "non-commercial" in err


def test_downloads_verifies_extracts_and_is_idempotent(tmp_path: Path, hub: FakeHub) -> None:
    out = dl.download("small", accept_license=True, raw_dir=tmp_path)
    assert [p.name for p in out] == ["MINDsmall_train", "MINDsmall_dev"]
    assert (tmp_path / "MINDsmall_train" / "news.tsv").read_text() == "N1\tnews\n"
    assert hub.calls == ["MINDsmall_train", "MINDsmall_dev"]

    dl.download("small", accept_license=True, raw_dir=tmp_path)
    assert len(hub.calls) == 2  # nothing fetched again


def test_checksum_mismatch_deletes_file(
    tmp_path: Path, hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = {k: MindFile(v.name, "0" * 64, v.size) for k, v in hub.manifest.items()}
    monkeypatch.setattr(dl, "FILES", bad)
    with pytest.raises(dl.DownloadError, match="verification"):
        dl.download("small", accept_license=True, raw_dir=tmp_path)
    assert not (tmp_path / "MINDsmall_train.zip").exists()


def test_rejects_zip_slip(tmp_path: Path) -> None:
    data = _zip_bytes({"../evil.tsv": "x", "news.tsv": "", "behaviors.tsv": ""})
    zpath = tmp_path / "MINDsmall_train.zip"
    zpath.write_bytes(data)
    f = MindFile("MINDsmall_train", hashlib.sha256(data).hexdigest(), len(data))
    with pytest.raises(dl.DownloadError, match="unsafe"):
        dl.extract(zpath, f, tmp_path)
    assert not (tmp_path.parent / "evil.tsv").exists()


def test_rejects_zip_missing_members(tmp_path: Path) -> None:
    data = _zip_bytes({"news.tsv": ""})
    zpath = tmp_path / "x.zip"
    zpath.write_bytes(data)
    f = MindFile("MINDsmall_dev", hashlib.sha256(data).hexdigest(), len(data))
    with pytest.raises(dl.DownloadError, match="missing"):
        dl.extract(zpath, f, tmp_path)


def test_manifest_has_no_test_split_and_valid_hashes() -> None:
    assert set(FILES) == {"MINDsmall_train", "MINDsmall_dev", "MINDlarge_train", "MINDlarge_dev"}
    for f in FILES.values():
        assert len(f.sha256) == 64
        int(f.sha256, 16)


def test_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        dl.download("medium", accept_license=True)
