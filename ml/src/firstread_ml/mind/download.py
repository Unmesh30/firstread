"""Download MIND from the pinned Hugging Face mirror, verify it and extract it.

Requires an explicit ``--accept-license``: MIND is licensed for non-commercial
research only and may not be redistributed. Files go to ``data/mind/raw``
(gitignored). Re-running is cheap: verified zips and extracted splits are skipped.
"""

import shutil
import sys
import zipfile
from pathlib import Path

from firstread_core.embedding import sha256_file
from firstread_ml.mind.manifest import (
    FILES,
    HF_REPO,
    HF_REVISION,
    LICENSE_URL,
    VARIANTS,
    MindFile,
    split_name,
)
from firstread_ml.paths import mind_raw_dir

LICENSE_NOTICE = f"""\
MIND (Microsoft News Dataset) is released under the Microsoft Research License Terms:
  {LICENSE_URL}

In short (read the full terms before accepting):
  * non-commercial research use only (public demonstrations count as research);
  * the data may not be redistributed: never commit, upload or share MIND files,
    MIND text or caches derived from it;
  * results may be published only if no material portion of the data is included.

FirstRead uses MIND only to train and evaluate models; MIND articles are never
shown in the app. Re-run with --accept-license to accept these terms and download.
"""

EXPECTED_MEMBERS = {"behaviors.tsv", "news.tsv"}


class DownloadError(RuntimeError):
    """A file failed verification or extraction."""


def _verify(path: Path, f: MindFile) -> bool:
    return path.is_file() and path.stat().st_size == f.size and sha256_file(path) == f.sha256


def fetch_zip(f: MindFile, raw_dir: Path) -> Path:
    """Download one zip (unless a verified copy exists) and check its SHA256."""
    from huggingface_hub import hf_hub_download

    dest = raw_dir / f.filename
    if _verify(dest, f):
        print(f"  {f.filename}: already downloaded and verified")
        return dest
    print(f"  {f.filename}: downloading {f.size / 1e6:.0f} MB from {HF_REPO}@{HF_REVISION[:8]}")
    path = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f.filename,
            revision=HF_REVISION,
            local_dir=raw_dir,
        )
    )
    if not _verify(path, f):
        path.unlink(missing_ok=True)
        raise DownloadError(f"{f.filename} failed SHA256/size verification and was deleted")
    print(f"  {f.filename}: SHA256 verified")
    return path


def extract(zip_path: Path, f: MindFile, raw_dir: Path) -> Path:
    """Extract into ``raw_dir/<name>/`` atomically (via a temp dir and a rename)."""
    out = raw_dir / f.name
    marker = out / ".verified"
    if marker.is_file() and marker.read_text().strip() == f.sha256:
        print(f"  {f.name}: already extracted")
        return out
    tmp = raw_dir / f".{f.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (tmp / member).resolve()
            if not target.is_relative_to(tmp.resolve()):
                raise DownloadError(f"unsafe path in {zip_path.name}: {member}")
        zf.extractall(tmp)
    missing = EXPECTED_MEMBERS - {p.name for p in tmp.iterdir()}
    if missing:
        shutil.rmtree(tmp)
        raise DownloadError(f"{zip_path.name} is missing {sorted(missing)}")
    (tmp / ".verified").write_text(f.sha256 + "\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"  {f.name}: extracted to {out}")
    return out


def download(variant: str, accept_license: bool, raw_dir: Path | None = None) -> list[Path]:
    """Download, verify and extract the train and dev splits of ``variant``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if not accept_license:
        print(LICENSE_NOTICE, file=sys.stderr)
        raise SystemExit(2)
    raw_dir = raw_dir or mind_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"MIND-{variant} -> {raw_dir} (license: {LICENSE_URL})")
    out = []
    for split in VARIANTS[variant]:
        f = FILES[split_name(variant, split)]
        out.append(extract(fetch_zip(f, raw_dir), f, raw_dir))
    return out
