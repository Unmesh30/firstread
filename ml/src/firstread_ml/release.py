"""Build, upload and verify a versioned model release (SPEC section 7.7, ADR 0005).

``build``: export the seed-0 MIND-large user encoder and the category classifier
to ONNX, re-check parity on the real weights, run the real-data eval gate,
benchmark ONNX Runtime latency, and write ``data/release/<version>/`` with a
SHA256 ``manifest.json``. Two modes:

* **dry run** (any branch): writes ``data/release/dry-run/``; the manifest's
  ``git_commit`` is ``dry-run:<sha>`` and ``upload`` refuses it.
* **real**: only on a clean ``main`` that equals ``origin/main``. The manifest
  records ``HEAD``, which is the commit we tag ``v<version>`` afterwards
  (``verify-tag`` checks it). A PR branch's SHA would not survive a rebase-merge.

A build never modifies a tracked file: everything it writes is under
gitignored ``data/release/``, and it fails if ``git status`` changed.

``upload``: push the folder to the Hugging Face Hub, tag it, and re-download
every file at the tag to check its SHA256. Only MIND-derived *weights and
aggregates* are released, never MIND rows, text or embedding caches (allowlist).
"""

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import skl2onnx
import sklearn
import torch
import yaml
from numpy.typing import NDArray

import firstread_core
from firstread_core.embedding import (
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_MODEL_FILE,
    EMBED_MODEL_REVISION,
    EMBED_MODEL_SOURCE,
    FASTEMBED_VERSION,
    embedding_cache_key,
)
from firstread_core.features import HISTORY_LEN, gather_history, history_index_matrix
from firstread_core.text import TEXT_PREP_VERSION
from firstread_ml.config import NRMSLiteConfig, load_config
from firstread_ml.export import parity
from firstread_ml.export.classifier import export_classifier
from firstread_ml.export.user_encoder import OPSET, export_user_encoder
from firstread_ml.mind.manifest import HF_REVISION
from firstread_ml.models.scorer import DEFAULT_CONFIG, load_checkpoint
from firstread_ml.paths import REPO_ROOT, data_dir, reports_dir

HF_REPO = "Unmesh30/firstread-models"
SHIPPED_SEED = 0  # the config's pre-declared seed; all seeds are reported in metrics.json
FILES = (
    "README.md",
    "category_classifier.onnx",
    "config.yaml",
    "labels.json",
    "manifest.json",
    "metrics.json",
    "population.npy",
    "user_encoder.onnx",
)
DRY_RUN_DIR = "dry-run"
REPORT_SLICES = ("overall", "cold_candidates", "cold_users")


class ReleaseError(RuntimeError):
    pass


# ---------------------------------------------------------------- git helpers


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def git_status(repo: Path) -> str:
    return git(repo, "status", "--porcelain")


def require_releasable_main(repo: Path) -> str:
    """Return HEAD if the tree is clean, on ``main``, and equal to ``origin/main``."""
    if git_status(repo):
        raise ReleaseError("working tree is not clean; commit or stash first")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        raise ReleaseError(f"real builds run on main only (on {branch!r}); use --dry-run")
    git(repo, "fetch", "--quiet", "origin", "main")
    head, remote = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "origin/main")
    if head != remote:
        raise ReleaseError(f"HEAD {head[:12]} != origin/main {remote[:12]}; pull or push first")
    return head


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Sources:
    """Everything a build reads. Tests point these at synthetic fixtures."""

    checkpoint: Path  # NRMS-lite checkpoint dir (model.pt, config.yaml, key.json)
    classifier_dir: Path  # model.joblib + labels.json
    results: Path  # committed dev results (aggregates)
    classifier_report: Path  # committed classifier report (aggregates)
    model_card: Path  # docs/model-card.md, shipped as README.md
    emb: NDArray[np.float32]  # embedding table for real-history parity
    histories: list[NDArray[np.int32]]  # real click histories (article indices)
    gate: Callable[[Path, Path], tuple[int, dict[str, Any]]]  # (onnx, out) -> (code, result)


def find_checkpoint(model_dir: Path, cfg: NRMSLiteConfig, embedding_id: str) -> Path:
    """The single checkpoint trained with ``cfg`` on the current embeddings."""
    want = cfg.model_dump()
    hits = []
    for key_file in sorted(model_dir.glob("*/key.json")):
        key = json.loads(key_file.read_text())
        if key.get("config") == want and key.get("embedding_cache_id") == embedding_id:
            hits.append(key_file.parent)
    if len(hits) != 1:
        raise ReleaseError(f"expected 1 checkpoint for seed {cfg.seed} in {model_dir}, got {hits}")
    return hits[0]


def default_sources(data_root: Path | None = None) -> Sources:
    """The real inputs: seed-0 MIND-large dev checkpoint, large classifier, MIND-small gate."""
    from firstread_ml.embed_cache import load_or_build
    from firstread_ml.eval import gate_real
    from firstread_ml.eval.harness import load_eval_data
    from firstread_ml.mind.load import load_news, prepare
    from firstread_ml.paths import mind_raw_dir, processed_dir

    root = data_root or data_dir()
    cfg = load_config(DEFAULT_CONFIG).model_copy(update={"seed": SHIPPED_SEED})
    ckpt = find_checkpoint(
        root / "models" / "large" / "dev" / "nrms_lite", cfg, embedding_cache_key()[0][:16]
    )
    v = gate_real.VARIANT
    proc = prepare(v, raw_dir=mind_raw_dir(data_root), out_dir=processed_dir(v, data_root))
    test = load_eval_data(v, "dev", data_root).test
    test = test.take(gate_real.subset_rows(test.n))
    return Sources(
        checkpoint=ckpt,
        classifier_dir=root / "models" / "large" / "classifier",
        results=reports_dir() / "results-mind-large-dev.json",
        classifier_report=reports_dir() / "classifier-mind-large.json",
        model_card=REPO_ROOT / "docs" / "model-card.md",
        emb=load_or_build(v, load_news(v, proc), None, data_root),
        histories=test.histories(slice(0, 512)),
        gate=lambda onnx_path, out: gate_real.main(onnx_path, out, data_root=data_root),
    )


# ---------------------------------------------------------------- pieces


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def machine() -> dict[str, str]:
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            chip = sysctl("machdep.cpu.brand_string") or chip
    return {"chip": chip, "os": platform.platform(), "cpus": str(os.cpu_count())}


def sysctl(name: str) -> str:
    out = subprocess.run(  # noqa: S603
        ["/usr/sbin/sysctl", "-n", name], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def benchmark(
    onnx_path: Path,
    batches: tuple[int, ...] = (1, 32),
    threads: tuple[int, ...] = (1, 4),
    warmup: int = 50,
    runs: int = 1000,
) -> list[dict[str, Any]]:
    """ONNX Runtime CPU latency of the user encoder with full (50-click) histories."""
    rows = []
    rng = np.random.default_rng(0)
    for t in threads:
        sess = parity.session(onnx_path, threads=t)
        for b in batches:
            hist, mask = parity.histories([HISTORY_LEN] * b, rng)
            feed = {"hist": hist, "mask": mask}
            for _ in range(warmup):
                sess.run(None, feed)
            times = np.empty(runs)
            for i in range(runs):
                t0 = time.perf_counter()
                sess.run(None, feed)
                times[i] = time.perf_counter() - t0
            ms = times * 1e3
            rows.append(
                {
                    "batch": b,
                    "threads": t,
                    "runs": runs,
                    "p50_ms": round(float(np.percentile(ms, 50)), 3),
                    "p95_ms": round(float(np.percentile(ms, 95)), 3),
                    "mean_ms": round(float(ms.mean()), 3),
                }
            )
            print(
                f"  latency batch={b:>2} threads={t}: p50 {rows[-1]['p50_ms']:.3f} ms, "
                f"p95 {rows[-1]['p95_ms']:.3f} ms"
            )
    return rows


def real_history_cases(
    emb: NDArray[np.float32], histories: list[NDArray[np.int32]]
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.bool_]]]:
    idx, mask = history_index_matrix(histories, HISTORY_LEN)
    hist = gather_history(emb, idx, mask)
    return {"real_histories": (hist, mask)}


def dev_metrics(results: dict[str, Any], classifier: dict[str, Any]) -> dict[str, Any]:
    """Aggregates copied from the committed reports (no MIND rows or text)."""
    res = results["results"]
    seeds = results["nrms_lite"]["seed_models"]
    keep = [*seeds, "mean_history", "popularity", "recency", "random"]
    return {
        "dataset": "MIND-large dev",
        "shipped_seed": SHIPPED_SEED,
        "shipped_model": f"nrms_lite_s{SHIPPED_SEED}",
        "slices": {sl: results["slices"][sl]["n_impressions"] for sl in REPORT_SLICES},
        "results": {m: {sl: res[m][sl] for sl in REPORT_SLICES} for m in keep},
        "seed_summary": {sl: results["nrms_lite"]["summary"][sl] for sl in REPORT_SLICES},
        "bootstrap": results.get("bootstrap"),
        "classifier": {
            "dataset": "MIND-large, articles only in dev news.tsv",
            "accuracy": classifier["test"]["accuracy"],
            "macro_f1": classifier["test"]["macro_f1"],
            "majority_class_accuracy": classifier["test"]["majority_class_accuracy"],
            "n_test": classifier["n_test_dev_only"],
            "labels": classifier["labels"],
            "dropped_labels": classifier["dropped_labels"],
        },
    }


def check_allowlist(release_dir: Path) -> None:
    found = sorted(p.name for p in release_dir.iterdir())
    if found != sorted(FILES):
        extra = sorted(set(found) - set(FILES))
        missing = sorted(set(FILES) - set(found))
        raise ReleaseError(f"release folder mismatch: extra {extra}, missing {missing}")


def verify_folder(release_dir: Path) -> dict[str, Any]:
    """Allowlist + every SHA256 in the manifest. Returns the manifest."""
    check_allowlist(release_dir)
    manifest: dict[str, Any] = json.loads((release_dir / "manifest.json").read_text())
    return verify_files(release_dir, manifest)


def verify_files(folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    for name, meta in manifest["files"].items():
        got = sha256(folder / name)
        if got != meta["sha256"]:
            raise ReleaseError(f"{name}: sha256 {got} != manifest {meta['sha256']}")
    return manifest


# ---------------------------------------------------------------- build


def build(
    version: str,
    sources: Sources,
    dry_run: bool,
    repo: Path = REPO_ROOT,
    out_root: Path | None = None,
    bench: Callable[[Path], list[dict[str, Any]]] = benchmark,
) -> Path:
    """Build the release folder; return its path. Raises ``ReleaseError`` on any failed check."""
    status_before = git_status(repo)
    if dry_run:
        commit = f"dry-run:{git(repo, 'rev-parse', 'HEAD')}"
    else:
        commit = require_releasable_main(repo)
    root = out_root or data_dir() / "release"
    name = DRY_RUN_DIR if dry_run else f"v{version}"
    out, work = root / name, root / f"{name}-build"
    for d in (out, work):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    print(f"building release v{version} ({'dry run' if dry_run else commit[:12]}) -> {out}")

    model, cfg = load_checkpoint(sources.checkpoint, torch.device("cpu"))
    if cfg.seed != SHIPPED_SEED:
        raise ReleaseError(f"checkpoint seed {cfg.seed} != shipped seed {SHIPPED_SEED}")
    clf = joblib.load(sources.classifier_dir / "model.joblib")
    labels = json.loads((sources.classifier_dir / "labels.json").read_text())
    meta = {"firstread_release": version, "hf_repo": HF_REPO}
    enc = export_user_encoder(model, out / "user_encoder.onnx", meta)
    cls = export_classifier(clf, out / "category_classifier.onnx", meta | {"labels": "labels.json"})

    print("parity (PyTorch/sklearn vs ONNX Runtime, max abs diff):")
    enc_parity = parity.check_user_encoder(model, enc)
    real_parity = parity.check_user_encoder(
        model, enc, parity.default_cases() | real_history_cases(sources.emb, sources.histories)
    )
    for case, diff in real_parity.max_abs_diff.items():
        print(f"  user_encoder {case}: {diff:.2e}")
    print(f"  user_encoder batch invariance: {real_parity.batch_invariance:.2e}")
    x = sources.emb[np.random.default_rng(0).choice(len(sources.emb), 2048)]
    cls_parity = parity.check_classifier(clf, cls, x)
    print(
        f"  classifier: {cls_parity['max_abs_diff']:.2e}, "
        f"label mismatches {cls_parity['label_mismatches']}/{cls_parity['n']}"
    )
    if not (enc_parity.ok and real_parity.ok and cls_parity["ok"]):
        raise ReleaseError(f"parity above {parity.TOLERANCE}")

    code, gate_result = sources.gate(enc, work / "eval_gate.json")
    if code != 0:
        raise ReleaseError(f"real-data eval gate failed: {gate_result['failures']}")

    print("latency (ONNX Runtime CPU, user encoder, 50-click histories):")
    latency = {"machine": machine(), "onnxruntime": ort.__version__, "rows": bench(enc)}

    (out / "labels.json").write_text(json.dumps(labels) + "\n")
    np.save(out / "population.npy", model.population.numpy().astype(np.float32))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    metrics = dev_metrics(
        json.loads(sources.results.read_text()), json.loads(sources.classifier_report.read_text())
    ) | {
        "parity": {
            "user_encoder_synthetic": enc_parity.as_dict(),
            "user_encoder_real": real_parity.as_dict(),
            "classifier": cls_parity,
        },
        "eval_gate_real": {k: gate_result[k] for k in ("ndcg@10", "meta", "slices", "passed")},
        "latency": latency,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n")
    shutil.copyfile(sources.model_card, out / "README.md")

    manifest = {
        "name": "firstread-models",
        "version": version,
        "tag": f"v{version}",
        "hf_repo": HF_REPO,
        "git_commit": commit,
        "dry_run": dry_run,
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {
            n: {"sha256": sha256(out / n), "bytes": (out / n).stat().st_size}
            for n in FILES
            if n != "manifest.json"
        },
        "user_encoder": {
            "file": "user_encoder.onnx",
            "inputs": {
                "hist": ["batch", HISTORY_LEN, EMBED_DIM],
                "mask": ["batch", HISTORY_LEN],
            },
            "input_dtypes": {"hist": "float32", "mask": "bool"},
            "output": {"user_vec": ["batch", EMBED_DIM]},
            "opset": OPSET,
            "score_scale": float(model.scale.detach()),
            "alpha": float(model.alpha.detach()),
            "seed": cfg.seed,
            "checkpoint_id": sources.checkpoint.name,
            "trained_on": "MIND-large train (all days)",
        },
        "category_classifier": {
            "file": "category_classifier.onnx",
            "input": {"embedding": ["batch", EMBED_DIM]},
            "outputs": {"label": ["batch"], "probabilities": ["batch", len(labels)]},
            "labels": "labels.json",
            "n_labels": len(labels),
        },
        "features": {"history_len": HISTORY_LEN, "embed_dim": EMBED_DIM, "right_aligned": True},
        "embedding": {
            "model": EMBED_MODEL,
            "source": EMBED_MODEL_SOURCE,
            "revision": EMBED_MODEL_REVISION,
            "file": EMBED_MODEL_FILE,
            "fastembed": FASTEMBED_VERSION,
            "text_prep_version": TEXT_PREP_VERSION,
            "l2_normalized": True,
        },
        "dataset": {"name": "MIND", "revision": HF_REVISION, "license": "MSR License Terms"},
        "license": "other (MIND-derived weights, non-commercial; see README.md)",
        "versions": {
            "firstread_core": firstread_core.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "scikit-learn": sklearn.__version__,
            "skl2onnx": skl2onnx.__version__,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    verify_folder(out)
    (work / "build.json").write_text(json.dumps({"latency": latency, "commit": commit}, indent=1))
    if git_status(repo) != status_before:
        raise ReleaseError("the build modified tracked files; it must only write under data/")
    print(f"OK: {len(FILES)} files, manifest verified, git status unchanged -> {out}")
    return out


# ---------------------------------------------------------------- upload


def _hub(api: Any) -> Any:
    if api is not None:
        return api
    if not os.environ.get("HF_TOKEN"):
        raise ReleaseError("HF_TOKEN is not set (put a write token in .env; never commit it)")
    from huggingface_hub import HfApi

    hub = HfApi()  # reads HF_TOKEN from the environment; never printed
    return hub


def tag_exists(api: Any, tag: str) -> bool:
    return any(t.name == tag for t in api.list_repo_refs(HF_REPO).tags)


def upload(
    version: str, release_dir: Path, api: Any = None, repo: Path = REPO_ROOT
) -> dict[str, Any]:
    """Upload a real (non-dry-run) build of the current clean HEAD, tag it, verify it."""
    manifest = verify_folder(release_dir)
    tag = f"v{version}"
    if manifest["dry_run"] or manifest["version"] != version:
        raise ReleaseError("refusing to upload a dry run or a build of another version")
    if git_status(repo):
        raise ReleaseError("working tree is not clean")
    head = git(repo, "rev-parse", "HEAD")
    if manifest["git_commit"] != head:
        raise ReleaseError(f"manifest commit {manifest['git_commit'][:12]} != HEAD {head[:12]}")
    hub = _hub(api)
    hub.create_repo(HF_REPO, private=False, exist_ok=True)
    if tag_exists(hub, tag):
        raise ReleaseError(f"{HF_REPO} already has tag {tag}; releases are immutable")
    commit = hub.upload_folder(
        repo_id=HF_REPO, folder_path=release_dir, commit_message=f"Release {tag}"
    )
    hub.create_tag(HF_REPO, tag=tag, revision=commit.oid, tag_message=f"FirstRead models {tag}")
    print(f"uploaded {HF_REPO}@{tag} (commit {commit.oid[:12]})")
    return verify_remote(version, hub, manifest)


def verify_remote(
    version: str, api: Any = None, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Download every file at the tag and check it against the published manifest."""
    hub = api or _public_hub()
    tag = f"v{version}"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        get = lambda n: Path(  # noqa: E731
            hub.hf_hub_download(HF_REPO, n, revision=tag, local_dir=folder)
        )
        manifest: dict[str, Any] = json.loads(get("manifest.json").read_text())
        if expected is not None and manifest != expected:
            raise ReleaseError("the manifest on the Hub differs from the local one")
        for name in manifest["files"]:
            get(name)
        verify_files(folder, manifest)
    for name, meta in sorted(manifest["files"].items()):
        print(f"  {meta['sha256']}  {name}")
    print(f"OK: {len(manifest['files'])} files at {HF_REPO}@{tag} match manifest.json")
    return manifest


def _public_hub() -> Any:
    from huggingface_hub import HfApi

    hub = HfApi()
    return hub


def verify_tag(version: str, api: Any = None, repo: Path = REPO_ROOT) -> str:
    """The Hub manifest's ``git_commit`` must be the commit git tag ``v<version>`` points to."""
    manifest = verify_remote(version, api)
    tagged = git(repo, "rev-parse", f"v{version}^{{commit}}")
    if manifest["git_commit"] != tagged:
        raise ReleaseError(
            f"manifest git_commit {manifest['git_commit'][:12]} != v{version} {tagged[:12]}"
        )
    print(f"OK: git tag v{version} = {tagged} = manifest git_commit")
    return tagged
