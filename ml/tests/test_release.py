"""Release build, allowlist, manifest, clean tree, upload and tag checks (synthetic; no network)."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from firstread_core.embedding import EMBED_DIM, Embedder
from firstread_ml import release
from firstread_ml.config import ModelConfig, NRMSLiteConfig
from firstread_ml.embed_cache import load_or_build
from firstread_ml.eval import gate_real
from firstread_ml.eval.harness import load_eval_data
from firstread_ml.mind.load import load_news
from firstread_ml.models.nrms_lite import NRMSLite
from firstread_ml.models.scorer import save_checkpoint
from firstread_ml.paths import processed_dir, reports_dir
from firstread_ml.testing import synthetic

VERSION = "0.0.1"


def git(repo: Path, *args: str) -> str:
    cmd = ["git", *args]  # trusted, test-only arguments
    out = subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True)  # noqa: S603
    return out.stdout.strip()


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "test")
    git(path, "config", "commit.gpgsign", "false")
    return path


def commit_all(repo: Path, msg: str = "fixture") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def no_must_beat(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny synthetic model isn't expected to beat mean_history; these tests
    exercise the plumbing. The rule itself is tested in ``test_gate_real_rules``."""
    monkeypatch.setattr(gate_real, "MUST_BEAT_ON", ())


@pytest.fixture
def setup(
    tmp_path: Path, synthetic_root: Path, hash_embedder: Embedder, no_must_beat: None
) -> dict[str, Any]:
    """A temp git repo (reports, card, gate reference committed; data/ ignored) + sources."""
    repo = init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("data/\n")
    rep = repo / "ml" / "reports"
    rep.mkdir(parents=True)
    for name in ("results-mind-large-dev.json", "classifier-mind-large.json"):
        shutil.copyfile(reports_dir() / name, rep / name)
    card = repo / "docs" / "model-card.md"
    card.parent.mkdir()
    card.write_text("---\nlicense: other\n---\n# test card\n")

    v = synthetic.VARIANT
    emb = load_or_build(
        v, load_news(v, processed_dir(v, synthetic_root)), hash_embedder, synthetic_root
    )
    torch.manual_seed(0)
    cfg = NRMSLiteConfig(seed=0, model=ModelConfig(heads=4, query_dim=16))
    model = NRMSLite(cfg.model, torch.from_numpy(emb.mean(axis=0)))
    with torch.no_grad():
        model.alpha.fill_(0.3)
    ckpt = repo / "data" / "models" / "ckpt0"
    save_checkpoint(model, cfg, ckpt)
    cls_dir = repo / "data" / "models" / "classifier"
    cls_dir.mkdir(parents=True)
    y = np.arange(len(emb)) % 3
    joblib.dump(LogisticRegression(max_iter=500).fit(emb, y), cls_dir / "model.joblib")
    (cls_dir / "labels.json").write_text(json.dumps(["a", "b", "c"]))

    reference = rep / "eval-gate-real.json"

    def gate(onnx_path: Path, out: Path, update: bool = False) -> tuple[int, dict[str, Any]]:
        return gate_real.main(
            onnx_path, out, update=update, reference_path=reference, variant=v,
            data_root=synthetic_root, embedder=hash_embedder,
        )  # fmt: skip

    sources = release.Sources(
        checkpoint=ckpt,
        classifier_dir=cls_dir,
        results=rep / "results-mind-large-dev.json",
        classifier_report=rep / "classifier-mind-large.json",
        model_card=card,
        emb=emb,
        histories=load_eval_data(v, "dev", synthetic_root).test.histories(slice(0, 64)),
        gate=gate,
    )
    return {"repo": repo, "sources": sources, "gate": gate, "reference": reference}


def fast_bench(p: Path) -> list[dict[str, Any]]:
    return release.benchmark(p, batches=(1, 4), threads=(1,), warmup=1, runs=5)


def build(setup: dict[str, Any], dry_run: bool = True) -> Path:
    repo = setup["repo"]
    return release.build(
        VERSION, setup["sources"], dry_run=dry_run, repo=repo,
        out_root=repo / "data" / "release", bench=fast_bench,
    )  # fmt: skip


@pytest.fixture
def built(setup: dict[str, Any]) -> dict[str, Any]:
    """Reference written once (``--update``) and committed, then a dry-run build."""
    repo = setup["repo"]
    code, _ = setup["gate"](_export_tmp(setup), repo / "data" / "tmp-gate.json", update=True)
    assert code == 0
    commit_all(repo, "gate reference")
    return setup | {"out": build(setup)}


def _export_tmp(setup: dict[str, Any]) -> Path:
    from firstread_ml.export.user_encoder import export_user_encoder
    from firstread_ml.models.scorer import load_checkpoint

    model, _ = load_checkpoint(setup["sources"].checkpoint, torch.device("cpu"))
    return export_user_encoder(model, setup["repo"] / "data" / "tmp" / "u.onnx")


def test_build_writes_exactly_the_allowlisted_files(built: dict[str, Any]) -> None:
    out = built["out"]
    assert sorted(p.name for p in out.iterdir()) == sorted(release.FILES)
    manifest = release.verify_folder(out)
    assert manifest["dry_run"] is True
    assert manifest["git_commit"] == f"dry-run:{git(built['repo'], 'rev-parse', 'HEAD')}"
    assert set(manifest["files"]) == set(release.FILES) - {"manifest.json"}
    assert manifest["features"]["history_len"] == 50
    assert manifest["user_encoder"]["seed"] == 0
    assert (out / "README.md").read_text() == built["sources"].model_card.read_text()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["parity"]["user_encoder_real"]["ok"]
    assert metrics["eval_gate_real"]["passed"]
    assert {r["batch"] for r in metrics["latency"]["rows"]} == {1, 4}
    pop = np.load(out / "population.npy")
    assert pop.shape == (EMBED_DIM,)
    assert np.linalg.norm(pop) == pytest.approx(1.0, abs=1e-5)


def test_build_leaves_git_status_clean(built: dict[str, Any]) -> None:
    repo = built["repo"]
    assert git(repo, "status", "--porcelain") == ""
    assert built["out"].is_relative_to(repo / "data" / "release")
    assert (repo / "data" / "release" / "dry-run-build" / "eval_gate.json").is_file()


def test_build_fails_if_a_tracked_file_changes(built: dict[str, Any]) -> None:
    def gate_that_writes(onnx_path: Path, out: Path) -> tuple[int, dict[str, Any]]:
        code, result = built["gate"](onnx_path, out)
        built["reference"].write_text("{}\n")  # a tracked file
        return code, result

    s = built["sources"]
    with pytest.raises(release.ReleaseError, match="modified tracked files"):
        release.build(
            VERSION, release.Sources(**{**s.__dict__, "gate": gate_that_writes}), dry_run=True,
            repo=built["repo"], out_root=built["repo"] / "data" / "release", bench=fast_bench,
        )  # fmt: skip


def test_extra_file_fails_the_allowlist(built: dict[str, Any]) -> None:
    (built["out"] / "news.tsv").write_text("leak")
    with pytest.raises(release.ReleaseError, match="extra"):
        release.verify_folder(built["out"])


def test_tampered_file_fails_verification(built: dict[str, Any]) -> None:
    (built["out"] / "labels.json").write_text('["x"]\n')
    with pytest.raises(release.ReleaseError, match="sha256"):
        release.verify_folder(built["out"])


def test_real_build_requires_clean_main_equal_to_origin(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    repo = init_repo(tmp_path / "work")
    (repo / "f.txt").write_text("1")
    head = commit_all(repo)
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "origin", "main")
    assert release.require_releasable_main(repo) == head

    (repo / "f.txt").write_text("2")
    with pytest.raises(release.ReleaseError, match="not clean"):
        release.require_releasable_main(repo)
    commit_all(repo)
    with pytest.raises(release.ReleaseError, match="origin/main"):
        release.require_releasable_main(repo)
    git(repo, "checkout", "-q", "-b", "phase-x")
    with pytest.raises(release.ReleaseError, match="main only"):
        release.require_releasable_main(repo)


# ---------------------------------------------------------------- upload (fake Hub)


class FakeHub:
    def __init__(self, tags: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.tags = list(tags)
        self.files: dict[str, bytes] = {}

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        assert kw == {"private": False, "exist_ok": True}
        self.calls.append("create_repo")

    def list_repo_refs(self, repo_id: str, **kw: Any) -> Any:
        self.calls.append("list_repo_refs")
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def upload_folder(self, **kw: Any) -> Any:
        self.calls.append("upload_folder")
        self.files = {p.name: p.read_bytes() for p in Path(kw["folder_path"]).iterdir()}
        return SimpleNamespace(oid="abc123def4567890")

    def create_tag(self, repo_id: str, **kw: Any) -> None:
        assert kw["revision"] == "abc123def4567890"
        self.calls.append("create_tag")
        self.tags.append(kw["tag"])

    def hf_hub_download(self, repo_id: str, filename: str, **kw: Any) -> str:
        assert kw["revision"] in self.tags
        path = Path(kw["local_dir"]) / filename
        path.write_bytes(self.files[filename])
        return str(path)


def as_real_build(built: dict[str, Any], version: str = VERSION) -> Path:
    """Turn the dry-run folder into what a real build of HEAD would look like."""
    real: Path = built["repo"] / "data" / "release" / f"v{version}"
    shutil.copytree(built["out"], real)
    m = json.loads((real / "manifest.json").read_text())
    m |= {"dry_run": False, "git_commit": git(built["repo"], "rev-parse", "HEAD")}
    (real / "manifest.json").write_text(json.dumps(m))
    return real


def test_upload_tags_and_verifies(built: dict[str, Any]) -> None:
    hub = FakeHub()
    manifest = release.upload(VERSION, as_real_build(built), api=hub, repo=built["repo"])
    assert hub.calls == ["create_repo", "list_repo_refs", "upload_folder", "create_tag"]
    assert hub.tags == [f"v{VERSION}"]
    assert manifest["git_commit"] == git(built["repo"], "rev-parse", "HEAD")


def test_upload_refuses_dry_runs(built: dict[str, Any]) -> None:
    hub = FakeHub()
    with pytest.raises(release.ReleaseError, match="dry run"):
        release.upload(VERSION, built["out"], api=hub, repo=built["repo"])
    assert hub.calls == []


def test_upload_refuses_existing_tag(built: dict[str, Any]) -> None:
    hub = FakeHub(tags=(f"v{VERSION}",))
    with pytest.raises(release.ReleaseError, match="immutable"):
        release.upload(VERSION, as_real_build(built), api=hub, repo=built["repo"])
    assert "upload_folder" not in hub.calls


def test_upload_refuses_a_build_of_another_commit(built: dict[str, Any]) -> None:
    real = as_real_build(built)
    (built["repo"] / "new.txt").write_text("x")
    commit_all(built["repo"], "moved on")
    hub = FakeHub()
    with pytest.raises(release.ReleaseError, match="!= HEAD"):
        release.upload(VERSION, real, api=hub, repo=built["repo"])
    assert hub.calls == []


def test_verify_tag_matches_git_tag(built: dict[str, Any]) -> None:
    repo, hub = built["repo"], FakeHub()
    release.upload(VERSION, as_real_build(built), api=hub, repo=repo)
    git(repo, "tag", f"v{VERSION}")
    assert release.verify_tag(VERSION, api=hub, repo=repo) == git(repo, "rev-parse", "HEAD")

    git(repo, "tag", "-d", f"v{VERSION}")
    (repo / "later.txt").write_text("x")
    commit_all(repo, "later")
    git(repo, "tag", f"v{VERSION}")
    with pytest.raises(release.ReleaseError, match="git_commit"):
        release.verify_tag(VERSION, api=hub, repo=repo)


def test_find_checkpoint_requires_exactly_one_match(tmp_path: Path) -> None:
    for name, seed in (("a", 0), ("b", 1)):
        d = tmp_path / name
        d.mkdir()
        key = {"config": NRMSLiteConfig(seed=seed).model_dump(), "embedding_cache_id": "e1"}
        (d / "key.json").write_text(json.dumps(key))
    assert release.find_checkpoint(tmp_path, NRMSLiteConfig(seed=0), "e1").name == "a"
    with pytest.raises(release.ReleaseError):
        release.find_checkpoint(tmp_path, NRMSLiteConfig(seed=0), "other-embeddings")


# ---------------------------------------------------------------- v0.2.0: inherit + topics

TOPIC_MAP_YAML = "version: 1\ncategories:\n  news: {default: us}\n"


def topic_sources(built: dict[str, Any], **overrides: Any) -> release.Sources:
    """Sources for a v0.2.0-style build: the encoder comes from ``built['out']``."""
    from firstread_core.topics import TOPIC_IDS
    from firstread_ml.models.topic_classifier import load_topic_map

    repo, s = built["repo"], built["sources"]
    topic_map = repo / "ml" / "configs" / "topic_map.yaml"
    topic_map.parent.mkdir(parents=True, exist_ok=True)
    topic_map.write_text(TOPIC_MAP_YAML)
    sha = load_topic_map(topic_map).sha256
    cls_dir = repo / "data" / "models" / "topic_classifier"
    cls_dir.mkdir(parents=True, exist_ok=True)
    labels = list(TOPIC_IDS)
    y = np.arange(len(s.emb)) % len(labels)
    joblib.dump(LogisticRegression(max_iter=500).fit(s.emb, y), cls_dir / "model.joblib")
    (cls_dir / "labels.json").write_text(json.dumps(labels))
    (cls_dir / "policy.json").write_text(json.dumps({"threshold": 0.4, "topic_map_sha256": sha}))
    per_mode = {"accuracy": 0.8, "macro_f1": 0.7, "majority_class_accuracy": 0.3, "per_class": []}
    report = repo / "ml" / "reports" / "topic-classifier-mind-large.json"
    report.write_text(
        json.dumps(
            {
                "labels": labels,
                "topic_map_sha256": sha,
                "threshold": 0.4,
                "n_test_dev_only": 10,
                "gate": {"passed": True, "failures": []},
                "test": {"title_abstract": per_mode, "title_only": per_mode},
            }
        )
    )
    commit_all(repo, "topic fixtures")
    fields = {
        **s.__dict__,
        "checkpoint": None,
        "classifier_dir": cls_dir,
        "classifier_report": report,
        "inherit": built["out"],
        "topic_map": topic_map,
    }
    return release.Sources(**(fields | overrides))


def build_from(built: dict[str, Any], sources: release.Sources, root: str = "release2") -> Path:
    repo = built["repo"]
    return release.build(
        "0.0.2", sources, dry_run=True, repo=repo, out_root=repo / "data" / root, bench=fast_bench
    )


def test_inherited_encoder_is_byte_identical_and_topics_ship(built: dict[str, Any]) -> None:
    prev = json.loads((built["out"] / "manifest.json").read_text())
    out = build_from(built, topic_sources(built))
    manifest = release.verify_folder(out)
    for name in release.INHERITED_FILES:
        assert manifest["files"][name]["sha256"] == prev["files"][name]["sha256"]
    assert manifest["user_encoder"]["inherited_from"] == prev["tag"]
    assert manifest["user_encoder"]["score_scale"] == prev["user_encoder"]["score_scale"]
    cc = manifest["category_classifier"]
    assert cc["kind"] == "topics"
    assert cc["n_labels"] == 13
    assert cc["low_confidence_threshold"] == 0.4
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["parity"]["user_encoder"]["inherited_from"] == prev["tag"]
    assert metrics["classifier"]["kind"] == "topics"


def test_tampered_inherited_encoder_is_refused(built: dict[str, Any]) -> None:
    sources = topic_sources(built)
    (built["out"] / "population.npy").write_bytes(b"not the released vector")
    with pytest.raises(release.ReleaseError, match="sha256"):
        build_from(built, sources)


def test_topic_classifier_must_match_the_committed_topic_map(built: dict[str, Any]) -> None:
    sources = topic_sources(built)
    assert sources.topic_map is not None
    sources.topic_map.write_text(TOPIC_MAP_YAML + "  sports: {default: sports}\n")
    with pytest.raises(release.ReleaseError, match="different topic_map"):
        build_from(built, sources)


def test_topic_classifier_must_cover_every_topic(built: dict[str, Any]) -> None:
    sources = topic_sources(built)
    (sources.classifier_dir / "labels.json").write_text(json.dumps(["us", "world"]))
    with pytest.raises(release.ReleaseError, match="TOPIC_IDS"):
        build_from(built, sources)


def test_failed_topic_gate_blocks_the_release(built: dict[str, Any]) -> None:
    sources = topic_sources(built)
    report = json.loads(sources.classifier_report.read_text())
    report["gate"] = {"passed": False, "failures": ["autos: title-only F1 0.1 < 0.3"]}
    sources.classifier_report.write_text(json.dumps(report))
    with pytest.raises(release.ReleaseError, match="topic gate failed"):
        build_from(built, sources)
