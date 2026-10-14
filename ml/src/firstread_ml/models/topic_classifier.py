"""Topic classifier: logistic regression from a frozen embedding to an app topic.

Replaces the 15-way MIND category classifier from v0.2.0 (ADR 0006). Labels are
the app topics (``firstread_core.topics``), mapped from MIND (category,
subcategory) pairs by ``ml/configs/topic_map.yaml``.

Live GDELT articles have a title but no summary, so the model must work on both
kinds of text:

* **Training** uses two rows per train article, its title+abstract embedding and
  its title-only embedding.
* **Selection** of C and class weighting maximizes the *mean* validation
  macro-F1 over the two text modes. The validation split is by article, so the
  two rows of one article never straddle it. The model is then refit on all
  train articles.
* **Test** on articles that appear only in dev ``news.tsv``, in both modes.

The low-confidence threshold for the feed-hint tie-breaker
(``firstread_core.topics.resolve_topic``) is chosen on the validation split: the
lowest top-1 probability from which every 0.05-wide confidence bin is at least
50% accurate.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from firstread_core.topics import TOPIC_IDS
from firstread_ml.embed_cache import load_or_build
from firstread_ml.mind.load import load_news, prepare, read_news
from firstread_ml.mind.manifest import split_name
from firstread_ml.paths import REPO_ROOT, data_dir, mind_raw_dir, processed_dir, reports_dir

matplotlib.rcParams["svg.hashsalt"] = "firstread"

TOPIC_MAP = REPO_ROOT / "ml" / "configs" / "topic_map.yaml"
EXCLUDE = "exclude"
MODES = ("title_abstract", "title_only")
GRID_C = (0.3, 1.0, 3.0, 10.0, 30.0)
GRID_CLASS_WEIGHT = (None, "balanced")
VAL_FRACTION = 0.1
BIN_WIDTH = 0.05
MIN_BIN_ACCURACY = 0.5
MIN_BIN_SIZE = 30

# Release gate (ADR 0006): every topic must be learnable from titles alone, and
# titles must not be much worse than title+abstract.
GATE_MIN_TOPIC_F1 = 0.30
GATE_MAX_MACRO_F1_GAP = 0.10


class TopicMapError(ValueError):
    """The topic map is malformed or doesn't cover a (category, subcategory) pair."""


@dataclass(frozen=True)
class TopicMap:
    categories: dict[str, dict[str, Any]]
    sha256: str

    def topic(self, category: str, subcategory: str) -> str:
        """The app topic (or ``"exclude"``) for one MIND pair; unknown pairs raise."""
        entry = self.categories.get(category)
        if entry is None:
            raise TopicMapError(f"category {category!r} is not in the topic map")
        subs: dict[str, str] = entry.get("subcategories") or {}
        if subcategory in subs:
            return subs[subcategory]
        if "default" in entry:
            return str(entry["default"])
        raise TopicMapError(
            f"{category}/{subcategory} is not mapped and {category!r} has no default"
        )


def load_topic_map(path: Path = TOPIC_MAP) -> TopicMap:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise TopicMapError(f"{path}: expected a mapping with version: 1")
    categories: dict[str, dict[str, Any]] = data.get("categories") or {}
    allowed = {*TOPIC_IDS, EXCLUDE}
    for cat, entry in categories.items():
        targets = list((entry.get("subcategories") or {}).values())
        if "default" in entry:
            targets.append(entry["default"])
        bad = sorted({str(t) for t in targets} - allowed)
        if bad:
            raise TopicMapError(f"{cat}: unknown topics {bad}")
    return TopicMap(categories=categories, sha256=hashlib.sha256(raw).hexdigest())


def map_topics(news: pl.DataFrame, topic_map: TopicMap) -> list[str]:
    pairs = zip(news["category"].to_list(), news["subcategory"].to_list(), strict=True)
    return [topic_map.topic(c, s) for c, s in pairs]


def _fit(x: NDArray[np.float32], y: NDArray[np.int64], c: float, cw: str | None, seed: int) -> Any:
    return LogisticRegression(C=c, class_weight=cw, max_iter=2000, random_state=seed).fit(x, y)


def _stack(
    embs: dict[str, NDArray[np.float32]], rows: NDArray[np.int64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Two training rows per article: one per text mode."""
    return np.concatenate([embs[m][rows] for m in MODES]), np.concatenate([y[rows]] * len(MODES))


def confidence_threshold(
    proba: NDArray[np.float64], correct: NDArray[np.bool_]
) -> tuple[float, list[dict[str, Any]]]:
    """Lowest bin edge from which every (non-tiny) 0.05 confidence bin is ≥ 50% accurate."""
    top = proba.max(axis=1)
    edges = np.round(np.arange(0.0, 1.0, BIN_WIDTH), 2)
    bins: list[dict[str, Any]] = []
    for lo in edges:
        hi = lo + BIN_WIDTH
        sel = (top >= lo) & ((top < hi) if hi < 1.0 else (top <= 1.0))
        n = int(sel.sum())
        acc = float(correct[sel].mean()) if n else None
        bins.append({"lo": float(lo), "hi": float(round(hi, 2)), "n": n, "accuracy": acc})
    threshold = 0.0
    for b in bins:  # walk up; the threshold sits above the last failing bin
        acc = b["accuracy"]
        if b["n"] >= MIN_BIN_SIZE and acc is not None and acc < MIN_BIN_ACCURACY:
            threshold = float(b["hi"])
    return threshold, bins


def select_and_fit(
    embs: dict[str, NDArray[np.float32]], y: NDArray[np.int64], seed: int = 0
) -> tuple[Any, dict[str, Any]]:
    """Grid-search on an article-level split (both modes), then refit on everything."""
    idx = np.arange(len(y))
    tr, val = train_test_split(idx, test_size=VAL_FRACTION, stratify=y, random_state=seed)
    grid: list[dict[str, Any]] = []
    fitted = {}
    for c in GRID_C:
        for cw in GRID_CLASS_WEIGHT:
            clf = _fit(*_stack(embs, tr, y), c, cw, seed)
            row: dict[str, Any] = {"C": c, "class_weight": cw or "none"}
            for m in MODES:
                pred = clf.predict(embs[m][val])
                row[f"val_macro_f1_{m}"] = float(f1_score(y[val], pred, average="macro"))
                row[f"val_accuracy_{m}"] = float(accuracy_score(y[val], pred))
            row["val_macro_f1_mean"] = float(np.mean([row[f"val_macro_f1_{m}"] for m in MODES]))
            grid.append(row)
            fitted[(c, cw)] = clf
    best = max(grid, key=lambda g: g["val_macro_f1_mean"])
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    val_clf = fitted[(best["C"], cw)]
    proba = np.concatenate([val_clf.predict_proba(embs[m][val]) for m in MODES])
    truth = np.concatenate([y[val]] * len(MODES))
    threshold, bins = confidence_threshold(proba, val_clf.classes_[proba.argmax(1)] == truth)
    selection = {
        "grid": grid,
        "best": best,
        "val_fraction": VAL_FRACTION,
        "threshold": threshold,
        "threshold_bins": bins,
    }
    return _fit(*_stack(embs, idx, y), best["C"], cw, seed), selection


def evaluate_mode(
    clf: Any, x: NDArray[np.float32], y: NDArray[np.int64], labels: list[str]
) -> tuple[dict[str, Any], NDArray[np.int64]]:
    pred = clf.predict(x)
    k = np.arange(len(labels))
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=k, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=k, zero_division=0)),
        "majority_class_accuracy": float(np.bincount(y, minlength=len(k)).max() / len(y)),
        "per_class": [
            {
                "label": labels[i],
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in k
        ],
    }
    return result, confusion_matrix(y, pred, labels=k)


def gate(test: dict[str, dict[str, Any]]) -> list[str]:
    """Release gate failures (empty = pass)."""
    failures = [
        f"{c['label']}: title-only F1 {c['f1']:.3f} < {GATE_MIN_TOPIC_F1}"
        for c in test["title_only"]["per_class"]
        if c["f1"] < GATE_MIN_TOPIC_F1
    ]
    gap = test["title_abstract"]["macro_f1"] - test["title_only"]["macro_f1"]
    if gap > GATE_MAX_MACRO_F1_GAP:
        failures.append(f"title-only macro-F1 is {gap:.3f} below title+abstract")
    return failures


def run(
    variant: str = "large",
    data_root: Path | None = None,
    out_dir: Path | None = None,
    topic_map_path: Path = TOPIC_MAP,
    seed: int = 0,
    embedder: Any = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    topic_map = load_topic_map(topic_map_path)
    raw = mind_raw_dir(data_root)
    proc = prepare(variant, raw_dir=raw, out_dir=processed_dir(variant, data_root))
    news = load_news(variant, proc)
    embs = {m: load_or_build(variant, news, embedder, data_root, text_mode=m) for m in MODES}
    topics = np.array(map_topics(news, topic_map))
    labels = [t for t in TOPIC_IDS if t in set(topics.tolist())]
    code = {t: i for i, t in enumerate(labels)}
    y_all = np.array([code.get(t, -1) for t in topics], dtype=np.int64)

    train_ids = set(read_news(raw / split_name(variant, "train") / "news.tsv")["nid"].to_list())
    in_train = news["nid"].is_in(list(train_ids)).to_numpy()
    keep = y_all >= 0
    tr, test = np.flatnonzero(in_train & keep), np.flatnonzero(~in_train & keep)
    clf, selection = select_and_fit({m: e[tr] for m, e in embs.items()}, y_all[tr], seed)

    results: dict[str, dict[str, Any]] = {}
    matrices: dict[str, NDArray[np.int64]] = {}
    for m in MODES:
        results[m], matrices[m] = evaluate_mode(clf, embs[m][test], y_all[test], labels)
    failures = gate(results)
    counts = {t: int((topics[tr] == t).sum()) for t in labels}
    result: dict[str, Any] = {
        "variant": variant,
        "labels": labels,
        "topic_map_sha256": topic_map.sha256,
        "n_train_articles": len(tr),
        "n_train_rows": len(tr) * len(MODES),
        "n_excluded": int((topics == EXCLUDE).sum()),
        "n_test_dev_only": len(test),
        "train_per_topic": counts,
        "selection": selection,
        "threshold": selection["threshold"],
        "test": results,
        "gate": {
            "min_topic_f1_title_only": GATE_MIN_TOPIC_F1,
            "max_macro_f1_gap": GATE_MAX_MACRO_F1_GAP,
            "failures": failures,
            "passed": not failures,
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    model_dir = (data_root or data_dir()) / "models" / variant / "topic_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "model.joblib")  # local only; ONNX export at release
    (model_dir / "labels.json").write_text(json.dumps(labels) + "\n")
    (model_dir / "policy.json").write_text(
        json.dumps({"threshold": result["threshold"], "topic_map_sha256": topic_map.sha256}) + "\n"
    )
    _write_report(result, matrices, out_dir or reports_dir())
    return result


def _plot(cm: NDArray[np.int64], labels: list[str], title: str, path: Path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def _write_report(r: dict[str, Any], cms: dict[str, NDArray[np.int64]], out_dir: Path) -> None:
    v = r["variant"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = f"topic-classifier-{v}-confusion-title-only.svg"
    _plot(
        cms["title_only"],
        r["labels"],
        f"MIND-{v} topic classifier, title-only dev-only articles (row-normalized)",
        fig_dir / fig,
    )
    sel, best, ta, to = r["selection"], r["selection"]["best"], *(r["test"][m] for m in MODES)
    lines = [
        f"# Topic classifier: MIND-{v}",
        "",
        "Generated by `firstread-ml topics`; do not edit by hand. Aggregates only.",
        "Logistic regression on frozen bge-small embeddings (`packages/core`) predicting the",
        "13 app topics; labels from `ml/configs/topic_map.yaml` (ADR 0006).",
        "",
        f"- Train: {r['n_train_articles']:,} train `news.tsv` articles x 2 text modes "
        f"(title+abstract, title only) = {r['n_train_rows']:,} rows. "
        f"{r['n_excluded']:,} articles mapped to `exclude`.",
        f"- Selection: mean validation macro-F1 over both modes on a stratified "
        f"{sel['val_fraction']:.0%} article split. Selected C={best['C']}, "
        f"class_weight={best['class_weight']}.",
        f"- Test: {r['n_test_dev_only']:,} articles that appear only in dev `news.tsv`.",
        f"- Low-confidence threshold for the feed-hint tie-breaker: **{r['threshold']:.2f}**.",
        f"- Topic map SHA256: `{r['topic_map_sha256']}`",
        "",
        "## Test results (dev-only articles)",
        "",
        "| Metric | Title + abstract | Title only |",
        "|---|---:|---:|",
        f"| Accuracy | **{ta['accuracy']:.4f}** | **{to['accuracy']:.4f}** |",
        f"| Macro-F1 | **{ta['macro_f1']:.4f}** | **{to['macro_f1']:.4f}** |",
        f"| Majority-class accuracy (reference) | {ta['majority_class_accuracy']:.4f} "
        f"| {to['majority_class_accuracy']:.4f} |",
        "",
        f"Release gate: every topic's title-only F1 ≥ {r['gate']['min_topic_f1_title_only']}, "
        f"and title-only macro-F1 at most {r['gate']['max_macro_f1_gap']} below "
        f"title+abstract: **{'PASS' if r['gate']['passed'] else 'FAIL'}**"
        + ("" if r["gate"]["passed"] else f" ({'; '.join(r['gate']['failures'])})"),
        "",
        "## Per topic",
        "",
        "| Topic | Train articles | F1 (title+abstract) | P / R / F1 (title only) | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for a, b in zip(ta["per_class"], to["per_class"], strict=True):
        lines.append(
            f"| {a['label']} | {r['train_per_topic'][a['label']]:,} | {a['f1']:.3f} "
            f"| {b['precision']:.3f} / {b['recall']:.3f} / {b['f1']:.3f} | {a['support']:,} |"
        )
    lines += [
        "",
        "## Confidence bins (validation, both modes)",
        "",
        "| Top-1 probability | n | Accuracy |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {b['lo']:.2f}-{b['hi']:.2f} | {b['n']:,} | {b['accuracy']:.3f} |"
        for b in sel["threshold_bins"]
        if b["n"]
    ]
    lines += [
        "",
        "## Model selection (validation split of train articles)",
        "",
        "| C | class_weight | Macro-F1 title+abstract | Macro-F1 title only | Mean |",
        "|---:|---|---:|---:|---:|",
    ]
    lines += [
        f"| {g['C']} | {g['class_weight']} | {g['val_macro_f1_title_abstract']:.4f} "
        f"| {g['val_macro_f1_title_only']:.4f} | {g['val_macro_f1_mean']:.4f} |"
        for g in sel["grid"]
    ]
    lines += ["", f"![Confusion matrix, title only](figures/{fig})", ""]
    (out_dir / f"topic-classifier-mind-{v}.md").write_text("\n".join(lines))
    (out_dir / f"topic-classifier-mind-{v}.json").write_text(
        json.dumps(r, indent=1, sort_keys=True) + "\n"
    )
