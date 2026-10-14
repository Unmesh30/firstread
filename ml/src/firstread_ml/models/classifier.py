"""Category classifier: logistic regression on frozen article embeddings.

Ingestion (P2) uses it to label live articles with a MIND category.

* Train on articles in MIND train ``news.tsv``. Hyperparameters (C, class
  weighting) are chosen by macro-F1 on a stratified 10% split of those
  articles, then the model is refit on all of them.
* Report accuracy and macro-F1 on articles that appear **only** in dev
  ``news.tsv``, which nothing was fitted or tuned on.
* Categories with fewer than ``MIN_TRAIN_PER_CLASS`` training articles (MIND has
  a few with 1-25) are dropped from the label set; dev articles with those
  labels are excluded from the test set and counted in the report.
"""

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from firstread_ml.embed_cache import load_or_build
from firstread_ml.mind.load import load_news, prepare, read_news
from firstread_ml.mind.manifest import split_name
from firstread_ml.paths import data_dir, mind_raw_dir, processed_dir, reports_dir

matplotlib.rcParams["svg.hashsalt"] = "firstread"

MIN_TRAIN_PER_CLASS = 10
GRID_C = (0.1, 0.3, 1.0, 3.0, 10.0)
GRID_CLASS_WEIGHT = (None, "balanced")
VAL_FRACTION = 0.1


def _fit(
    x: NDArray[np.float32], y: NDArray[np.int64], c: float, cw: str | None, seed: int
) -> LogisticRegression:
    clf = LogisticRegression(C=c, class_weight=cw, max_iter=2000, random_state=seed)
    return clf.fit(x, y)


def select_and_fit(
    x: NDArray[np.float32], y: NDArray[np.int64], seed: int = 0
) -> tuple[LogisticRegression, list[dict[str, Any]], dict[str, Any]]:
    """Grid-search on a stratified validation split, then refit on everything."""
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=VAL_FRACTION, stratify=y, random_state=seed
    )
    grid: list[dict[str, Any]] = []
    for c in GRID_C:
        for cw in GRID_CLASS_WEIGHT:
            pred = _fit(x_tr, y_tr, c, cw, seed).predict(x_val)
            grid.append(
                {
                    "C": c,
                    "class_weight": cw or "none",
                    "val_macro_f1": float(f1_score(y_val, pred, average="macro")),
                    "val_accuracy": float(accuracy_score(y_val, pred)),
                }
            )
    best = max(grid, key=lambda g: g["val_macro_f1"])
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    return _fit(x, y, best["C"], cw, seed), grid, best


def run(
    variant: str, data_root: Path | None = None, out_dir: Path | None = None, seed: int = 0
) -> dict[str, Any]:
    raw = mind_raw_dir(data_root)
    proc = prepare(variant, raw_dir=raw, out_dir=processed_dir(variant, data_root))
    news = load_news(variant, proc)
    emb = load_or_build(variant, news, None, data_root)
    train_ids = set(read_news(raw / split_name(variant, "train") / "news.tsv")["nid"].to_list())
    in_train = news["nid"].is_in(list(train_ids)).to_numpy()
    cats = news["category"].to_numpy()

    labels, counts = np.unique(cats[in_train], return_counts=True)
    keep = sorted(str(c) for c in labels[counts >= MIN_TRAIN_PER_CLASS])
    dropped = sorted(str(c) for c in labels[counts < MIN_TRAIN_PER_CLASS])
    code = {c: i for i, c in enumerate(keep)}
    known = np.array([c in code for c in cats])
    y_all = np.array([code.get(c, -1) for c in cats], dtype=np.int64)

    tr = in_train & known
    test = ~in_train & known
    excluded = int((~in_train & ~known).sum())
    t0 = time.perf_counter()
    clf, grid, best = select_and_fit(emb[tr], y_all[tr], seed)
    pred = clf.predict(emb[test])
    y_test = y_all[test]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=np.arange(len(keep)), zero_division=0
    )
    result: dict[str, Any] = {
        "variant": variant,
        "labels": keep,
        "dropped_labels": dropped,
        "min_train_per_class": MIN_TRAIN_PER_CLASS,
        "n_train": int(tr.sum()),
        "n_test_dev_only": int(test.sum()),
        "n_test_excluded_unknown_label": excluded,
        "selection": {"grid": grid, "best": best, "val_fraction": VAL_FRACTION},
        "test": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", labels=np.arange(len(keep)))),
            "majority_class_accuracy": float(np.bincount(y_test).max() / len(y_test)),
            "per_class": [
                {
                    "label": keep[i],
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(len(keep))
            ],
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    model_dir = (data_root or data_dir()) / "models" / variant / "classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "model.joblib")  # local only; ONNX export in 1c
    (model_dir / "labels.json").write_text(json.dumps(keep) + "\n")
    _write_report(
        result,
        confusion_matrix(y_test, pred, labels=np.arange(len(keep))),
        out_dir or reports_dir(),
    )
    return result


def _write_report(r: dict[str, Any], cm: NDArray[np.int64], out_dir: Path) -> None:
    v = r["variant"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(r["labels"])), r["labels"], rotation=90)
    ax.set_yticks(range(len(r["labels"])), r["labels"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"MIND-{v} category classifier: confusion (row-normalized), dev-only articles")
    fig.tight_layout()
    fig.savefig(fig_dir / f"classifier-{v}-confusion.svg", format="svg", metadata={"Date": None})
    plt.close(fig)

    t, sel = r["test"], r["selection"]
    lines = [
        f"# Category classifier: MIND-{v}",
        "",
        "Generated by `firstread-ml classify`; do not edit by hand. Aggregates only.",
        "Logistic regression on frozen bge-small embeddings (`packages/core`).",
        "",
        f"- Train: {r['n_train']:,} articles from train `news.tsv`; "
        f"hyperparameters by macro-F1 on a stratified {sel['val_fraction']:.0%} split, "
        "then refit on all.",
        f"- Test: {r['n_test_dev_only']:,} articles that appear only in dev `news.tsv` "
        f"(excluded {r['n_test_excluded_unknown_label']:,} with a dropped label).",
        f"- Labels: {len(r['labels'])}; dropped (< {r['min_train_per_class']} training articles): "
        f"{', '.join(r['dropped_labels']) or 'none'}.",
        f"- Selected: C={sel['best']['C']}, class_weight={sel['best']['class_weight']}",
        "",
        "## Test results (dev-only articles)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | **{t['accuracy']:.4f}** |",
        f"| Macro-F1 | **{t['macro_f1']:.4f}** |",
        f"| Majority-class accuracy (reference) | {t['majority_class_accuracy']:.4f} |",
        "",
        "## Per class",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {c['label']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} "
        f"| {c['support']:,} |"
        for c in t["per_class"]
    ]
    lines += [
        "",
        "## Model selection (validation split of train articles)",
        "",
        "| C | class_weight | Val macro-F1 | Val accuracy |",
        "|---:|---|---:|---:|",
    ]
    lines += [
        f"| {g['C']} | {g['class_weight']} | {g['val_macro_f1']:.4f} | {g['val_accuracy']:.4f} |"
        for g in sel["grid"]
    ]
    lines += ["", f"![Confusion matrix](figures/classifier-{v}-confusion.svg)", ""]
    (out_dir / f"classifier-mind-{v}.md").write_text("\n".join(lines))
    (out_dir / f"classifier-mind-{v}.json").write_text(
        json.dumps(r, indent=1, sort_keys=True) + "\n"
    )
