"""Run scorers over a split and compute every metric on every slice.

split="holdout": fit on train minus its last day, test on that last day (tuning).
split="dev":     fit on all of train, test on MIND dev (final numbers, report once).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from firstread_core.embedding import Embedder, embedding_cache_key
from firstread_ml import __version__
from firstread_ml.baselines import BASELINES, FitContext, Scorer
from firstread_ml.embed_cache import load_or_build
from firstread_ml.eval.metrics import METRICS, mean_metrics
from firstread_ml.eval.slices import LOW_N, build_slices
from firstread_ml.mind.data import Impressions
from firstread_ml.mind.load import load_impressions, load_news, prepare
from firstread_ml.mind.manifest import HF_REVISION
from firstread_ml.mind.splits import clicked_articles, first_seen, holdout_split
from firstread_ml.paths import data_dir, mind_raw_dir, processed_dir

SPLITS = ("holdout", "dev")
TRAINABLE = ("nrms_lite",)
MODELS = (*BASELINES, *TRAINABLE)


def make_scorer(name: str, nrms_config: Path | None = None) -> Scorer:
    if name in BASELINES:
        scorer: Scorer = BASELINES[name]()
        return scorer
    if name == "nrms_lite":
        from firstread_ml.models.scorer import DEFAULT_CONFIG, NRMSLiteScorer

        return NRMSLiteScorer(nrms_config or DEFAULT_CONFIG)
    raise ValueError(f"unknown model {name!r}; available: {list(MODELS)}")


@dataclass
class EvalData:
    fit: Impressions
    test: Impressions
    n_news: int
    category: np.ndarray
    category_names: list[str]
    first_seen: np.ndarray


def load_eval_data(variant: str, split: str, data_root: Path | None = None) -> EvalData:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    proc = processed_dir(variant, data_root)
    news = load_news(variant, proc)
    train = load_impressions(variant, "train", proc)
    if split == "holdout":
        fit, test = holdout_split(train)
    else:
        fit, test = train, load_impressions(variant, "dev", proc)
    names = sorted(news["category"].unique().to_list())
    code = {c: i for i, c in enumerate(names)}
    category = np.array([code[c] for c in news["category"].to_list()], dtype=np.int32)
    return EvalData(
        fit=fit,
        test=test,
        n_news=news.height,
        category=category,
        category_names=names,
        first_seen=first_seen([fit, test], news.height),
    )


def evaluate(
    scorers: list[Scorer], ctx: FitContext, data: EvalData, seed: int = 0
) -> dict[str, Any]:
    slices = build_slices(
        data.test,
        clicked_articles(data.fit, data.n_news),
        data.category,
        data.category_names,
    )
    results: dict[str, dict[str, dict[str, float]]] = {}
    for scorer in scorers:
        t0 = time.perf_counter()
        scorer.fit(ctx)
        scores = scorer.score(data.test)
        if scores.shape != data.test.cand.shape:
            raise ValueError(f"{scorer.name} returned {scores.shape} scores")
        results[scorer.name] = {
            s.name: mean_metrics(scores[s.idx], data.test.label[s.idx], s.offsets, seed)
            for s in slices
            if s.n_impressions > 0
        }
        print(f"  {scorer.name}: {time.perf_counter() - t0:.1f}s", flush=True)
    return {
        "slices": {
            s.name: {"n_impressions": s.n_impressions, "low_n": s.n_impressions < LOW_N, **s.extra}
            for s in slices
        },
        "results": results,
    }


def run(
    variant: str,
    split: str,
    models: list[str],
    seed: int = 0,
    embedder: Embedder | None = None,
    data_root: Path | None = None,
    raw_dir: Path | None = None,
    nrms_config: Path | None = None,
) -> dict[str, Any]:
    """Prepare (cached), embed (cached) and evaluate ``models`` on ``split``."""
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; available: {list(MODELS)}")
    proc = prepare(
        variant,
        raw_dir=raw_dir or mind_raw_dir(data_root),
        out_dir=processed_dir(variant, data_root),
    )
    data = load_eval_data(variant, split, data_root)
    scorers = [make_scorer(m, nrms_config) for m in models]
    emb = None
    meta: dict[str, Any] = {
        "variant": variant,
        "split": split,
        "seed": seed,
        "firstread_ml": __version__,
        "dataset_revision": HF_REVISION if variant in ("small", "large") else variant,
        "test_impressions": data.test.n,
        "fit_impressions": data.fit.n,
        "metrics": list(METRICS),
    }
    if any(s.needs_embeddings for s in scorers):
        emb = load_or_build(variant, load_news(variant, proc), embedder, data_root)
        # Named "id" (not "key"): it's a public config hash, and secret scanners
        # flag hex strings next to the word "key".
        meta["embedding_cache_id"] = embedding_cache_key()[0][:16]
    ctx = FitContext(
        train=data.fit,
        n_news=data.n_news,
        first_seen=data.first_seen,
        emb=emb,
        seed=seed,
        embedding_id=meta.get("embedding_cache_id", ""),
        model_dir=(data_root or data_dir()) / "models" / variant / split,
        tag=f"{variant}-{split}",
    )
    print(f"evaluating {models} on MIND-{variant} {split} ({data.test.n} impressions)")
    out = evaluate(scorers, ctx, data, seed)
    meta["excluded_impressions"] = data.test.n - out["slices"]["overall"]["n_impressions"]
    return {"meta": meta, **out}
