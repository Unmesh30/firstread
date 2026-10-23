"""Tune and report the ranking blend on MIND, following ADR 0014's pre-registered rules.

``rank-tune`` (holdout, MIND-small, an encoder trained on the fit days only):

1. blend: 540 configs (normalization x half-life x w_fresh x w_trend), rule on
   J = ½(nDCG@10 overall + cold candidates) with a cold-user guard;
2. MMR λ with the 2-per-story cap, rule: smallest λ within 1% of λ = 1;
3. the cold-start constant k on users with stated-topic proxies;
4. whether cold users need their own blend weights.

``rank-report`` (dev, MIND-large, the shipped v0.2.0 ONNX encoder): the chosen
config once, never tuned on.

Every ranking function comes from ``firstread_core.ranking`` (the API's code).
Reports hold aggregates only.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from firstread_core import pools, stories
from firstread_core.embedding import embedding_cache_key, l2_normalize
from firstread_core.ranking import (
    RankingConfig,
    blend,
    cold_start_vector,
    freshness,
    mmr,
    normalize,
    trend_feature,
)
from firstread_core.ranking.config import config_path
from firstread_core.ranking.features import NORMS, Norm
from firstread_core.topics import TOPIC_IDS
from firstread_ml.baselines import SCORE_CHUNK, FitContext
from firstread_ml.embed_cache import load_or_build
from firstread_ml.eval.harness import EvalData, load_eval_data
from firstread_ml.eval.metrics import per_impression
from firstread_ml.eval.slices import COLD_USER_MAX_HISTORY, Slice, build_slices, restrict
from firstread_ml.mind.data import Impressions
from firstread_ml.mind.load import load_news, load_user_ids
from firstread_ml.mind.splits import clicked_articles
from firstread_ml.models.topic_classifier import load_topic_map, map_topics
from firstread_ml.paths import data_dir, reports_dir

HOUR = 3600.0
TREND_WINDOW_H, TREND_HALF_LIFE_H = 24.0, 6.0

# ---- pre-registered grids and rules (ADR 0014; do not edit after results) ----
HALF_LIVES = (3.0, 6.0, 12.0, 24.0, 48.0)
WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
LAMBDAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)
KS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
COLD_USER_GUARD = 0.01  # step 1: drop configs > 1% below model-only on cold users
J_TIE = 0.001  # step 1: configs within this of the best J are "tied"
MMR_KEEP = 0.99  # step 2: keep >= 99% of lambda=1's overall nDCG@10
K_KEEP = 0.995  # step 3: overall >= 99.5% of k=0's
K_MIN_COLD_IMPRESSIONS = 500  # step 3: soundness threshold
K_JUDGMENT = 3.0
SEGMENT_GAIN = 0.01  # step 4: adopt per-segment weights only with >= 1% relative gain
STORY_CAP = 2
DEV_SAMPLE = 50_000
STATED_TOPICS, MIN_STATED_CLICKS = 3, 3
HISTORY_BUCKETS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 50))
NORM_ORDER = {"raw": 0, "zscore": 1, "percentile": 2}

# The defaults the rest of the config carries (not tuned on MIND; ADR 0014/0017).
FIXED = {
    "trend_half_life_hours": TREND_HALF_LIFE_H,
    "story_cap": STORY_CAP,
    "rerank_size": 150,
    "slate_size": 100,
    "page_size": 20,
}


# ---- features ----


@dataclass
class Features:
    """Everything the ranking needs, per flat candidate of ``test``."""

    test: Impressions
    emb: NDArray[np.float32]
    user_vecs: NDArray[np.float32]  # [n_imp, d], L2-normalized encoder output
    model: NDArray[np.float64]  # cos(user, candidate)
    age_h: NDArray[np.float64]
    trend_raw: NDArray[np.float64]
    slices: list[Slice]
    population: NDArray[np.float32]
    centroids: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    stated: dict[int, list[str]] = field(default_factory=dict)  # impression row -> topics

    def slice(self, name: str) -> Slice:
        return next(s for s in self.slices if s.name == name)


def trend_scores(
    test: Impressions, click_parts: Sequence[Impressions], window_h: float, half_life_h: float
) -> NDArray[np.float64]:
    """Decayed clicks on each candidate strictly before its impression (ADR 0014 proxy).

    ``Σ 2^(-(t - t_c) / half_life)`` over clicks with ``t - window < t_c < t``.
    """
    rows = [click_rows(p) for p in click_parts]
    arts = np.concatenate([p.cand[pos] for p, (_, pos) in zip(click_parts, rows, strict=True)])
    arts = arts.astype(np.int64)
    times = np.concatenate([p.time[r] for p, (r, _) in zip(click_parts, rows, strict=True)])
    t0 = int(min(times.min(), test.time.min())) - int(window_h * HOUR) - 1
    rel = (times - t0).astype(np.int64)
    span = int(max(rel.max(), (test.time - t0).max())) + 1
    key = arts * span + rel
    order = np.argsort(key, kind="stable")
    key, rel = key[order], rel[order]
    h = half_life_h * HOUR
    prefix = np.concatenate([[0.0], np.cumsum(np.power(2.0, rel / h))])
    cand = test.cand.astype(np.int64)
    t = (test.time[test.cand_imp] - t0).astype(np.int64)
    lo = np.searchsorted(key, cand * span + (t - int(window_h * HOUR)), side="right")
    hi = np.searchsorted(key, cand * span + t, side="left")  # strictly before t
    out: NDArray[np.float64] = (prefix[hi] - prefix[lo]) * np.power(2.0, -t / h)
    return out


def user_vectors(
    imps: Impressions, encode: Callable[[Impressions, slice], NDArray[np.float32]]
) -> NDArray[np.float32]:
    parts = [
        encode(imps, slice(s, min(s + SCORE_CHUNK, imps.n))) for s in range(0, imps.n, SCORE_CHUNK)
    ]
    return l2_normalize(np.concatenate(parts))


def cosines(
    imps: Impressions, users: NDArray[np.float32], emb: NDArray[np.float32]
) -> NDArray[np.float64]:
    """cos(user, candidate) per flat candidate, in chunks of impressions (MIND-large dev
    has 14M candidates; gathering them all at once would need ~20 GB)."""
    out = np.empty(len(imps.cand), np.float64)
    for start in range(0, imps.n, SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, imps.n)
        lo, hi = imps.cand_offsets[start], imps.cand_offsets[stop]
        local = np.repeat(np.arange(stop - start), imps.cand_len[start:stop])
        out[lo:hi] = np.einsum("ij,ij->i", users[start:stop][local], emb[imps.cand[lo:hi]])
    return out


def click_rows(imps: Impressions) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(impression row, flat candidate index) of every click, without per-candidate
    temporaries (MIND-large train has 83M candidates)."""
    pos = np.flatnonzero(imps.label == 1)
    rows = np.searchsorted(imps.cand_offsets, pos, side="right") - 1
    return rows.astype(np.int64), pos.astype(np.int64)


def population_vector(fit: Impressions, emb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalized mean embedding of all fit clicks (the empty-history vector), as a
    click-count-weighted sum instead of gathering every clicked row (~5 GB on large)."""
    counts = np.bincount(fit.cand[fit.label == 1], minlength=len(emb)).astype(np.float64)
    return l2_normalize((counts @ emb.astype(np.float64)) / counts.sum())


def stated_topics(
    fit: Impressions,
    test: Impressions,
    fit_users: NDArray[np.int64],
    test_users: NDArray[np.int64],
    article_topic: NDArray[np.str_],
) -> dict[int, list[str]]:
    """Test row -> its user's top-3 app topics among their fit-period clicks (≥ 3 clicks).

    Revealed, not stated, interests (ADR 0014): an optimistic stand-in.
    """
    per_user: dict[int, Counter[str]] = {}
    rows, pos = click_rows(fit)
    for u, a in zip(fit_users[rows].tolist(), fit.cand[pos].tolist(), strict=True):
        t = str(article_topic[a])
        if t in TOPIC_IDS:
            per_user.setdefault(int(u), Counter())[t] += 1
    order = {t: i for i, t in enumerate(TOPIC_IDS)}
    out: dict[int, list[str]] = {}
    for row, u in enumerate(test_users.tolist()):
        c = per_user.get(u)
        if c is None or sum(c.values()) < MIN_STATED_CLICKS:
            continue
        ranked = sorted(c.items(), key=lambda kv: (-kv[1], order[kv[0]]))
        out[row] = [t for t, _ in ranked[:STATED_TOPICS]]
    return out


def user_codes(
    variant: str, files: Sequence[str], fit: Impressions, test: Impressions
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Integer user codes for the fit and test impressions (one code table)."""
    frames = [load_user_ids(variant, s) for s in files]
    users = sorted(set(pl.concat([fr.select("user") for fr in frames])["user"].to_list()))
    code = {u: i for i, u in enumerate(users)}

    def lookup(frame: pl.DataFrame, imps: Impressions) -> NDArray[np.int64]:
        by_imp = dict(zip(frame["imp_id"].to_list(), frame["user"].to_list(), strict=True))
        return np.array([code[by_imp[int(i)]] for i in imps.imp_id], np.int64)

    return lookup(frames[0], fit), lookup(frames[-1], test)


def build_features(
    variant: str,
    split: str,
    encoder: Any,
    model_dir: Path | None = None,
    with_stated: bool = True,
) -> Features:
    """Load MIND, encode users, and compute every candidate's components."""
    data: EvalData = load_eval_data(variant, split)
    news = load_news(variant)
    emb = load_or_build(variant, news)
    ctx = FitContext(
        train=data.fit,
        n_news=data.n_news,
        first_seen=data.first_seen,
        emb=emb,
        seed=0,
        embedding_id=embedding_cache_key()[0][:16],
        model_dir=model_dir,
        tag=f"{variant}-{split}",
        _population=population_vector(data.fit, emb),
    )
    encoder.fit(ctx)
    users = user_vectors(data.test, encoder.user_vectors)
    test = data.test
    age = (np.repeat(test.time, test.cand_len) - data.first_seen[test.cand]) / HOUR
    trend = trend_scores(test, [data.fit, test], TREND_WINDOW_H, TREND_HALF_LIFE_H)
    slices = build_slices(
        test, clicked_articles(data.fit, data.n_news), data.category, data.category_names
    )
    article_topic = np.array(map_topics(news, load_topic_map()))
    seen = np.zeros(data.n_news, np.bool_)
    seen[data.fit.cand] = True
    seen[data.fit.hist] = True
    centroids = pools.topic_centroids(emb[seen], article_topic[seen].tolist())
    stated: dict[int, list[str]] = {}
    if with_stated:
        # Holdout's fit and test both come from the train file; dev's test from dev.
        files = ["train"] if split == "holdout" else ["train", "dev"]
        fit_u, test_u = user_codes(variant, files, data.fit, test)
        stated = stated_topics(data.fit, test, fit_u, test_u, article_topic)
    return Features(
        test=test,
        emb=emb,
        user_vecs=users,
        model=cosines(test, users, emb),
        age_h=age.astype(np.float64),
        trend_raw=trend,
        slices=slices,
        population=ctx.population_vector,
        centroids=centroids,
        stated=stated,
    )


# ---- evaluation ----

MAIN_SLICES = ("overall", "cold_candidates", "cold_users")


def metrics(scores: NDArray[np.float64], f: Features, names: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in names:
        sl = f.slice(n)
        if sl.n_impressions == 0:
            continue
        m = per_impression(scores[sl.idx], f.test.label[sl.idx], sl.offsets, 0)
        out[n] = {k: round(float(v.mean()), 4) for k, v in m.items()}
    return out


def final_scores(
    f: Features,
    norm: Norm,
    half_life: float,
    w_fresh: float,
    w_trend: float,
    model: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    off = f.test.cand_offsets
    return blend(
        f.model if model is None else model,
        freshness(f.age_h, half_life),
        trend_feature(f.trend_raw, off),
        norm,
        w_fresh,
        w_trend,
        off,
    )


# ---- step 1: blend grid ----


def blend_grid(f: Features, progress: Callable[[str], None] = print) -> list[dict[str, Any]]:
    off = f.test.cand_offsets
    trend = trend_feature(f.trend_raw, off)
    normed = {n: normalize(f.model, n, off) for n in NORMS}
    fresh = {h: freshness(f.age_h, h) for h in HALF_LIVES}
    sl = {n: f.slice(n) for n in MAIN_SLICES}
    labels = {n: f.test.label[s.idx] for n, s in sl.items()}
    rows = []
    t0 = time.perf_counter()
    for norm in NORMS:
        for h in HALF_LIVES:
            for wf in WEIGHTS:
                for wt in WEIGHTS:
                    if wf == 0 and h != HALF_LIVES[0]:
                        continue  # freshness unused: one half-life is enough
                    s = normed[norm] + wf * fresh[h] + wt * trend
                    r: dict[str, Any] = {"norm": norm, "half_life": h, "w_fresh": wf,
                                         "w_trend": wt}  # fmt: skip
                    for n, x in sl.items():
                        m = per_impression(s[x.idx], labels[n], x.offsets, 0)
                        r[n] = float(m["ndcg@10"].mean())
                    r["J"] = 0.5 * (r["overall"] + r["cold_candidates"])
                    rows.append(r)
        progress(f"  blend grid: {norm} done ({time.perf_counter() - t0:.0f}s)")
    return rows


def choose_blend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """ADR 0014 step 1."""
    base = next(r for r in rows if r["norm"] == "raw" and r["w_fresh"] == 0 and r["w_trend"] == 0)
    ok = [r for r in rows if r["cold_users"] >= base["cold_users"] * (1 - COLD_USER_GUARD)]
    best = max(r["J"] for r in ok)
    tied = [r for r in ok if r["J"] >= best - J_TIE]
    return min(
        tied,
        key=lambda r: (
            r["w_fresh"] + r["w_trend"],
            NORM_ORDER[r["norm"]],
            r["half_life"],
            -r["J"],
            r["w_trend"],
        ),
    )


# ---- step 2: MMR ----


def groups_for(f: Features, rows: NDArray[np.int64], rel: NDArray[np.float64]) -> list[list[int]]:
    """Per impression in ``rows``: story groups, built in relevance order (complete
    linkage at τ_feed; MIND text is title + abstract, each article its own cluster)."""
    out = []
    off = f.test.cand_offsets
    for i in rows.tolist():
        a, b = off[i], off[i + 1]
        order = np.argsort(-rel[a:b], kind="stable")
        vecs = f.emb[f.test.cand[a:b][order]]
        ids = [uuid.UUID(int=int(x) + 1) for x in f.test.cand[a:b][order]]
        g_sorted = stories.group(vecs, ["title_summary"] * len(order), ids)
        g = [0] * len(order)
        for pos, item in enumerate(order.tolist()):
            g[item] = g_sorted[pos]
        out.append(g)
    return out


def mmr_scores(
    f: Features,
    rel: NDArray[np.float64],
    lam: float,
    rows: Sequence[int],
    groups: Sequence[Sequence[int]],
) -> tuple[NDArray[np.float64], float]:
    """Rank positions as scores (higher = earlier) and mean ILD@10 over ``rows``."""
    out = np.full(len(rel), -1e9)
    off = f.test.cand_offsets
    ild = []
    for k, i in enumerate(rows):
        a, b = off[i], off[i + 1]
        vecs = f.emb[f.test.cand[a:b]]
        res = mmr(rel[a:b], vecs, lam, groups[k], cap=STORY_CAP)
        ranking = res.order + res.capped  # capped-out items appended (ADR 0014)
        out[a + np.asarray(ranking)] = -np.arange(len(ranking), dtype=np.float64)
        top = vecs[ranking[:10]]
        if len(top) > 1:
            s = top @ top.T
            ild.append(float((1 - s)[~np.eye(len(top), dtype=np.bool_)].mean()))
    return out, float(np.mean(ild)) if ild else float("nan")


def subset_slices(f: Features, rows: NDArray[np.int64]) -> dict[str, Slice]:
    """Main slices restricted to the impressions in ``rows``."""
    keep_imp = np.zeros(f.test.n, np.bool_)
    keep_imp[rows] = True
    keep = keep_imp[f.test.cand_imp]
    hist = f.test.hist_len[f.test.cand_imp]
    cold_mask = np.zeros(len(f.test.cand), np.bool_)
    cold_mask[f.slice("cold_candidates").idx] = True
    return {
        "overall": restrict(f.test, keep, "overall"),
        "cold_candidates": restrict(f.test, keep & cold_mask, "cold_candidates"),
        "cold_users": restrict(f.test, keep & (hist <= COLD_USER_MAX_HISTORY), "cold_users"),
    }


def eval_on(scores: NDArray[np.float64], f: Features, sl: dict[str, Slice]) -> dict[str, float]:
    out = {}
    for n, s in sl.items():
        if s.n_impressions:
            m = per_impression(scores[s.idx], f.test.label[s.idx], s.offsets, 0)
            out[n] = float(m["ndcg@10"].mean())
    return out


def mmr_sweep(
    f: Features,
    rel: NDArray[np.float64],
    rows: NDArray[np.int64],
    lambdas: Sequence[float] = LAMBDAS,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    groups = groups_for(f, rows, rel)
    sl = subset_slices(f, rows)
    out = []
    for lam in lambdas:
        s, ild = mmr_scores(f, rel, lam, rows.tolist(), groups)
        r = {"lambda": lam, "ild@10": ild, **eval_on(s, f, sl)}
        out.append(r)
        progress(f"  mmr λ={lam}: nDCG@10 {r['overall']:.4f}, ILD@10 {ild:.4f}")
    return out


def choose_lambda(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """ADR 0014 step 2."""
    base = next(r for r in rows if r["lambda"] == 1.0)
    ok = [r for r in rows if r["overall"] >= MMR_KEEP * base["overall"]]
    return min(ok, key=lambda r: r["lambda"])


# ---- step 3: cold-start k ----


def cold_start_model(f: Features, rows: NDArray[np.int64], k: float) -> NDArray[np.float64]:
    """Model scores with the cold-start user vector for the impressions in ``rows``."""
    model = f.model.copy()
    off = f.test.cand_offsets
    hist_len = f.test.hist_len
    for i in rows.tolist():
        n = int(hist_len[i])
        click = f.user_vecs[i] if n > 0 else None
        u = cold_start_vector(f.centroids, f.stated[i], click, n, k, f.population)
        a, b = off[i], off[i + 1]
        model[a:b] = f.emb[f.test.cand[a:b]] @ u
    return model


def bucket_slices(f: Features, rows: NDArray[np.int64]) -> dict[str, Slice]:
    keep_imp = np.zeros(f.test.n, np.bool_)
    keep_imp[rows] = True
    hist = f.test.hist_len
    out = {"stated_overall": restrict(f.test, keep_imp[f.test.cand_imp], "stated_overall")}
    for lo, hi in HISTORY_BUCKETS:
        m = keep_imp & (hist >= lo) & (hist <= hi)
        out[f"history_{lo}-{hi}"] = restrict(f.test, m[f.test.cand_imp], f"history_{lo}-{hi}")
    cold = keep_imp & (hist <= COLD_USER_MAX_HISTORY)
    out["stated_cold_users"] = restrict(f.test, cold[f.test.cand_imp], "stated_cold_users")
    return out


def k_sweep(
    f: Features,
    choice: dict[str, Any],
    lam: float,
    ks: Sequence[float] = KS,
    rows: NDArray[np.int64] | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], int]:
    rows = np.array(sorted(f.stated), np.int64) if rows is None else rows
    sl = bucket_slices(f, rows)
    n_cold = sl["stated_cold_users"].n_impressions
    out = []
    for k in ks:
        model = cold_start_model(f, rows, k)
        rel = final_scores(f, choice["norm"], choice["half_life"], choice["w_fresh"],
                           choice["w_trend"], model)  # fmt: skip
        s, _ = mmr_scores(f, rel, lam, rows.tolist(), groups_for(f, rows, rel))
        r = {"k": k, **eval_on(s, f, sl)}
        out.append(r)
        progress(f"  k={k}: cold {r.get('stated_cold_users', float('nan')):.4f}")
    return out, n_cold


def choose_k(rows: list[dict[str, Any]], n_cold: int) -> tuple[float, str]:
    """ADR 0014 step 3: (k, "tuned" | "judgment")."""
    if n_cold < K_MIN_COLD_IMPRESSIONS:
        return K_JUDGMENT, "judgment"
    base = next(r for r in rows if r["k"] == 0.0)
    ok = [r for r in rows if r["stated_overall"] >= K_KEEP * base["stated_overall"]]
    best = max(ok, key=lambda r: (r["stated_cold_users"], -r["k"]))
    return float(best["k"]), "tuned"


# ---- step 4: per-segment weights ----


def segment_test(f: Features, choice: dict[str, Any]) -> dict[str, Any]:
    cold = f.slice("cold_users")
    off = f.test.cand_offsets
    trend = trend_feature(f.trend_raw, off)
    normed = normalize(f.model, choice["norm"], off)
    fresh = freshness(f.age_h, choice["half_life"])
    labels = f.test.label[cold.idx]
    grid = []
    for wf in WEIGHTS:
        for wt in WEIGHTS:
            s = normed + wf * fresh + wt * trend
            m = per_impression(s[cold.idx], labels, cold.offsets, 0)
            grid.append({"w_fresh": wf, "w_trend": wt, "cold_users": float(m["ndcg@10"].mean())})
    shared = next(
        g for g in grid if g["w_fresh"] == choice["w_fresh"] and g["w_trend"] == choice["w_trend"]
    )
    best = max(grid, key=lambda g: (g["cold_users"], -(g["w_fresh"] + g["w_trend"])))
    gain = best["cold_users"] / shared["cold_users"] - 1
    return {
        "shared": shared,
        "best": best,
        "relative_gain": gain,
        "adopt": gain >= SEGMENT_GAIN,
        "cold_impressions": cold.n_impressions,
    }


# ---- the config ----


def config_from(
    choice: dict[str, Any], lam: float, k: float, k_source: str, segment: dict[str, Any]
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "version": "rank-1",
        "model_norm": choice["norm"],
        "w_fresh": choice["w_fresh"],
        "w_trend": choice["w_trend"],
        "half_life_hours": choice["half_life"],
        "mmr_lambda": lam,
        "cold_start_k": k,
        "cold_start_k_source": k_source,
        **FIXED,
    }
    if segment["adopt"]:
        cfg["cold_segment"] = {
            "max_history": COLD_USER_MAX_HISTORY,
            "w_fresh": segment["best"]["w_fresh"],
            "w_trend": segment["best"]["w_trend"],
        }
    RankingConfig.from_dict(cfg)  # validates
    return cfg


def write_config(cfg: dict[str, Any], path: Path | None = None) -> Path:
    import yaml

    path = path or config_path(cfg["version"])
    header = (
        "# Ranking config (ADR 0014). Written by `firstread-ml rank-tune --write-config`\n"
        "# from MIND-small holdout; see ml/reports/ranking-mind-small-holdout.md.\n"
        "# A new constant means a new version file (the test pins each file's SHA-256).\n"
    )
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    return path


# ---- commands ----


def tune(progress: Callable[[str], None] = print, write: bool = False) -> dict[str, Any]:
    """ADR 0014 steps 1-4 on MIND-small holdout."""
    from firstread_ml.models.scorer import DEFAULT_CONFIG, NRMSLiteScorer

    t0 = time.perf_counter()
    f = build_features(
        "small", "holdout", NRMSLiteScorer(DEFAULT_CONFIG, seed=0),
        model_dir=data_dir() / "models" / "small" / "holdout",
    )  # fmt: skip
    progress(f"features ready ({time.perf_counter() - t0:.0f}s): {f.test.n} impressions")
    grid = blend_grid(f, progress)
    choice = choose_blend(grid)
    progress(f"step 1: {choice}")
    rel = final_scores(f, choice["norm"], choice["half_life"], choice["w_fresh"], choice["w_trend"])
    rows = np.arange(f.test.n, dtype=np.int64)
    sweep = mmr_sweep(f, rel, rows, progress=progress)
    lam = float(choose_lambda(sweep)["lambda"])
    progress(f"step 2: lambda {lam}")
    ks, n_cold = k_sweep(f, choice, lam, progress=progress)
    k, k_source = choose_k(ks, n_cold)
    progress(f"step 3: k {k} ({k_source}, {n_cold} cold stated impressions)")
    segment = segment_test(f, choice)
    progress(f"step 4: adopt per-segment weights: {segment['adopt']}")
    cfg = config_from(choice, lam, k, k_source, segment)
    base = next(r for r in grid if r["norm"] == "raw" and r["w_fresh"] == 0 and r["w_trend"] == 0)
    report = {
        "split": "holdout",
        "variant": "small",
        "impressions": f.test.n,
        "slices": {n: f.slice(n).n_impressions for n in MAIN_SLICES},
        "stated_topic_impressions": len(f.stated),
        "step1": {
            "configs": len(grid),
            "model_only": base,
            "choice": choice,
            "top": sorted(grid, key=lambda r: -r["J"])[:15],
            "by_norm": {
                n: max((r for r in grid if r["norm"] == n), key=lambda r: r["J"]) for n in NORMS
            },
        },
        "step2": {"sweep": sweep, "lambda": lam},
        "step3": {"sweep": ks, "cold_stated_impressions": n_cold, "k": k, "source": k_source},
        "step4": segment,
        "config": cfg,
        "seconds": round(time.perf_counter() - t0),
    }
    write_report(report, "ranking-mind-small-holdout")
    if write:
        progress(f"wrote {write_config(cfg)}")
    return report


def dev_report(
    release: Path, version: str = "rank-1", progress: Callable[[str], None] = print
) -> dict[str, Any]:
    """The chosen config on MIND-large dev with the shipped encoder, once."""
    from firstread_core.ranking import load_config
    from firstread_ml.models.onnx_scorer import ORTUserEncoderScorer

    t0 = time.perf_counter()
    cfg = load_config(version)
    f = build_features("large", "dev", ORTUserEncoderScorer(release / "user_encoder.onnx"))
    progress(f"features ready ({time.perf_counter() - t0:.0f}s): {f.test.n} impressions")
    names = ["overall", "cold_candidates", "cold_users"] + [
        s.name for s in f.slices if s.name.startswith("category:")
    ]
    model_only = final_scores(f, "raw", cfg.half_life_hours, 0.0, 0.0)
    blended = final_scores(f, cfg.model_norm, cfg.half_life_hours, cfg.w_fresh, cfg.w_trend)
    if cfg.cold_segment is not None:
        seg = final_scores(f, cfg.model_norm, cfg.half_life_hours, cfg.cold_segment.w_fresh,
                           cfg.cold_segment.w_trend)  # fmt: skip
        cold = f.test.hist_len[f.test.cand_imp] <= cfg.cold_segment.max_history
        blended = np.where(cold, seg, blended)
    full = {"model_only": metrics(model_only, f, names), "blend": metrics(blended, f, names)}
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(f.test.n, size=min(DEV_SAMPLE, f.test.n), replace=False))
    sl = subset_slices(f, rows)
    sample = {"model_only": eval_on(model_only, f, sl), "blend": eval_on(blended, f, sl)}
    for lam in sorted({1.0, cfg.mmr_lambda}, reverse=True):
        s, ild = mmr_scores(f, blended, lam, rows.tolist(), groups_for(f, rows, blended))
        sample[f"blend+mmr(λ={lam})"] = {**eval_on(s, f, sl), "ild@10": ild}
        progress(f"  dev mmr λ={lam} done")
    stated_rows = np.array([r for r in rows.tolist() if r in f.stated], np.int64)
    choice = {"norm": cfg.model_norm, "half_life": cfg.half_life_hours,
              "w_fresh": cfg.w_fresh, "w_trend": cfg.w_trend}  # fmt: skip
    ks, n_cold = k_sweep(f, choice, cfg.mmr_lambda, sorted({0.0, cfg.cold_start_k}),
                         stated_rows, progress)  # fmt: skip
    report = {
        "split": "dev",
        "variant": "large",
        "encoder": str(release.name),
        "config": cfg.version,
        "impressions": f.test.n,
        "slices": {n: f.slice(n).n_impressions for n in names},
        "full": full,
        "sample": {"impressions": len(rows), "rows": sample},
        "cold_start": {"stated_impressions": len(stated_rows), "cold": n_cold, "sweep": ks},
        "seconds": round(time.perf_counter() - t0),
    }
    write_report(report, "ranking-mind-large-dev")
    return report


# ---- reports (aggregates only) ----


def _f(x: Any) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def write_report(r: dict[str, Any], stem: str, out_dir: Path | None = None) -> None:
    out_dir = out_dir or reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    (out_dir / f"{stem}.md").write_text(markdown(r) + "\n")


def markdown(r: dict[str, Any]) -> str:
    lines = [
        f"# Ranking: MIND-{r['variant']} {r['split']}",
        "",
        "Generated by `firstread-ml rank-tune` / `rank-report`; do not edit by hand. "
        "Aggregates only. Rules: ADR 0014 (pre-registered).",
        "",
    ]
    if r["split"] == "holdout":
        s1 = r["step1"]
        c = s1["choice"]
        lines += [
            f"{r['impressions']} impressions; slices: {r['slices']}; "
            f"stated-topic impressions: {r['stated_topic_impressions']}.",
            "",
            "## Step 1: blend (nDCG@10)",
            "",
            "| Config | overall | cold candidates | cold users | J |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, row in [("model only", s1["model_only"]), ("**chosen**", c)] + [
            (f"best {n}", row) for n, row in s1["by_norm"].items()
        ]:
            desc = (
                f"{label}: {row['norm']}, h={row['half_life']}, "
                f"w_fresh={row['w_fresh']}, w_trend={row['w_trend']}"
            )
            lines.append(
                f"| {desc} | {_f(row['overall'])} | {_f(row['cold_candidates'])} | "
                f"{_f(row['cold_users'])} | {_f(row['J'])} |"
            )
        lines += ["", "## Step 2: MMR with the 2-per-story cap", "",
                  "| λ | nDCG@10 overall | cold candidates | cold users | ILD@10 |",
                  "|---:|---:|---:|---:|---:|"]  # fmt: skip
        for row in r["step2"]["sweep"]:
            mark = " **(chosen)**" if row["lambda"] == r["step2"]["lambda"] else ""
            lines.append(
                f"| {row['lambda']}{mark} | {_f(row['overall'])} | "
                f"{_f(row.get('cold_candidates'))} | {_f(row.get('cold_users'))} | "
                f"{_f(row['ild@10'])} |"
            )
        s3 = r["step3"]
        lines += [
            "", "## Step 3: cold-start k (stated-topic users)", "",
            f"Cold stated impressions: {s3['cold_stated_impressions']} "
            f"(soundness threshold {K_MIN_COLD_IMPRESSIONS}). "
            f"Chosen k = {s3['k']} ({s3['source']}).", "",
        ]  # fmt: skip
        keys = [k for k in s3["sweep"][0] if k != "k"]
        lines += ["| k | " + " | ".join(keys) + " |", "|---:|" + "---:|" * len(keys)]
        for row in s3["sweep"]:
            lines.append(f"| {row['k']} | " + " | ".join(_f(row.get(k)) for k in keys) + " |")
        s4 = r["step4"]
        lines += [
            "", "## Step 4: separate weights for cold users?", "",
            f"Shared weights on cold users: {_f(s4['shared']['cold_users'])}; best cold-only "
            f"weights (w_fresh={s4['best']['w_fresh']}, w_trend={s4['best']['w_trend']}): "
            f"{_f(s4['best']['cold_users'])}; relative gain {s4['relative_gain'] * 100:.2f}% "
            f"→ adopt: **{s4['adopt']}** (threshold {SEGMENT_GAIN * 100:.0f}%).",
            "", "## Config written", "", "```yaml",
            *(f"{k}: {v}" for k, v in r["config"].items()), "```",
        ]  # fmt: skip
    else:
        lines += [
            f"Encoder: {r['encoder']} (shipped); config: {r['config']}; "
            f"{r['impressions']} impressions.",
            "", "## Full dev set (nDCG@10)", "",
            "| Slice | model only | blend |", "|---|---:|---:|",
        ]  # fmt: skip
        for n in r["full"]["blend"]:
            a, b = r["full"]["model_only"][n]["ndcg@10"], r["full"]["blend"][n]["ndcg@10"]
            lines.append(f"| {n} | {_f(a)} | {_f(b)} |")
        lines += [
            "", f"## Fixed sample ({r['sample']['impressions']} impressions, seed 0)", "",
            "| Row | overall | cold candidates | cold users | ILD@10 |",
            "|---|---:|---:|---:|---:|",
        ]  # fmt: skip
        for name, row in r["sample"]["rows"].items():
            lines.append(
                f"| {name} | {_f(row.get('overall'))} | {_f(row.get('cold_candidates'))} | "
                f"{_f(row.get('cold_users'))} | {_f(row.get('ild@10', ''))} |"
            )
        cs = r["cold_start"]
        lines += [
            "", f"## Cold start (stated-topic users in the sample: {cs['stated_impressions']}, "
            f"cold: {cs['cold']})", "",
        ]  # fmt: skip
        keys = [k for k in cs["sweep"][0] if k != "k"]
        lines += ["| k | " + " | ".join(keys) + " |", "|---:|" + "---:|" * len(keys)]
        for row in cs["sweep"]:
            lines.append(f"| {row['k']} | " + " | ".join(_f(row.get(k)) for k in keys) + " |")
    lines.append(f"\nRun time: {r['seconds']} s.")
    return "\n".join(lines)
