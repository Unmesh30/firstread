"""How much ranking quality do we lose when articles have a title but no summary?

The model was trained on ``title + ". " + abstract``, but GDELT gives titles
only (train/serve skew). This study runs the released ONNX user encoder and the
``mean_history`` baseline on the fixed real-gate subset (5,000 MIND-small dev
impressions) under three text conditions:

* **A** history and candidates use title + abstract (what the model saw in training);
* **B** candidates use titles only, history keeps abstracts;
* **C** history and candidates both use titles only (a GDELT-only reader).

Decision rule, fixed before running (Phase 2 plan, ADR 0007): if C loses at most
5% relative nDCG@10 versus A on the overall slice, and the model still beats
``mean_history`` in C, title-only articles are ingested as normal.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import Embedder
from firstread_ml.baselines import FitContext, MeanHistoryScorer, Scorer, score_by_user_vectors
from firstread_ml.embed_cache import load_or_build
from firstread_ml.eval import gate_real
from firstread_ml.eval.harness import evaluate, load_eval_data
from firstread_ml.mind.data import Impressions
from firstread_ml.mind.load import load_news, prepare
from firstread_ml.models.onnx_scorer import ORTUserEncoderScorer
from firstread_ml.paths import mind_raw_dir, processed_dir, reports_dir

CONDITIONS = {
    "A": ("title_abstract", "title_abstract"),
    "B": ("title_abstract", "title_only"),
    "C": ("title_only", "title_only"),
}
LABELS = {
    "A": "history + candidates: title + abstract (reference)",
    "B": "candidates: title only",
    "C": "history + candidates: title only",
}
SLICES = ("overall", "cold_candidates", "cold_users")
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")
MAX_RELATIVE_LOSS = 0.05


class SplitTextScorer:
    """Build user vectors from one embedding table, score candidates in another."""

    needs_embeddings = True

    def __init__(self, inner: Any, hist_emb: NDArray[np.float32], cand_emb: NDArray[np.float32]):
        self.inner, self.hist_emb, self.cand_emb = inner, hist_emb, cand_emb
        self.name = inner.name

    def fit(self, ctx: FitContext) -> None:
        ctx.emb = self.hist_emb
        self.inner.fit(ctx)

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return score_by_user_vectors(imps, self.cand_emb, self.inner.user_vectors)


def run(
    onnx_path: Path, data_root: Path | None = None, embedder: Embedder | None = None
) -> dict[str, Any]:
    v = gate_real.VARIANT
    proc = prepare(v, raw_dir=mind_raw_dir(data_root), out_dir=processed_dir(v, data_root))
    news = load_news(v, proc)
    embs = {
        m: load_or_build(v, news, embedder, data_root, text_mode=m)
        for m in ("title_abstract", "title_only")
    }
    data = load_eval_data(v, "dev", data_root)
    data.test = data.test.take(gate_real.subset_rows(data.test.n))
    results: dict[str, Any] = {}
    slices: dict[str, int] = {}
    for cond, (hist_mode, cand_mode) in CONDITIONS.items():
        print(f"condition {cond}: {LABELS[cond]}", flush=True)
        ctx = FitContext(
            train=data.fit, n_news=data.n_news, first_seen=data.first_seen, tag=f"skew-{cond}"
        )
        scorers: list[Scorer] = [
            SplitTextScorer(inner, embs[hist_mode], embs[cand_mode])
            for inner in (MeanHistoryScorer(), ORTUserEncoderScorer(onnx_path))
        ]
        out = evaluate(scorers, ctx, data)
        results[cond] = {
            m: {sl: {k: out["results"][m][sl][k] for k in METRICS} for sl in SLICES}
            for m in out["results"]
        }
        slices = {sl: out["slices"][sl]["n_impressions"] for sl in SLICES}
    abstracts = np.array([bool(a) for a in news["abstract"].to_list()])
    return {
        "meta": {
            "variant": v,
            "split": "dev",
            "subset_size": int(data.test.n),
            "subset_seed": gate_real.SUBSET_SEED,
            "user_encoder": str(onnx_path.name),
            "share_of_articles_with_abstract": round(float(abstracts.mean()), 4),
        },
        "conditions": {c: {"history": h, "candidates": k} for c, (h, k) in CONDITIONS.items()},
        "slices": slices,
        "results": results,
        "decision": decide(results),
    }


def relative_loss(results: dict[str, Any], model: str, cond: str, sl: str = "overall") -> float:
    a = results["A"][model][sl]["ndcg@10"]
    return float((a - results[cond][model][sl]["ndcg@10"]) / a)


def decide(results: dict[str, Any]) -> dict[str, Any]:
    model, base = gate_real.MODEL, gate_real.BASELINE
    loss = relative_loss(results, model, "C")
    beats = results["C"][model]["overall"]["ndcg@10"] > results["C"][base]["overall"]["ndcg@10"]
    ok = loss <= MAX_RELATIVE_LOSS and beats
    return {
        "rule": f"C loses <= {MAX_RELATIVE_LOSS:.0%} relative nDCG@10 vs A (overall) "
        f"and {model} beats {base} in C",
        "relative_loss_C": round(loss, 4),
        "relative_loss_B": round(relative_loss(results, model, "B"), 4),
        "model_beats_baseline_in_C": bool(beats),
        "title_only_ok": bool(ok),
    }


def write_report(r: dict[str, Any], out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or reports_dir()
    d = r["decision"]
    outcome = "ingested as normal" if d["title_only_ok"] else "flagged for a P3 mitigation"
    lines = [
        "# Title-only skew study: MIND-small dev",
        "",
        "Generated by `firstread-ml skew`; do not edit by hand. Aggregates only.",
        "",
        "GDELT articles arrive with a title but no summary, while the model was trained on",
        "title + abstract. This measures the cost on the fixed real-gate subset "
        f"({r['meta']['subset_size']:,} MIND-small dev impressions, seed "
        f"{r['meta']['subset_seed']}), scoring with the released ONNX user encoder.",
        f"{r['meta']['share_of_articles_with_abstract']:.1%} of MIND-small articles have an "
        "abstract.",
        "",
        "| Condition | History text | Candidate text |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {v['history']} | {v['candidates']} |" for c, v in r["conditions"].items()]
    for sl in SLICES:
        lines += [
            "",
            f"## nDCG@10 and AUC, {sl} ({r['slices'][sl]:,} impressions)",
            "",
            "| Model | A nDCG@10 | B nDCG@10 | C nDCG@10 | A AUC | B AUC | C AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for m in r["results"]["A"]:
            row = [r["results"][c][m][sl] for c in CONDITIONS]
            lines.append(
                f"| {m} | "
                + " | ".join(f"{x['ndcg@10']:.4f}" for x in row)
                + " | "
                + " | ".join(f"{x['auc']:.4f}" for x in row)
                + " |"
            )
    lines += [
        "",
        "## Decision",
        "",
        f"Rule (fixed before running): {d['rule']}.",
        "",
        f"- Relative nDCG@10 loss of {gate_real.MODEL}, overall: B {d['relative_loss_B']:.2%}, "
        f"C {d['relative_loss_C']:.2%}.",
        f"- {gate_real.MODEL} beats {gate_real.BASELINE} in C: "
        f"{'yes' if d['model_beats_baseline_in_C'] else 'no'}.",
        f"- **Title-only articles are {outcome}.**",
        "",
    ]
    md = out_dir / "skew-title-only-mind-small.md"
    js = out_dir / "skew-title-only-mind-small.json"
    md.write_text("\n".join(lines))
    js.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    return js, md
