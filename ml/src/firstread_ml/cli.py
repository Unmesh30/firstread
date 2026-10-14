"""``firstread-ml``: one entry point for the offline ML pipeline.

uv run firstread-ml download --variant small --accept-license
uv run firstread-ml eval --variant small --split dev
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_BASELINES = "random,popularity,recency,mean_history"


def _download(args: argparse.Namespace) -> None:
    from firstread_ml.mind.download import download

    download(args.variant, args.accept_license)


def _prepare(args: argparse.Namespace) -> None:
    from firstread_ml.mind.load import prepare

    prepare(args.variant)


def _embed(args: argparse.Namespace) -> None:
    from firstread_ml.embed_cache import load_or_build
    from firstread_ml.mind.load import load_news, prepare

    prepare(args.variant)
    emb = load_or_build(args.variant, load_news(args.variant))
    print(f"embeddings: {emb.shape} {emb.dtype}")


def _eval(args: argparse.Namespace) -> None:
    from firstread_ml.eval import harness, report
    from firstread_ml.paths import data_dir, reports_dir

    models = args.models.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    result = harness.run(
        args.variant, args.split, models, seed=args.seed, nrms_config=args.config, nrms_seeds=seeds
    )
    for line in report.acceptance(result):
        print(line)
    try:
        j, m = report.write(result, reports_dir())
    except report.IncompleteReportError as e:
        # Partial runs (e.g. --models nrms_lite) go to gitignored scratch, never ml/reports.
        scratch = data_dir() / "reports-scratch"
        print(f"{e}; writing to {scratch} instead of ml/reports")
        j, m = (
            scratch / f"results-mind-{args.variant}-{args.split}.{ext}" for ext in ("json", "md")
        )
        scratch.mkdir(parents=True, exist_ok=True)
        j.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        m.write_text(report.to_markdown(result))
    print(f"wrote {j} and {m}")


def _tune(args: argparse.Namespace) -> None:
    from firstread_ml.config import load_config
    from firstread_ml.models.scorer import DEFAULT_CONFIG
    from firstread_ml.models.tune import apply_overrides, tune

    cfg = apply_overrides(load_config(args.config or DEFAULT_CONFIG), args.set)
    tune(args.variant, cfg, args.label)


def _classify(args: argparse.Namespace) -> None:
    from firstread_ml.models.classifier import run

    r = run(args.variant)
    t = r["test"]
    print(
        f"classifier: accuracy {t['accuracy']:.4f}, macro-F1 {t['macro_f1']:.4f} "
        f"on {r['n_test_dev_only']:,} dev-only articles"
    )


def _topics(args: argparse.Namespace) -> None:
    from firstread_ml.models.topic_classifier import run

    r = run(args.variant)
    for mode, t in r["test"].items():
        print(
            f"topics ({mode}): accuracy {t['accuracy']:.4f}, macro-F1 {t['macro_f1']:.4f} "
            f"on {r['n_test_dev_only']:,} dev-only articles"
        )
    verdict = "PASS" if r["gate"]["passed"] else "FAIL"
    print(f"low-confidence threshold {r['threshold']:.2f}; gate {verdict}")
    for f in r["gate"]["failures"]:
        print(f"  GATE FAIL {f}")


def _skew(args: argparse.Namespace) -> None:
    from firstread_ml.eval import skew

    r = skew.run(args.release / "user_encoder.onnx")
    j, m = skew.write_report(r)
    print(json.dumps(r["decision"], indent=1))
    print(f"wrote {j} and {m}")


def _eda(args: argparse.Namespace) -> None:
    from firstread_ml.eda import run

    print(f"wrote {run(args.variant)}")


def _eval_gate(args: argparse.Namespace) -> None:
    if not args.real:
        from firstread_ml.eval import gate

        raise SystemExit(gate.main(update=args.update))
    from firstread_ml.eval import gate_real
    from firstread_ml.paths import data_dir

    out = data_dir() / "release" / "eval-gate-real-last.json"
    code, _ = gate_real.main(args.release / "user_encoder.onnx", out, update=args.update)
    raise SystemExit(code)


def _release(args: argparse.Namespace) -> None:
    from firstread_ml import release
    from firstread_ml.paths import data_dir

    try:
        if args.action == "build":
            sources = release.default_sources(inherit=args.inherit_user_encoder)
            release.build(args.version, sources, dry_run=args.dry_run)
        elif args.action == "upload":
            release.upload(args.version, data_dir() / "release" / f"v{args.version}")
        elif args.action == "verify":
            release.verify_remote(args.version)
        else:
            release.verify_tag(args.version)
    except release.ReleaseError as e:
        print(f"RELEASE FAIL: {e}")
        raise SystemExit(1) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firstread-ml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def variant_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--variant", choices=["small", "large"], default="small")

    p = sub.add_parser("download", help="download + verify MIND (needs --accept-license)")
    variant_arg(p)
    p.add_argument(
        "--accept-license",
        action="store_true",
        help="accept the Microsoft Research License Terms for MIND (printed without this flag)",
    )
    p.set_defaults(func=_download)

    p = sub.add_parser("prepare", help="parse MIND TSVs into parquet (cached)")
    variant_arg(p)
    p.set_defaults(func=_prepare)

    p = sub.add_parser("embed", help="embed all articles with the shared embedder (cached)")
    variant_arg(p)
    p.set_defaults(func=_embed)

    p = sub.add_parser("eval", help="evaluate models; writes ml/reports/results-*.{md,json}")
    variant_arg(p)
    p.add_argument("--split", choices=["holdout", "dev"], default="dev")
    p.add_argument("--models", default=DEFAULT_BASELINES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, default=None, help="NRMS-lite YAML config")
    p.add_argument("--seeds", default="0", help="NRMS-lite seeds, e.g. 0,1,2 (one model each)")
    p.set_defaults(func=_eval)

    p = sub.add_parser("tune", help="train on fit days, eval holdout every epoch (no dev)")
    variant_arg(p)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=0.0003")
    p.add_argument("--label", required=True, help="run name in the tuning log")
    p.set_defaults(func=_tune)

    p = sub.add_parser("classify", help="train + evaluate the category classifier")
    variant_arg(p)
    p.set_defaults(func=_classify)

    p = sub.add_parser("topics", help="train + evaluate the topic classifier (ADR 0006)")
    p.add_argument("--variant", choices=["small", "large"], default="large")
    p.set_defaults(func=_topics)

    p = sub.add_parser("skew", help="title-only vs title+abstract study on MIND-small dev")
    p.add_argument(
        "--release", type=Path, default=Path("data/release/v0.1.0"), help="release folder"
    )
    p.set_defaults(func=_skew)

    p = sub.add_parser("eda", help="aggregate EDA report -> ml/reports/eda-mind-<variant>.md")
    variant_arg(p)
    p.set_defaults(func=_eda)

    p = sub.add_parser("eval-gate", help="eval-regression gate (synthetic; --real for MIND-small)")
    p.add_argument("--update", action="store_true", help="rewrite the expected values")
    p.add_argument(
        "--real", action="store_true", help="gate an exported ONNX user encoder on MIND-small dev"
    )
    p.add_argument(
        "--release", type=Path, default=Path("data/release/dry-run"), help="release folder"
    )
    p.set_defaults(func=_eval_gate)

    p = sub.add_parser("release", help="build / upload / verify a model release (ADR 0005)")
    p.add_argument("action", choices=["build", "upload", "verify", "verify-tag"])
    p.add_argument("--version", required=True, help="e.g. 0.1.0 (tag v0.1.0)")
    p.add_argument("--dry-run", action="store_true", help="build on any branch; never uploadable")
    p.add_argument(
        "--inherit-user-encoder",
        metavar="VERSION",
        default=None,
        help="copy the user encoder from release VERSION (e.g. 0.1.0); ships the topic classifier",
    )
    p.set_defaults(func=_release)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
