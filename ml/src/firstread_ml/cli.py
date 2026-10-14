"""``firstread-ml``: one entry point for the offline ML pipeline.

uv run firstread-ml download --variant small --accept-license
uv run firstread-ml eval --variant small --split dev
"""

import argparse
from collections.abc import Sequence

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
    from firstread_ml.paths import reports_dir

    models = args.models.split(",")
    result = harness.run(args.variant, args.split, models, seed=args.seed)
    baselines = [m for m in models if m in DEFAULT_BASELINES.split(",")]
    j, m = report.write(result, reports_dir(), baselines)
    for line in report.acceptance(result, None, baselines):
        print(line)
    print(f"wrote {j} and {m}")


def _eda(args: argparse.Namespace) -> None:
    from firstread_ml.eda import run

    print(f"wrote {run(args.variant)}")


def _eval_gate(args: argparse.Namespace) -> None:
    from firstread_ml.eval import gate

    raise SystemExit(gate.main(update=args.update))


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
    p.set_defaults(func=_eval)

    p = sub.add_parser("eda", help="aggregate EDA report -> ml/reports/eda-mind-<variant>.md")
    variant_arg(p)
    p.set_defaults(func=_eda)

    p = sub.add_parser("eval-gate", help="synthetic eval-regression gate (no MIND needed)")
    p.add_argument("--update", action="store_true", help="rewrite the expected values")
    p.set_defaults(func=_eval_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
