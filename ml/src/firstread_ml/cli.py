"""``firstread-ml``: one entry point for the offline ML pipeline.

uv run firstread-ml download --variant small --accept-license
"""

import argparse
from collections.abc import Sequence


def _download(args: argparse.Namespace) -> None:
    from firstread_ml.mind.download import download

    download(args.variant, args.accept_license)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firstread-ml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download + verify MIND (needs --accept-license)")
    p.add_argument("--variant", choices=["small", "large"], default="small")
    p.add_argument(
        "--accept-license",
        action="store_true",
        help="accept the Microsoft Research License Terms for MIND (printed without this flag)",
    )
    p.set_defaults(func=_download)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
