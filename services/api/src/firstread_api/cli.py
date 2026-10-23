"""``firstread-api``: export the OpenAPI spec (P4 generates its client from it).

uv run firstread-api openapi            # writes services/api/openapi.json
uv run firstread-api openapi --check    # exit 1 if the committed spec is stale
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parents[2] / "openapi.json"


def spec() -> dict[str, Any]:
    """The app's OpenAPI document, with settings that don't depend on the environment."""
    from firstread_api.main import create_app
    from firstread_api.testing import make_settings

    out: dict[str, Any] = create_app(make_settings()).openapi()
    return out


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="firstread-api", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("openapi", help="write (or --check) services/api/openapi.json")
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", type=Path, default=SPEC_PATH)
    args = parser.parse_args(argv)
    text = render(spec())
    if args.check:
        current = args.out.read_text() if args.out.is_file() else ""
        if current != text:
            print(f"{args.out} is stale: run `uv run firstread-api openapi`")
            raise SystemExit(1)
        print(f"{args.out} is up to date")
        return
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
