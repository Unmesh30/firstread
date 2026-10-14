"""Score labelled pairs from ``sample_pairs.py`` and suggest dedupe thresholds.

For each pair type and cosine bin: how many pairs were labelled duplicate,
same_story and different. A threshold is accepted when **every** bin at or above
it reaches the precision target (conservative: the sample is stratified by bin,
not proportional, so a pooled precision would over-weight empty-ish bins):

* ``tau_story``: precision of (same_story or duplicate) >= 0.95;
* ``tau_dup``: precision of duplicate >= 0.95.

Writes aggregates only (no headlines) to ``--out`` for the docs.

    uv run python services/ingest/scripts/score_pairs.py --out docs/learn/data/02-thresholds.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = Path(__file__).resolve().parents[3] / "data" / "validation" / "pairs.jsonl"
TARGET = 0.95
MIN_BIN = 3  # bins with fewer labelled pairs don't block a threshold


def lowest_ok(bins: dict[float, Counter[str]], positive: set[str]) -> float | None:
    """Lowest bin edge from which every bin with >= MIN_BIN pairs meets TARGET."""
    edges = sorted(bins)
    best = None
    for lo in reversed(edges):
        c = bins[lo]
        n = sum(c.values())
        if n >= MIN_BIN and sum(c[p] for p in positive) / n < TARGET:
            break
        best = lo
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines()]
    by_type: dict[str, dict[float, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for p in pairs:
        by_type[p["pair_type"]][p["bin"]][p["label"]] += 1
    report: dict[str, object] = {"target_precision": TARGET, "min_bin": MIN_BIN, "types": {}}
    for ptype, bins in sorted(by_type.items()):
        story = lowest_ok(bins, {"same_story", "duplicate"})
        dup = lowest_ok(bins, {"duplicate"})
        table = {f"{lo:.3f}": dict(c) for lo, c in sorted(bins.items())}
        report["types"][ptype] = {  # type: ignore[index]
            "n": sum(sum(c.values()) for c in bins.values()),
            "bins": table,
            "tau_story": story,
            "tau_dup": dup,
        }
        print(f"\n{ptype} (n={report['types'][ptype]['n']})")  # type: ignore[index]
        print("  bin    dup  same  diff   P(story)  P(dup)")
        for lo, c in sorted(bins.items()):
            n = sum(c.values())
            ps = (c["same_story"] + c["duplicate"]) / n
            print(f"  {lo:.3f} {c['duplicate']:4d} {c['same_story']:5d} {c['different']:5d}"
                  f"   {ps:6.2f}    {c['duplicate'] / n:5.2f}")  # fmt: skip
        print(f"  -> tau_story {story}, tau_dup {dup}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
