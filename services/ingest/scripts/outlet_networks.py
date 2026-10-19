"""Find syndication networks: domains that carry the same stories (ADR 0013).

One PA wire story republished across dozens of UK local sites counts as dozens of
outlets in trending's breadth score (ADR 0007). This script looks for such networks
in a replay database's ``story_coverage``:

1. keep domains that covered at least ``--min-stories`` stories;
2. for every pair of those domains, the Jaccard similarity of their story sets;
3. link pairs with Jaccard ≥ ``--jaccard``; networks are connected components of at
   least ``--min-size`` domains.

It writes a draft ``outlet_networks.yaml`` for review (the reviewed copy lives in
``firstread_ingest/config/``) plus the evidence to gitignored ``data/replay/``.

    uv run python services/ingest/scripts/outlet_networks.py --db firstread_replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from firstread_core.db import connect
from firstread_core.dbtest import with_database
from firstread_ingest.config import Settings
from firstread_ingest.networks import networks

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "replay"


async def main(db: str, min_stories: int, jaccard: float, min_size: int) -> None:
    settings = Settings()
    conn = await connect(with_database(settings.database_url.get_secret_value(), db))
    try:
        rows = await conn.fetch(
            "select story_cluster_id::text as s, source_domain as d from public.story_coverage"
        )
        span = await conn.fetchrow(
            "select min(first_seen_at) as lo, max(first_seen_at) as hi from public.story_coverage"
        )
    finally:
        await conn.close()
    found, by_domain = networks([(r["s"], r["d"]) for r in rows], min_stories, jaccard, min_size)
    stories_total = len({r["s"] for r in rows})
    summary = {
        "coverage_rows": len(rows),
        "stories": stories_total,
        "window": [str(span["lo"]), str(span["hi"])],
        "params": {"min_stories": min_stories, "jaccard": jaccard, "min_size": min_size},
        "networks": [
            {
                "size": len(n),
                "domains": n,
                "stories_covered": len(set().union(*(by_domain[d] for d in n))),
            }
            for n in found
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "outlet_networks.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# DRAFT from scripts/outlet_networks.py: review before copying to config/.",
        f"# {len(rows)} coverage rows, {stories_total} stories, {summary['window']}.",
        "version: 1",
        "networks:",
    ]
    for k, n in enumerate(found, 1):
        lines.append(f"  network-{k}:  # {len(n)} domains")
        lines += [f"    - {d}" for d in n]
    (OUT / "outlet_networks.draft.yaml").write_text("\n".join(lines) + "\n")
    for k, n in enumerate(found, 1):
        print(f"network-{k}: {len(n)} domains, e.g. {', '.join(n[:6])}")
    print(f"wrote {OUT / 'outlet_networks.json'} and the draft yaml")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="firstread_replay")
    p.add_argument("--min-stories", type=int, default=10)
    p.add_argument("--jaccard", type=float, default=0.5)
    p.add_argument("--min-size", type=int, default=3)
    a = p.parse_args()
    asyncio.run(main(a.db, a.min_stories, a.jaccard, a.min_size))
