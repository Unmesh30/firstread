"""Sample article pairs for validating the dedupe/clustering thresholds (manual, never CI).

Fetches the enabled RSS feeds and the last ``--files`` GKG files (no per-run cap),
normalizes and embeds everything, and, for each item, finds its nearest neighbour
with a different URL and headline. Pairs are stratified by **pair type**
(title<->title = GDELT-GDELT, title_summary<->title_summary = RSS-RSS, title<->
title_summary = GDELT-RSS) and by cosine bin (0.70-1.00, step 0.025), and up to
``--per-bin`` pairs are kept per stratum and bin.

Output: ``data/validation/pairs.jsonl`` (gitignored; it contains headlines). Label
each pair's ``label`` as ``duplicate``, ``same_story`` or ``different``, then run
``score_pairs.py``.

    uv run python services/ingest/scripts/sample_pairs.py --files 8 --per-bin 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from firstread_core.embedding import Embedder
from firstread_ingest.config import load_feeds
from firstread_ingest.dedupe import pair_type
from firstread_ingest.http import make_client
from firstread_ingest.normalize import normalize_all
from firstread_ingest.sources import gkg, rss
from firstread_ingest.sources.gdelt_domains import ensure_lookup, load_domain_countries

OUT = Path(__file__).resolve().parents[3] / "data" / "validation"
BINS = np.round(np.arange(0.70, 1.0001, 0.025), 3)


async def collect(files: int) -> list:  # type: ignore[type-arg]
    now = datetime.now(UTC)
    async with make_client() as client:
        results = await rss.fetch_all(client, load_feeds(), {}, 8)
        places = load_domain_countries(await ensure_lookup(client))
        g = await gkg.fetch_gkg(client, None, now, places, max_files=files)
    raw = [i for r in results for i in r.items] + g.items
    cands, dropped = normalize_all(raw, now)
    print(f"fetched {len(raw)}, kept {len(cands)}, dropped {dict(dropped)}")
    seen, uniq = set(), []
    for c in cands:
        if c.url_canonical not in seen:
            seen.add(c.url_canonical)
            uniq.append(c)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--max-gdelt", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cands = asyncio.run(collect(args.files))
    gdelt = [c for c in cands if c.raw.origin == "gdelt"]
    rssc = [c for c in cands if c.raw.origin == "rss"]
    gdelt.sort(key=lambda c: c.raw.seen_at, reverse=True)
    cands = rssc + gdelt[: args.max_gdelt]
    print(
        f"embedding {len(cands)} items ({len(rssc)} RSS, {min(len(gdelt), args.max_gdelt)} GDELT)"
    )
    vecs = Embedder().embed([c.embed_text for c in cands])
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    strata: dict[tuple[tuple[str, str], int], list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    done = set()
    for i, c in enumerate(cands):
        # Nearest neighbour per pair type, so every stratum gets candidates.
        for want in ("title", "title_summary"):
            mask = np.array([d.text_source == want for d in cands])
            row = np.where(mask, sims[i], -1)
            for j in np.argsort(-row)[:3]:
                d = cands[int(j)]
                cos = float(row[j])
                if cos < BINS[0] or d.title_fingerprint == c.title_fingerprint:
                    continue
                key = tuple(sorted((c.url_canonical, d.url_canonical)))
                if key in done:
                    continue
                done.add(key)
                b = int(np.searchsorted(BINS, cos, side="right") - 1)
                strata[(pair_type(c.text_source, d.text_source), b)].append(
                    {
                        "pair_type": "-".join(pair_type(c.text_source, d.text_source)),
                        "cosine": round(cos, 4),
                        "bin": float(BINS[b]),
                        "a": {"title": c.title, "domain": c.raw.source_domain,
                              "origin": c.raw.origin, "url": c.url_canonical},
                        "b": {"title": d.title, "domain": d.raw.source_domain,
                              "origin": d.raw.origin, "url": d.url_canonical},
                        "label": None,
                    }
                )  # fmt: skip
    rng = random.Random(args.seed)  # noqa: S311 (reproducible sampling)
    OUT.mkdir(parents=True, exist_ok=True)
    picked = []
    for (_, _), pairs in sorted(strata.items()):
        rng.shuffle(pairs)
        picked += pairs[: args.per_bin]
    with (OUT / "pairs.jsonl").open("w") as f:
        for p in picked:
            f.write(json.dumps(p) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for p in picked:
        counts[p["pair_type"]] += 1
    print(f"wrote {len(picked)} pairs -> {OUT / 'pairs.jsonl'}: {dict(counts)}")


if __name__ == "__main__":
    main()
