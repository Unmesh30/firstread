"""Sample article pairs for validating the dedupe/clustering thresholds (manual, never CI).

Two sources of candidates:

* live (default): the enabled RSS feeds and the last ``--files`` GKG files;
* ``--from-replay``: every placeable record in the newest ``--hours`` of cached
  replay GKG files (``data/replay/gkg``), i.e. the candidate stream ingest saw,
  before dedupe and the per-run cap, plus the live RSS feeds.

Items are normalized (incl. the press-release filter) and embedded. For each item the
nearest neighbours of each text type are found (chunked, so 15K+ items fit in memory).
Pairs are stratified by pair type (title<->title = GDELT-GDELT, title_summary<->
title_summary = RSS-RSS, title<->title_summary = GDELT-RSS) and cosine bin
(0.70-1.00, step 0.025). With ``--per-story`` (default 3), at most that many pairs
per story and pair type are kept (stories = connected components of pairs >= 0.80),
so a few big stories can't dominate the sample.

Output: ``data/validation/pairs.jsonl`` (or ``--out``; gitignored, it contains
headlines). Label each pair, then run ``score_pairs.py``.

    uv run python services/ingest/scripts/sample_pairs.py --from-replay --hours 4 \
        --per-bin 20 --out data/validation/pairs-2b.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from firstread_core.embedding import Embedder
from firstread_ingest.config import load_feeds
from firstread_ingest.dedupe import pair_type
from firstread_ingest.http import make_client
from firstread_ingest.items import Candidate
from firstread_ingest.normalize import normalize_all
from firstread_ingest.sources import gkg, rss
from firstread_ingest.sources.gdelt_domains import ensure_lookup, load_domain_countries

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "validation"
REPLAY_CACHE = REPO / "data" / "replay" / "gkg"
BINS = np.round(np.arange(0.70, 1.0001, 0.025), 3)
STORY_LINK = 0.80
CHUNK = 1024
TOP_K = 3


def unique(cands: list[Candidate]) -> list[Candidate]:
    seen_url: set[str] = set()
    seen_fp: set[str] = set()
    out = []
    for c in cands:
        if c.url_canonical in seen_url or c.title_fingerprint in seen_fp:
            continue
        seen_url.add(c.url_canonical)
        seen_fp.add(c.title_fingerprint)
        out.append(c)
    return out


async def collect_live(files: int) -> tuple[list[Candidate], datetime]:
    now = datetime.now(UTC)
    async with make_client() as client:
        results = await rss.fetch_all(client, load_feeds(), {}, 8, now)
        places = load_domain_countries(await ensure_lookup(client))
        g = await gkg.fetch_gkg(client, None, now, places, max_files=files)
    raw = [i for r in results for i in r.items] + g.items
    cands, dropped = normalize_all(raw, now)
    print(f"live: fetched {len(raw)}, kept {len(cands)}, dropped {dict(dropped)}")
    return cands, now


async def collect_replay(hours: int) -> tuple[list[Candidate], datetime]:
    files = sorted(REPLAY_CACHE.glob("*.gkg.csv.zip"))[-hours * 4 :]
    newest = gkg.parse_slot(files[-1].name.split(".")[0])
    now = newest + timedelta(minutes=20)
    async with make_client() as client:
        places = load_domain_countries(await ensure_lookup(client))
        results = await rss.fetch_all(client, load_feeds(), {}, 8, datetime.now(UTC))
    raw = []
    dropped_src: Counter[str] = Counter()
    for f in files:
        for rec in gkg.parse_gkg(f.read_bytes()):
            if places.blocked(rec.domain):
                dropped_src["blocked_domain"] += 1
            elif places.press_wire(rec.domain):
                dropped_src["press_wire_domain"] += 1
            elif (fips := places.fips(rec.domain)) is None:
                dropped_src["unplaced_domain"] += 1
            else:
                raw.append(gkg.to_raw(rec, fips))
    gdelt, dropped = normalize_all(raw, now)
    # RSS is fetched live: judge its staleness against the wall clock, not the replay's.
    rss_raw = [i for r in results for i in r.items]
    rss_c, rss_dropped = normalize_all(rss_raw, datetime.now(UTC))
    print(
        f"replay: {len(files)} files ({gkg.slot_name(newest)} newest), GDELT kept {len(gdelt)} "
        f"(source drops {dict(dropped_src)}, normalize drops {dict(dropped)}); "
        f"RSS kept {len(rss_c)} (drops {dict(rss_dropped)})"
    )
    return gdelt + rss_c, now


def neighbours(vecs: np.ndarray, kinds: np.ndarray) -> list[tuple[int, int, float]]:
    """Top-K nearest neighbours of each item within each text type, chunked."""
    pairs: list[tuple[int, int, float]] = []
    for want in ("title", "title_summary"):
        cols = np.flatnonzero(kinds == want)
        if len(cols) == 0:
            continue
        target = vecs[cols]
        for start in range(0, len(vecs), CHUNK):
            block = vecs[start : start + CHUNK] @ target.T
            for r in range(block.shape[0]):
                i = start + r
                row = block[r]
                row[cols == i] = -1.0  # not itself
                k = min(TOP_K, len(cols))
                for j in np.argpartition(-row, k - 1)[:k]:
                    if row[j] >= BINS[0]:
                        pairs.append((i, int(cols[j]), float(row[j])))
    return pairs


def stories(n: int, pairs: list[tuple[int, int, float]]) -> list[int]:
    """Union-find over pairs >= STORY_LINK: a story id per item."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, cos in pairs:
        if cos >= STORY_LINK:
            parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-replay", action="store_true")
    ap.add_argument("--hours", type=int, default=4)
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--per-story", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT / "pairs.jsonl")
    args = ap.parse_args()
    collect = collect_replay(args.hours) if args.from_replay else collect_live(args.files)
    cands, _ = asyncio.run(collect)
    cands = unique(cands)
    kinds = np.array([c.text_source for c in cands])
    print(f"embedding {len(cands)} items ({Counter(kinds.tolist())})")
    vecs = Embedder().embed([c.embed_text for c in cands])
    raw_pairs = neighbours(vecs, kinds)
    story = stories(len(cands), raw_pairs)
    rng = random.Random(args.seed)  # noqa: S311 (reproducible sampling)
    rng.shuffle(raw_pairs)
    seen: set[tuple[int, int]] = set()
    per_story: Counter[tuple[int, str]] = Counter()
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for i, j, cos in raw_pairs:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        a, b = cands[i], cands[j]
        ptype = "-".join(pair_type(a.text_source, b.text_source))
        b_idx = int(np.searchsorted(BINS, cos, side="right") - 1)
        bucket = strata[(ptype, b_idx)]
        if len(bucket) >= args.per_bin:
            continue
        if per_story[(story[i], ptype)] >= args.per_story:
            continue
        per_story[(story[i], ptype)] += 1
        bucket.append(
            {
                "pair_type": ptype,
                "cosine": round(cos, 4),
                "bin": float(BINS[b_idx]),
                "story": story[i],
                "a": {"title": a.title, "domain": a.raw.source_domain, "origin": a.raw.origin,
                      "url": a.url_canonical},
                "b": {"title": b.title, "domain": b.raw.source_domain, "origin": b.raw.origin,
                      "url": b.url_canonical},
                "label": None,
            }
        )  # fmt: skip
    picked = [p for _, ps in sorted(strata.items()) for p in ps]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in picked:
            f.write(json.dumps(p) + "\n")
    counts = Counter(p["pair_type"] for p in picked)
    print(f"wrote {len(picked)} pairs -> {args.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
