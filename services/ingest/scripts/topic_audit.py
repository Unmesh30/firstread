"""Steady-state topic audit (manual, never CI).

``sample``: draws GDELT articles from the replay database (stratified by predicted
topic; half from US outlets, half from non-US by ``geo``) and RSS articles from the
dev database (a live RSS run), into ``data/validation/topic_audit_2b.jsonl`` with
``true: null``. Label each ``true`` with the ADR 0006 meanings (us/politics = US
domestic, world = outside the US, the rest global subjects).

``score``: accuracy reweighted to each origin's live topic mix (stratified estimator,
Agresti-Coull-adjusted variance per stratum), with n and a 95% interval, by origin,
by US/non-US outlet within GDELT, and combined by the measured origin shares
(GDELT/day from the replay report, RSS/day from its live estimate). Also per-topic
results, top confusions, and how often the geo rule fired and was right. Aggregates
go to ``docs/learn/data/02-topic-audit-2b.json`` (no headlines).

    uv run python services/ingest/scripts/topic_audit.py sample --gdelt 90 --rss 30
    uv run python services/ingest/scripts/topic_audit.py score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from firstread_core.db import connect

REPO = Path(__file__).resolve().parents[3]
SAMPLE = REPO / "data" / "validation" / "topic_audit_2b.jsonl"
POPULATION = REPO / "data" / "validation" / "topic_audit_2b_population.json"
REPLAY_REPORT = REPO / "data" / "replay" / "report.json"
OUT = REPO / "docs" / "learn" / "data" / "02-topic-audit-2b.json"
DEV = "postgresql://firstread:firstread@localhost:5432/firstread"
REPLAY = "postgresql://firstread:firstread@localhost:5432/firstread_replay"
QUERY = (
    "select id::text as id, title, source_domain, origin, geo, category, category_source, "
    "category_confidence::float8 as conf from public.articles where origin = $1"
)


def allocate(by: dict[str, list[Any]], n: int) -> dict[str, int]:
    """At least 1 per predicted topic, then largest remainder by population."""
    alloc = {t: 1 for t in by}
    while sum(alloc.values()) < min(n, sum(len(v) for v in by.values())):
        t = max((t for t in by if alloc[t] < len(by[t])), key=lambda t: len(by[t]) / alloc[t])
        alloc[t] += 1
    return alloc


def draw(
    rows: list[dict[str, Any]], n: int, rng: random.Random, group: str
) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    out = []
    for t, k in allocate(by, n).items():
        out += [{**r, "group": group, "true": None} for r in rng.sample(by[t], k)]
    return out


async def sample(n_gdelt: int, n_rss: int, seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 (reproducible sampling)
    replay = await connect(REPLAY)
    dev = await connect(DEV)
    try:
        gdelt = [dict(r) for r in await replay.fetch(QUERY, "gdelt")]
        rss = [dict(r) for r in await dev.fetch(QUERY, "rss")]
    finally:
        await replay.close()
        await dev.close()
    us = [r for r in gdelt if r["geo"] == "us"]
    intl = [r for r in gdelt if r["geo"] == "intl"]
    picked = draw(us, n_gdelt // 2, rng, "gdelt-us")
    picked += draw(intl, n_gdelt - n_gdelt // 2, rng, "gdelt-intl")
    picked += draw(rss, n_rss, rng, "rss")
    population = {
        "gdelt-us": Counter(r["category"] for r in us),
        "gdelt-intl": Counter(r["category"] for r in intl),
        "rss": Counter(r["category"] for r in rss),
    }
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text("".join(json.dumps(r) + "\n" for r in picked))
    POPULATION.write_text(json.dumps({k: dict(v) for k, v in population.items()}) + "\n")
    print(f"wrote {len(picked)} items -> {SAMPLE}: {dict(Counter(r['group'] for r in picked))}")


def stratified(rows: list[dict[str, Any]], pop: dict[str, int]) -> dict[str, Any]:
    """Reweighted accuracy over predicted-topic strata, with a 95% interval."""
    total = sum(pop.values())
    by: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r["true"] == r["category"])
    est = var = 0.0
    covered = 0
    for t, xs in by.items():
        w = pop.get(t, 0) / total
        covered += pop.get(t, 0)
        n, x = len(xs), sum(xs)
        p_adj = (x + 2) / (n + 4)  # no zero-variance strata
        est += w * x / n
        var += w * w * p_adj * (1 - p_adj) / n
    se = math.sqrt(var)
    return {
        "n": len(rows),
        "population": total,
        "population_covered": covered,
        "accuracy": round(est, 3),
        "ci95": [round(max(0.0, est - 1.96 * se), 3), round(min(1.0, est + 1.96 * se), 3)],
        "se": se,
        "unweighted": round(sum(r["true"] == r["category"] for r in rows) / len(rows), 3),
    }


def score() -> None:
    rows = [json.loads(line) for line in SAMPLE.read_text().splitlines()]
    if any(r["true"] is None for r in rows):
        raise SystemExit("label every item's 'true' topic first")
    pop: dict[str, dict[str, int]] = json.loads(POPULATION.read_text())
    groups = {g: [r for r in rows if r["group"] == g] for g in ("gdelt-us", "gdelt-intl", "rss")}
    out: dict[str, Any] = {"groups": {g: stratified(rs, pop[g]) for g, rs in groups.items()}}
    # GDELT as a whole: the two geo halves weighted by their populations.
    n_us, n_intl = sum(pop["gdelt-us"].values()), sum(pop["gdelt-intl"].values())
    w_us = n_us / (n_us + n_intl)
    gu, gi = out["groups"]["gdelt-us"], out["groups"]["gdelt-intl"]
    g_acc = w_us * gu["accuracy"] + (1 - w_us) * gi["accuracy"]
    g_se = math.sqrt((w_us * gu["se"]) ** 2 + ((1 - w_us) * gi["se"]) ** 2)
    out["gdelt"] = {"accuracy": round(g_acc, 3), "ci95": [round(g_acc - 1.96 * g_se, 3),
                    round(g_acc + 1.96 * g_se, 3)], "share_us": round(w_us, 3)}  # fmt: skip
    report = json.loads(REPLAY_REPORT.read_text())
    gdelt_day = report["size_estimate"]["gdelt_only"]["articles_per_day"]
    rss_day = report["rss_estimate"]["per_day"]
    w_g = gdelt_day / (gdelt_day + rss_day)
    r = out["groups"]["rss"]
    c_acc = w_g * g_acc + (1 - w_g) * r["accuracy"]
    c_se = math.sqrt((w_g * g_se) ** 2 + ((1 - w_g) * r["se"]) ** 2)
    out["combined"] = {"accuracy": round(c_acc, 3), "ci95": [round(c_acc - 1.96 * c_se, 3),
                       round(c_acc + 1.96 * c_se, 3)], "share_gdelt": round(w_g, 3)}  # fmt: skip
    per_topic: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in rows:
        k = f"{row['group']}:{row['category']}"
        per_topic[k]["n"] += 1
        per_topic[k]["correct"] += row["true"] == row["category"]
    out["per_topic"] = dict(sorted(per_topic.items()))
    out["confusions"] = [
        {"group": g, "predicted": p, "true": t, "n": n}
        for (g, p, t), n in Counter(
            (row["group"], row["category"], row["true"])
            for row in rows
            if row["true"] != row["category"]
        ).most_common()
    ]
    geo = [row for row in rows if row["category_source"] == "geo_rule"]
    out["geo_rule"] = {
        "fired": len(geo),
        "correct": sum(row["true"] == row["category"] for row in geo),
    }
    out["notes"] = [
        "LLM-labeled; spot-checked by a second, independent LLM session, approved by the "
        "owner (see ADR 0009 for the outcome).",
        "GDELT from a 3-day replay of archived GKG files; RSS from a live run "
        "(RSS can't be replayed).",
    ]
    for g in out["groups"].values():
        g.pop("se")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("groups", "gdelt", "combined", "geo_rule")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--gdelt", type=int, default=90)
    s.add_argument("--rss", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    args = ap.parse_args()
    if args.cmd == "sample":
        asyncio.run(sample(args.gdelt, args.rss, args.seed))
    else:
        score()


if __name__ == "__main__":
    main()
