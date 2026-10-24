"""Junk-title rule validation (ADR 0007, P3e amendment): AP digests, dated forecasts.

    uv run python services/api/scripts/persona_replay.py run --out data/persona-replay/junk-titles
    uv run python services/api/scripts/junk_titles_report.py scan     # ~40 s, cached GKG files
    uv run python services/api/scripts/junk_titles_report.py sample   # samples to label
    uv run python services/api/scripts/junk_titles_report.py report   # aggregates

``persona_replay.py run`` re-runs the 3-day replay (fresh ``firstread_replay``) with
the rule and keeps its per-run summaries, so the drops by pattern are the pipeline's
own counts. ``scan`` passes every cached GKG record through the same source filters
and ``normalize`` without the rest of the pipeline: the drops with their titles, and
the near-misses (kept titles that look like the two families). ``sample`` draws 30
drops, 30 near-misses and 40 story-cap drops of the rank-3 persona pages. Labels go in
``data/junk-titles/labels.json``. Headlines stay in ``data/junk-titles/`` (gitignored)
and the PR description; ``report`` writes aggregates only to
``docs/learn/data/03-junk-titles.json`` (03-replay.json and 03-personas.json are cited
by ADRs and stay as they are).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from firstread_ingest.config import Settings, load_junk_titles
from firstread_ingest.normalize import normalize
from firstread_ingest.replay import CACHE, RUN_DELAY
from firstread_ingest.sources import gkg
from firstread_ingest.sources.gdelt_domains import load_domain_countries, lookup_path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "junk-titles"
SCAN = OUT / "scan.jsonl"
SAMPLES = OUT / "samples.json"
LABELS = OUT / "labels.json"
RUN = REPO / "data" / "persona-replay" / "junk-titles" / "report.json"
BEFORE_REPLAY = REPO / "docs" / "learn" / "data" / "03-replay.json"
BEFORE_PERSONAS = REPO / "docs" / "learn" / "data" / "03-personas.json"
AGGREGATES = REPO / "docs" / "learn" / "data" / "03-junk-titles.json"
RANKING = "rank-3"
N_DROPS, N_NEAR, CAP_PER_PERSONA = 30, 30, 8
# Kept titles that look like either family: what the rule must leave alone.
NEAR = {
    "ap_digest": re.compile(r"\bap news\b|\bnews (summary|in brief|digest)\b", re.I),
    "dated_weather_forecast": re.compile(
        r"\bweather\b.*\bforecast|\bforecast\b.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov"
        r"|dec)[a-z]*\.? \d{1,2}\b",
        re.I,
    ),
}


def scan() -> None:
    places = load_domain_countries(lookup_path(None))
    junk = load_junk_titles()
    min_words = Settings().ingest_min_title_words
    reasons: Counter[str] = Counter()
    OUT.mkdir(parents=True, exist_ok=True)
    with SCAN.open("w") as f:
        for path in sorted(CACHE.glob("*.gkg.csv.zip")):
            slot = gkg.parse_slot(path.name.split(".")[0])
            for rec in gkg.parse_gkg(path.read_bytes()):
                # fetch_gkg's source filters, in its order
                if (
                    places.blocked(rec.domain)
                    or places.press_wire(rec.domain)
                    or places.press_release_url(rec.domain, rec.url)
                    or (fips := places.fips(rec.domain)) is None
                ):
                    continue
                out = normalize(gkg.to_raw(rec, fips), slot + RUN_DELAY, min_words, junk)
                reason = out if isinstance(out, str) else "kept"
                reasons[reason] += 1
                title = rec.title if isinstance(out, str) else out.title
                near = [k for k, rx in NEAR.items() if rx.search(title)]
                if reason.startswith("junk_title:") or (reason == "kept" and near):
                    row = {"slot": gkg.slot_name(slot), "reason": reason, "family": near,
                           "title": title, "domain": rec.domain}  # fmt: skip
                    f.write(json.dumps(row) + "\n")
    (OUT / "scan-reasons.json").write_text(json.dumps(dict(reasons), indent=1) + "\n")
    print(dict(reasons))


def distinct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        out.setdefault((r["title"].casefold(), r["domain"]), r)
    return list(out.values())


def cap_drops(times: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up to 8 distinct rank-3 story-cap drops per persona (steady times), τ_feed
    merges first, as in persona_replay.pick_samples."""
    out = []
    for persona in sorted({p for t in times for p in t["personas"][RANKING]}):
        seen: set[tuple[str, ...]] = set()
        pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in times:
            if not t["steady"]:
                continue
            for d in t["personas"][RANKING].get(persona, {}).get("drops", []):
                key = (d["dropped"]["id"], *sorted(k["id"] for k in d["kept"]))
                if key not in seen:
                    seen.add(key)
                    pool[d["via"]].append({**d, "persona": persona, "at": t["now"]})
        rng = random.Random(0)  # noqa: S311 (sampling, not security)
        chosen = rng.sample(pool["tau_feed"], min(CAP_PER_PERSONA, len(pool["tau_feed"])))
        rest = CAP_PER_PERSONA - len(chosen)
        chosen += rng.sample(pool["cluster"], min(rest, len(pool["cluster"])))
        out += chosen
    return out


def sample() -> None:
    rows = [json.loads(line) for line in SCAN.open()]
    rng = random.Random(0)  # noqa: S311 (sampling, not security)
    drops = distinct([r for r in rows if r["reason"].startswith("junk_title:")])
    near = distinct([r for r in rows if r["reason"] == "kept"])
    picked: dict[str, list[dict[str, Any]]] = {"drops": [], "near_misses": []}
    # Half of each sample per family (or all of a family when it has fewer).
    for key, pool, n in (("drops", drops, N_DROPS), ("near_misses", near, N_NEAR)):
        by_family = {
            fam: [r for r in pool if fam in r["reason"] or fam in r["family"]] for fam in NEAR
        }
        small, big = sorted(by_family, key=lambda f: len(by_family[f]))
        take_small = min(n // 2, len(by_family[small]))
        got = rng.sample(by_family[small], take_small)
        rest = [r for r in by_family[big] if r not in got]
        got += rng.sample(rest, min(n - take_small, len(rest)))
        picked[key] = sorted(got, key=lambda r: (r["reason"], r["title"]))
    # Seeded: drawing again once the persona replay has run gives the same drops and
    # near-misses.
    if RUN.is_file():
        picked["cap_drops"] = cap_drops(json.loads(RUN.read_text())["times"])
    for items in picked.values():
        for k, s in enumerate(items):
            s["n"] = k + 1
    SAMPLES.write_text(json.dumps(picked, indent=1) + "\n")
    print({k: len(v) for k, v in picked.items()}, f"-> {SAMPLES}")


def label_counts(items: list[dict[str, Any]], labels: dict[str, str], by: str) -> dict[str, Any]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for s in items:
        if (lab := labels.get(str(s["n"]))) is not None:
            out[by_key(s, by)][lab] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def by_key(s: dict[str, Any], by: str) -> str:
    if by == "reason":
        return str(s["reason"])
    if by == "family":
        return ",".join(s["family"]) or "none"
    return f"{s['via']}:{s['kept'][0]['pair_type']}"


def report() -> None:
    run = json.loads(RUN.read_text())
    runs, times = run["runs"], run["times"]
    before = json.loads(BEFORE_REPLAY.read_text())
    totals: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    records = 0
    for r in runs:
        for k in ("inserted", "duplicates", "near_duplicates", "clustered", "errors"):
            totals[k] += r[k]
        dropped.update({k: v for k, v in r["dropped"].items() if k.startswith("junk_title")})
        records += r["gdelt"]["records"] or 0
    junk = sum(v for k, v in dropped.items() if k.startswith("junk_title:"))
    scan_reasons = json.loads((OUT / "scan-reasons.json").read_text())
    samples = json.loads(SAMPLES.read_text())
    labels = json.loads(LABELS.read_text())
    steady = [t for t in times if t["steady"]]
    pages = [t["personas"][RANKING][p] for t in steady for p in t["personas"][RANKING]]
    cap = [d for x in pages for d in x.get("drops", [])]
    before_p = json.loads(BEFORE_PERSONAS.read_text())["all_personas"][RANKING]
    rows = [json.loads(line) for line in SCAN.open()]
    agg = {
        "rule": {
            "config": "services/ingest/src/firstread_ingest/config/junk_titles.yaml (patterns)",
            "patterns": [name for name, _ in load_junk_titles().patterns],
            "drop_reason": "junk_title:<pattern>, before dedupe",
        },
        "replay": {
            "window": {"first": runs[0]["slot"], "last": runs[-1]["slot"], "slots": len(runs)},
            "gdelt_records": records,
            "dropped_by_pattern": {
                k.split(":", 1)[1]: v for k, v in sorted(dropped.items()) if ":" in k
            },
            "dropped_junk_title_total": junk,
            "share_of_gdelt_records": round(junk / max(records, 1), 6),
            "totals_with_rule": dict(totals),
            "totals_before (03-replay.json)": before["totals"],
            "statuses": dict(Counter(r["status"] for r in runs)),
            "counting": "dropped_by_pattern sums the run summaries, which count a GKG file "
            "each time it is read: at the window's start the 7 earlier slots (not cached) stay "
            "pending for 2 h, so the first 6 files are re-read up to 7 times (as in "
            "03-replay.json). distinct_records counts each file once.",
            "distinct_records_by_pattern": {
                k.split(":", 1)[1]: v
                for k, v in scan_reasons.items()
                if k.startswith("junk_title:")
            },
            "scan_distinct_titles_dropped": len(
                {r["title"].casefold() for r in rows if r["reason"].startswith("junk_title:")}
            ),
        },
        "sampled_drops": {
            "n": len(samples["drops"]),
            "labels": label_counts(samples["drops"], labels.get("drops", {}), "reason"),
        },
        "sampled_near_misses": {
            "n": len(samples["near_misses"]),
            "labels": label_counts(samples["near_misses"], labels.get("near_misses", {}), "family"),
        },
        "persona_replay": {
            "ranking": RANKING,
            "steady_clock_times": len(steady),
            "pages": len(pages),
            "cap_drops_slate": len(cap),
            "cap_drops_slate_mean": round(len(cap) / max(len(pages), 1), 2),
            "cap_drops_by_way": dict(Counter(d["via"] for d in cap)),
            "cap_drops_slate_mean_before (03-personas.json)": before_p["cap_drops_slate_mean"],
            "cap_drops_by_way_before (03-personas.json)": before_p["cap_drops_by_way"],
            "junk_titles_in_cap_drops": sum(
                1
                for d in cap
                for t in [d["dropped"], *d["kept"]]
                if any(rx.search(t["title"]) for _, rx in load_junk_titles().patterns)
            ),
            "sampled_cap_drops": {
                "n": len(samples["cap_drops"]),
                "by_way": dict(Counter(s["via"] for s in samples["cap_drops"])),
                "labels": label_counts(samples["cap_drops"], labels.get("cap_drops", {}), "way"),
            },
        },
        "labelled_by": labels.get("labelled_by"),
        "spot_check": labels.get("spot_check"),
        "note": "headlines stay in data/junk-titles/ (gitignored) and the PR description",
    }
    AGGREGATES.write_text(json.dumps(agg, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {AGGREGATES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scan", "sample", "report"))
    args = parser.parse_args()
    {"scan": scan, "sample": sample, "report": report}[args.command]()


if __name__ == "__main__":
    main()
