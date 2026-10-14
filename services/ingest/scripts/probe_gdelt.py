"""Manual probe of the GDELT DOC 2.0 API (never run in CI).

Sends one artlist request every ``--spacing`` seconds (default 6 min) and
records, for each call: the query window, HTTP status, latency, whether the
body was JSON, the number of articles and the field names. The first
successful JSON response is saved under ``data/probes/gdelt/`` (gitignored) so
a sanitized fixture can be cut from it. A summary JSON is written at the end.

    uv run python services/ingest/scripts/probe_gdelt.py --calls 10

Be gentle: the API throttles hard (Phase 2 plan), and every call here is serial.
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = "FirstRead/0.1 (+https://github.com/Unmesh30/firstread)"
OUT = Path(__file__).resolve().parents[3] / "data" / "probes" / "gdelt"
QUERY = "sourcelang:english sourcecountry:US"


def windows(now: datetime) -> list[dict[str, str]]:
    """The parameter sets to try, in order (cycled if there are more calls)."""
    end = (now - timedelta(minutes=20)).strftime("%Y%m%d%H%M%S")

    def start(minutes: int) -> str:
        return (now - timedelta(minutes=20 + minutes)).strftime("%Y%m%d%H%M%S")

    return [
        {"timespan": "1h"},
        {"timespan": "30min"},
        {"startdatetime": start(30), "enddatetime": end},
        {"timespan": "20min"},
        {"timespan": "2h"},
        {"timespan": "15min"},
        {"startdatetime": start(60), "enddatetime": end},
    ]


def call(client: httpx.Client, params: dict[str, str]) -> dict[str, Any]:
    full = {
        "query": QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": "250",
        "sort": "datedesc",
        **params,
    }
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "params": params}
    try:
        r = client.get(URL, params=full)
    except httpx.HTTPError as e:
        return rec | {"error": type(e).__name__, "seconds": round(time.perf_counter() - t0, 1)}
    rec |= {
        "status": r.status_code,
        "seconds": round(time.perf_counter() - t0, 1),
        "content_type": r.headers.get("content-type"),
        "retry_after": r.headers.get("retry-after"),
        "bytes": len(r.content),
    }
    try:
        data = r.json()
    except ValueError:
        return rec | {"json": False, "body_head": r.text[:120]}
    arts = data.get("articles", []) if isinstance(data, dict) else []
    rec |= {"json": True, "n_articles": len(arts)}
    if arts:
        rec |= {
            "fields": sorted({k for a in arts for k in a}),
            "seendate_min": min(a.get("seendate", "") for a in arts),
            "seendate_max": max(a.get("seendate", "") for a in arts),
            "languages": sorted({a.get("language", "") for a in arts}),
            "countries": sorted({a.get("sourcecountry", "") for a in arts}),
            "with_socialimage": sum(bool(a.get("socialimage")) for a in arts),
            "distinct_domains": len({a.get("domain") for a in arts}),
        }
        raw = OUT / f"artlist-{rec['at'].replace(':', '')}.json"
        raw.write_text(json.dumps(data))
        rec["saved"] = str(raw.name)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--spacing", type=float, default=360.0, help="seconds between calls")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=httpx.Timeout(90.0)) as client:
        for i in range(args.calls):
            params = windows(datetime.now(UTC))[i % 7]
            rec = call(client, params)
            records.append(rec)
            print(json.dumps(rec), flush=True)
            if i + 1 < args.calls:
                time.sleep(args.spacing)
    summary = {
        "query": QUERY,
        "spacing_s": args.spacing,
        "calls": records,
        "ok": sum(1 for r in records if r.get("json")),
        "http_429": sum(1 for r in records if r.get("status") == 429),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"ok {summary['ok']}/{len(records)}, 429s {summary['http_429']}")


if __name__ == "__main__":
    main()
