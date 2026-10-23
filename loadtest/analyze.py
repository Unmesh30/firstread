"""Summarize a local k6 run into ``loadtest/results/03-local.json`` (aggregates only).

    uv run python loadtest/analyze.py

Reads ``results/raw.csv`` (every sample, from ``--out csv``), ``k6-summary.json``
and the small files ``run_local.sh`` writes, and reports per scenario (50 and 100
requests/s) and request kind: p50/p95/p99 of the client-side duration and of the
server time (the sum of the feed's ``Server-Timing`` stages), the error rate, the
degraded responses and the throughput actually achieved.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SLO_P95_MS = 150  # SPEC §11: local stack, p95 server time at 100 RPS


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))], 1)


def stats(xs: list[float]) -> dict[str, Any]:
    return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99),
            "mean": round(statistics.mean(xs), 1) if xs else None}  # fmt: skip


def tags(row: dict[str, str]) -> dict[str, str]:
    out = {}
    for kv in (row.get("extra_tags") or "").split("&"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def machine() -> dict[str, Any]:
    def sh(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603 (fixed commands)

    return {
        "host": f"{sh('sysctl', '-n', 'machdep.cpu.brand_string')}, {platform.system()}",
        "host_cpus": sh("sysctl", "-n", "hw.ncpu"),
        "docker": sh("docker", "info", "--format", "{{.NCPU}} CPUs, {{.MemTotal}} bytes"),
        "api_container": "--cpus=1 --memory=512m (Cloud Run's shape), one uvicorn worker",
        "db_and_redis": "the same Docker Desktop VM, no CPU limit (not Supabase Nano or Upstash)",
    }


def main() -> None:
    dur: dict[tuple[str, str], list[float]] = defaultdict(list)
    srv: dict[tuple[str, str], list[float]] = defaultdict(list)
    errs: dict[str, list[float]] = defaultdict(list)
    degraded: dict[tuple[str, str], float] = defaultdict(float)
    reqs: dict[str, int] = defaultdict(int)
    with (RES / "raw.csv").open() as f:
        for row in csv.DictReader(f):
            name, sc = row["metric_name"], row.get("scenario") or ""
            kind = tags(row).get("kind", "")
            v = float(row["metric_value"])
            if name == "http_req_duration":
                dur[(sc, kind)].append(v)
                dur[(sc, "all")].append(v)
                reqs[sc] += 1
            elif name == "server_ms":
                srv[(sc, kind)].append(v)
                srv[(sc, "feed")].append(v)
            elif name == "errors":
                errs[sc].append(v)
            elif name.startswith("degraded_") and name != "degraded_responses":
                degraded[(sc, name.removeprefix("degraded_"))] += v
            elif name == "degraded_responses":
                degraded[(sc, "any")] += v
    summary = json.loads((RES / "k6-summary.json").read_text())
    dropped = summary["metrics"].get("dropped_iterations", {}).get("values", {}).get("count", 0)
    scenarios = {}
    for sc, rate in (("rps50", 50), ("rps100", 100)):
        kinds = ("all", "page1", "cursor", "events")
        feed_p95 = pct(srv[(sc, "feed")], 95)
        scenarios[sc] = {
            "target_rps": rate,
            "achieved_rps": round(reqs[sc] / 60, 1),
            "client_ms": {k: stats(dur[(sc, k)]) for k in kinds},
            "server_ms": {k: stats(srv[(sc, k)]) for k in ("feed", "page1", "cursor")},
            "error_rate": round(statistics.mean(errs[sc]), 4) if errs[sc] else None,
            "degraded": {k[1]: int(v) for k, v in degraded.items() if k[0] == sc},
            "slo_p95_server_ms": SLO_P95_MS,
            "meets_slo": feed_p95 is not None and feed_p95 <= SLO_P95_MS,
        }
    startup = [int(x) for x in (RES / "startup-ms.txt").read_text().split()]
    out = {
        "machine": machine(),
        "image_bytes": int((RES / "image-bytes.txt").read_text()),
        "startup_to_ready_ms": {"runs": startup, "median": statistics.median(startup)},
        "memory_after_run": (RES / "memory-after.txt").read_text().strip(),
        "k6": "grafana/k6:2.3.0, constant arrival rate, 60 s at each rate",
        "mix": "60% first pages, 25% cursor pages, 15% event batches",
        "dropped_iterations": int(dropped),
        "scenarios": scenarios,
    }
    (RES / "03-local.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
