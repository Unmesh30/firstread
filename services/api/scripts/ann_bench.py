"""ANN vs exact search on a 14-day-sized article table (ADR 0017). Local Docker only.

The 3-day replay fits inside the feed's 72 h window, so it can't show the problem
HNSW has with a time filter: at the default ef_search the filter only sees the ~40
rows the index returns. This builds a table the size of 14 days of retention:

* the 14,400 replayed articles, plus 4 copies shifted 3, 6, 9 and 12 days older with
  small re-normalized noise, so only ~20% of rows fall in the 72 h window;
* 200 query vectors: normalized means of 10 random in-window articles (synthetic
  readers).

It compares exact search (``order by d + 0``, no index) with HNSW iterative scans at
ef_search 200 and 400, on recall@200 against exact and on latency.

**Rule (pre-registered in the plan):** use HNSW if recall@200 ≥ 0.95 **and** its p95
≤ exact's p95; otherwise exact. Writes ``docs/learn/data/03-ann.json``.

    uv run python services/api/scripts/ann_bench.py
"""

from __future__ import annotations

import asyncio
import json
import platform
import statistics
import time
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
from pgvector import HalfVector

from firstread_api.candidates import ANN_EXACT, ANN_HNSW
from firstread_core.db import apply_migrations, connect, register_vector_codec
from firstread_core.dbtest import MIGRATIONS, admin_dsn, temp_database, with_database

OUT = Path(__file__).resolve().parents[3] / "docs" / "learn" / "data" / "03-ann.json"
COPIES = (3, 6, 9, 12)  # days older
QUERIES, LIMIT, RUNS = 200, 200, 3
SIGMA = 0.02
RECALL_MIN = 0.95

COLS = (
    "id, url, url_canonical, title, source_name, source_domain, geo, category, "
    "category_confidence, category_source, first_seen_at, embedding, embedding_model, "
    "text_source, title_fingerprint, story_cluster_id, origin"
)


async def build(dst: object, rows: list, rng: np.random.Generator) -> None:  # type: ignore[type-arg]
    records = []
    for shift in (0, *COPIES):
        for r in rows:
            v = np.asarray(r["embedding"].to_numpy(), np.float32)
            if shift:
                v = v + rng.normal(scale=SIGMA, size=v.shape).astype(np.float32)
                v /= np.linalg.norm(v)
            aid = uuid.uuid4() if shift else r["id"]
            records.append((
                aid, r["url"] + (f"#c{shift}" if shift else ""),
                r["url_canonical"] + (f"#c{shift}" if shift else ""), r["title"],
                r["source_name"], r["source_domain"], r["geo"], r["category"],
                r["category_confidence"], r["category_source"],
                r["first_seen_at"] - timedelta(days=shift), HalfVector(v), r["embedding_model"],
                r["text_source"], f"{r['title_fingerprint']}-{shift}", r["story_cluster_id"],
                r["origin"],
            ))  # fmt: skip
    await dst.copy_records_to_table(  # type: ignore[attr-defined]
        "articles", schema_name="public", records=records,
        columns=[c.strip() for c in COLS.split(",")],
    )  # fmt: skip
    await dst.execute("vacuum analyze public.articles")  # type: ignore[attr-defined]


async def timed(
    conn: object, mode: str, ef: int, q: np.ndarray, since: object
) -> tuple[list, float]:  # type: ignore[type-arg]
    args = (HalfVector(q), since, [], [], LIMIT, [], [])
    t0 = time.perf_counter()
    async with conn.transaction():  # type: ignore[attr-defined]
        if mode == "hnsw":
            await conn.execute(f"set local hnsw.ef_search = {ef}")  # type: ignore[attr-defined]
            await conn.execute("set local hnsw.iterative_scan = relaxed_order")  # type: ignore[attr-defined]
            rows = await conn.fetch(ANN_HNSW, *args)  # type: ignore[attr-defined]
        else:
            rows = await conn.fetch(ANN_EXACT, *args)  # type: ignore[attr-defined]
    return rows, (time.perf_counter() - t0) * 1000


async def main() -> None:
    rng = np.random.default_rng(0)
    src = await connect(with_database(admin_dsn(), "firstread_replay"))
    await register_vector_codec(src)
    rows = await src.fetch(f"select {COLS} from public.articles")  # noqa: S608 (constant)
    await src.close()
    now = max(r["first_seen_at"] for r in rows) + timedelta(minutes=1)
    since = now - timedelta(hours=72)
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            await register_vector_codec(conn)
            t0 = time.perf_counter()
            await build(conn, rows, rng)
            total = await conn.fetchval("select count(*) from public.articles")
            in_window = await conn.fetchval(
                "select count(*) from public.articles where first_seen_at > $1", since
            )
            build_s = time.perf_counter() - t0
            window = [r for r in rows if r["first_seen_at"] > since]
            queries = []
            for _ in range(QUERIES):
                pick = rng.choice(len(window), 10, replace=False)
                m = np.mean([window[i]["embedding"].to_numpy() for i in pick], axis=0)
                queries.append((m / np.linalg.norm(m)).astype(np.float32))
            plans = {
                "exact": ("exact", 0),
                "hnsw_ef200": ("hnsw", 200),
                "hnsw_ef400": ("hnsw", 400),
            }
            truth = {}
            results: dict[str, dict[str, float]] = {}
            for name, (mode, ef) in plans.items():
                lat, recalls, sizes = [], [], []
                for k, q in enumerate(queries):
                    for _ in range(RUNS):
                        got, ms = await timed(conn, mode, ef, q, since)
                        lat.append(ms)
                    ids = {r["id"] for r in got}
                    sizes.append(len(ids))
                    if name == "exact":
                        truth[k] = ids
                    recalls.append(len(ids & truth[k]) / len(truth[k]))
                lat.sort()
                results[name] = {
                    "p50_ms": round(statistics.median(lat), 2),
                    "p95_ms": round(lat[int(0.95 * len(lat)) - 1], 2),
                    "recall_at_200": round(float(np.mean(recalls)), 4),
                    "min_recall": round(float(np.min(recalls)), 4),
                    "mean_results": round(float(np.mean(sizes)), 1),
                }
                print(name, results[name], flush=True)
            plan_text = await conn.fetch(
                "explain " + ANN_HNSW.replace("$1::extensions.halfvec(384)", "$1"),
                HalfVector(queries[0]), since, [], [], LIMIT,
            )  # fmt: skip
        finally:
            await conn.close()
    best = min(
        (n for n in results if n != "exact" and results[n]["recall_at_200"] >= RECALL_MIN
         and results[n]["p95_ms"] <= results["exact"]["p95_ms"]),
        key=lambda n: results[n]["p95_ms"], default=None,
    )  # fmt: skip
    report = {
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "postgres": "local Docker (pgvector/pgvector:0.8.7-pg17, Docker Desktop 4 GB)",
        "rows": total,
        "rows_in_72h_window": in_window,
        "build_seconds": round(build_s, 1),
        "queries": QUERIES,
        "runs_per_query": RUNS,
        "limit": LIMIT,
        "results": results,
        "rule": "HNSW if recall@200 >= 0.95 and p95 <= exact p95; otherwise exact",
        "decision": "exact" if best is None else best,
        "hnsw_plan_uses_index": any("articles_embedding_hnsw" in r[0] for r in plan_text),
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
