"""Encoder latency inside the API image (ADR 0017): batch 1 vs 32, 1 vs 2 ONNX threads.

    docker run --rm --cpus=1 --memory=512m --entrypoint python firstread-api \\
        -m firstread_api.bench_encoder > docs/learn/data/03-encoder-bench.json

Every history is full (50 clicks: the worst case; padding costs the same). Prints
JSON: per configuration, the p50/p95 of one ``session.run`` and the cost per user.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from firstread_api.encoder import UserEncoder
from firstread_core.embedding import EMBED_DIM
from firstread_core.features import HISTORY_LEN


def histories(batch: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(batch, HISTORY_LEN, EMBED_DIM)).astype(np.float32)
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    return h, np.ones((batch, HISTORY_LEN), np.bool_)


def run(folder: Path, tag: str, threads: int, batch: int, n: int, warmup: int) -> dict[str, float]:
    enc = UserEncoder.load(folder, tag, threads)
    hist, mask = histories(batch)
    for _ in range(warmup):
        enc.run(hist, mask)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        enc.run(hist, mask)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50, p95 = statistics.median(times), times[int(0.95 * (len(times) - 1))]
    return {"p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "per_user_p50_ms": round(p50 / batch, 3)}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("API_MODEL_DIR"))
    parser.add_argument("--tag", default=os.environ.get("API_MODEL_TAG", "v0.2.0"))
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    if args.model_dir is None:
        raise SystemExit("set API_MODEL_DIR or --model-dir (the image sets it)")
    import onnxruntime

    results = {
        f"threads{t}_batch{b}": run(args.model_dir, args.tag, t, b, args.runs, args.warmup)
        for t in (1, 2)
        for b in (1, 32)
    }
    print(json.dumps({
        "where": "API runtime image (docker run --cpus=1 --memory=512m)",
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "cpus_visible": os.cpu_count(),
        "onnxruntime": onnxruntime.__version__,
        "model_tag": args.tag,
        "history_len": HISTORY_LEN,
        "runs": args.runs,
        "results": results,
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
