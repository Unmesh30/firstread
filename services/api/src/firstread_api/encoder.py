"""The pinned NRMS-lite user encoder and the micro-batcher in front of it (ADR 0017).

**Loading:** the release files are baked into the image at build time (Docker
``models`` stage) and re-verified here against the same pin and ``packages/core``'s
feature constants. No network and no ``huggingface_hub`` at runtime.

**Micro-batching:** concurrent feed requests that need a fresh user vector share one
ONNX call. The first request of a batch starts a short-lived *driver* task that waits
up to ``API_BATCH_WAIT_MS`` (or until ``API_BATCH_MAX`` requests joined), then runs one
``session.run`` on a dedicated worker thread and hands each waiter its row. There's
no background loop: the driver exists only while requests are waiting, which suits
Cloud Run's request-based billing.

* A waiter cancelled while waiting is skipped; the batch still runs for the others.
* An exception in the ONNX call reaches every waiter of that batch.
* ``close()`` rejects new work and waits for in-flight batches.
* ``API_ENCODER_BATCHING=false`` runs each request alone (P6 compares the two).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_api.metrics import ENCODER_BATCH, ENCODER_SECONDS
from firstread_core.embedding import EMBED_DIM
from firstread_core.features import HISTORY_LEN
from firstread_core.release import ENCODER_FILES, pin, verify_release


class EncoderClosedError(RuntimeError):
    """The encoder is shutting down."""


@dataclass
class UserEncoder:
    session: Any
    population: NDArray[np.float32]
    tag: str
    alpha: float

    @classmethod
    def load(cls, folder: Path, tag: str, threads: int = 1) -> UserEncoder:
        import onnxruntime as ort

        manifest = verify_release(folder, pin(tag)["manifest_sha256"], ENCODER_FILES)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(folder / "user_encoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        population = np.load(folder / "population.npy").astype(np.float32)
        if population.shape != (EMBED_DIM,):
            raise ValueError(f"population vector has shape {population.shape}")
        return cls(
            session, population, str(manifest["tag"]), float(manifest["user_encoder"]["alpha"])
        )

    def run(self, hist: NDArray[np.float32], mask: NDArray[np.bool_]) -> NDArray[np.float32]:
        """``hist[B, 50, 384]`` (right-aligned, zero padding), ``mask[B, 50]`` → ``[B, 384]``."""
        out = self.session.run(
            None, {"hist": hist.astype(np.float32), "mask": mask.astype(np.bool_)}
        )
        return np.asarray(out[0], np.float32)


@dataclass
class _Batch:
    hists: list[NDArray[np.float32]] = field(default_factory=list)
    masks: list[NDArray[np.bool_]] = field(default_factory=list)
    futures: list[asyncio.Future[NDArray[np.float32]]] = field(default_factory=list)
    full: asyncio.Event = field(default_factory=asyncio.Event)


class MicroBatcher:
    def __init__(
        self,
        encoder: UserEncoder,
        max_batch: int = 32,
        max_wait_ms: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.encoder = encoder
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.enabled = enabled
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx")
        self._open: _Batch | None = None
        self._drivers: set[asyncio.Task[None]] = set()
        self._closed = False

    async def encode(
        self, hist: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """One user's ``hist[50, 384]`` and ``mask[50]`` → their 384-d vector."""
        if self._closed:
            raise EncoderClosedError("encoder is shutting down")
        if hist.shape != (HISTORY_LEN, EMBED_DIM) or mask.shape != (HISTORY_LEN,):
            raise ValueError("expected hist[50, 384] and mask[50]")
        loop = asyncio.get_running_loop()
        if not self.enabled:
            out: NDArray[np.float32] = (await self._run_now(hist[None], mask[None]))[0]
            return out
        batch = self._open
        if batch is None:
            batch = _Batch()
            self._open = batch
            task = loop.create_task(self._drive(batch))
            self._drivers.add(task)
            task.add_done_callback(self._drivers.discard)
        fut: asyncio.Future[NDArray[np.float32]] = loop.create_future()
        batch.hists.append(hist)
        batch.masks.append(mask)
        batch.futures.append(fut)
        if len(batch.futures) >= self.max_batch:
            batch.full.set()
            if self._open is batch:
                self._open = None  # the next request starts a new batch
        return await fut

    async def _run_now(
        self, hist: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        out = await loop.run_in_executor(self.executor, self.encoder.run, hist, mask)
        ENCODER_SECONDS.observe(time.perf_counter() - t0)
        ENCODER_BATCH.observe(len(hist))
        return out

    async def _drive(self, batch: _Batch) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(batch.full.wait(), self.max_wait)
        if self._open is batch:
            self._open = None
        live = [i for i, f in enumerate(batch.futures) if not f.done()]
        if not live:
            return  # every waiter was cancelled
        hist = np.stack([batch.hists[i] for i in live])
        mask = np.stack([batch.masks[i] for i in live])
        try:
            out = await self._run_now(hist, mask)
        except Exception as exc:
            for i in live:
                if not batch.futures[i].done():
                    batch.futures[i].set_exception(exc)
            return
        for row, i in enumerate(live):
            if not batch.futures[i].done():
                batch.futures[i].set_result(out[row])

    async def close(self) -> None:
        """Reject new work, finish in-flight batches, stop the worker thread."""
        self._closed = True
        if self._open is not None:
            self._open.full.set()
        if self._drivers:
            await asyncio.gather(*self._drivers, return_exceptions=True)
        self.executor.shutdown(wait=True)
