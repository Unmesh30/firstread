"""The micro-batcher (ADR 0017) and the real pinned encoder."""

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from firstread_api.encoder import EncoderClosedError, MicroBatcher, UserEncoder


class Fake:
    """Row i of the output is the sum of the user's history, so results are checkable."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.sizes: list[int] = []
        self.delay, self.fail = delay, fail
        self.threads: set[str] = set()

    def run(self, hist: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.threads.add(threading.current_thread().name)
        self.sizes.append(len(hist))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("onnx exploded")
        return hist.sum(axis=1).astype(np.float32)


def user(i: int) -> tuple[np.ndarray, np.ndarray]:
    h = np.zeros((50, 384), np.float32)
    h[-1, 0] = i
    m = np.zeros(50, np.bool_)
    m[-1] = True
    return h, m


def batcher(fake: Fake, **kw: Any) -> MicroBatcher:
    return MicroBatcher(fake, **kw)  # type: ignore[arg-type]


async def test_concurrent_requests_share_one_call_and_get_their_own_row() -> None:
    fake = Fake()
    b = batcher(fake, max_batch=32, max_wait_ms=20)
    outs = await asyncio.gather(*(b.encode(*user(i)) for i in range(10)))
    assert fake.sizes == [10]
    assert [float(o[0]) for o in outs] == list(range(10))
    assert all(name.startswith("onnx") for name in fake.threads)  # the dedicated thread
    await b.close()


async def test_a_full_batch_flushes_without_waiting_and_the_rest_follow() -> None:
    fake = Fake()
    b = batcher(fake, max_batch=4, max_wait_ms=1000)
    t0 = time.perf_counter()
    outs = await asyncio.wait_for(asyncio.gather(*(b.encode(*user(i)) for i in range(8))), 0.5)
    assert time.perf_counter() - t0 < 0.5  # never waited the 1 s
    assert fake.sizes == [4, 4]
    assert [float(o[0]) for o in outs] == list(range(8))
    await b.close()


async def test_a_lone_request_flushes_on_the_timeout() -> None:
    fake = Fake()
    b = batcher(fake, max_batch=32, max_wait_ms=5)
    out = await b.encode(*user(3))
    assert float(out[0]) == 3
    assert fake.sizes == [1]
    await b.close()


async def test_a_cancelled_waiter_does_not_cancel_the_batch() -> None:
    fake = Fake()
    b = batcher(fake, max_batch=32, max_wait_ms=30)
    leader = asyncio.ensure_future(b.encode(*user(1)))  # it started the batch
    others = [asyncio.ensure_future(b.encode(*user(i))) for i in (2, 3)]
    await asyncio.sleep(0.005)
    leader.cancel()
    results = await asyncio.gather(*others)
    assert [float(r[0]) for r in results] == [2, 3]
    assert fake.sizes == [2]  # the cancelled row isn't computed
    assert leader.cancelled()
    await b.close()


async def test_an_error_reaches_every_waiter_of_the_batch() -> None:
    b = batcher(Fake(fail=True), max_batch=32, max_wait_ms=5)
    results = await asyncio.gather(*(b.encode(*user(i)) for i in range(3)), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) and "exploded" in str(r) for r in results)
    await b.close()


async def test_close_drains_in_flight_work_and_rejects_new_work() -> None:
    fake = Fake(delay=0.05)
    b = batcher(fake, max_batch=32, max_wait_ms=5)
    pending = asyncio.ensure_future(b.encode(*user(5)))
    await asyncio.sleep(0.01)  # the batch is running on the worker thread
    await b.close()
    assert float((await pending)[0]) == 5  # drained, not dropped
    with pytest.raises(EncoderClosedError):
        await b.encode(*user(6))


async def test_batching_off_runs_each_request_alone() -> None:
    fake = Fake()
    b = batcher(fake, enabled=False)
    await asyncio.gather(*(b.encode(*user(i)) for i in range(4)))
    assert fake.sizes == [1, 1, 1, 1]
    await b.close()


def release_folder() -> Path | None:
    for p in (os.environ.get("API_MODEL_DIR"), "data/release/v0.2.0",
              str(Path.home() / ".cache/firstread/models/firstread-models/v0.2.0")):  # fmt: skip
        if p and (Path(p) / "user_encoder.onnx").is_file():
            return Path(p)
    return None


@pytest.mark.model
async def test_the_real_encoder_is_batch_invariant() -> None:
    folder = release_folder()
    if folder is None:
        from firstread_core.release import ENCODER_FILES, download_release

        folder = download_release("v0.2.0", ENCODER_FILES)
    enc = UserEncoder.load(folder, "v0.2.0")
    assert enc.tag == "v0.2.0"
    rng = np.random.default_rng(0)
    users = []
    for n in range(32):  # 0..31 clicks
        h = np.zeros((50, 384), np.float32)
        m = np.zeros(50, np.bool_)
        if n:
            v = rng.normal(size=(n, 384)).astype(np.float32)
            h[-n:] = v / np.linalg.norm(v, axis=1, keepdims=True)
            m[-n:] = True
        users.append((h, m))
    together = enc.run(np.stack([h for h, _ in users]), np.stack([m for _, m in users]))
    for i, (h, m) in enumerate(users):
        alone = enc.run(h[None], m[None])[0]
        assert np.abs(alone - together[i]).max() <= 1e-6
    b = MicroBatcher(enc, max_batch=32, max_wait_ms=20)
    via = await asyncio.gather(*(b.encode(h, m) for h, m in users))
    assert np.abs(np.stack(via) - together).max() <= 1e-6
    await b.close()
