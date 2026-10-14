"""PyTorch/sklearn vs ONNX Runtime parity checks (SPEC section 7.7: max abs diff <= 1e-4).

Shared by the CI tests (tiny random models) and ``release build`` (real weights).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_DIM, l2_normalize
from firstread_core.features import HISTORY_LEN
from firstread_ml.models.nrms_lite import NRMSLite

TOLERANCE = 1e-4
CASE_LENGTHS = (0, 1, 7, HISTORY_LEN - 1, HISTORY_LEN)


def session(path: Path, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def histories(
    lengths: list[int], rng: np.random.Generator
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Random unit-norm, right-aligned histories with the given numbers of clicks."""
    hist = np.zeros((len(lengths), HISTORY_LEN, EMBED_DIM), dtype=np.float32)
    mask = np.zeros((len(lengths), HISTORY_LEN), dtype=np.bool_)
    for i, n in enumerate(lengths):
        if n:
            hist[i, -n:] = l2_normalize(rng.standard_normal((n, EMBED_DIM)).astype(np.float32))
            mask[i, -n:] = True
    return hist, mask


def default_cases(seed: int = 0) -> dict[str, tuple[NDArray[np.float32], NDArray[np.bool_]]]:
    """Empty, partial and full histories; alone, mixed, batch 1 and batch 32."""
    rng = np.random.default_rng(seed)
    cases = {f"len{n}": histories([n], rng) for n in CASE_LENGTHS}
    cases["mixed"] = histories(list(CASE_LENGTHS), rng)
    cases["batch32"] = histories(list(rng.integers(0, HISTORY_LEN + 1, 32)), rng)
    return cases


@dataclass(frozen=True)
class ParityResult:
    max_abs_diff: dict[str, float]  # per case
    batch_invariance: float  # a row alone vs inside a batch of 32 (ONNX vs ONNX)

    @property
    def worst(self) -> float:
        return max([*self.max_abs_diff.values(), self.batch_invariance])

    @property
    def ok(self) -> bool:
        return self.worst <= TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "tolerance": TOLERANCE,
            "max_abs_diff": self.max_abs_diff,
            "batch_invariance": self.batch_invariance,
            "worst": self.worst,
            "ok": self.ok,
        }


@torch.no_grad()
def check_user_encoder(
    model: NRMSLite,
    onnx_path: Path,
    cases: dict[str, tuple[NDArray[np.float32], NDArray[np.bool_]]] | None = None,
) -> ParityResult:
    model = model.cpu().eval()
    sess = session(onnx_path)
    diffs: dict[str, float] = {}
    for name, (hist, mask) in (cases or default_cases()).items():
        ref = model(torch.from_numpy(hist), torch.from_numpy(mask)).numpy()
        got = sess.run(None, {"hist": hist, "mask": mask})[0]
        diffs[name] = float(np.abs(ref - got).max())
    hist, mask = (cases or default_cases())["batch32"]
    batched = sess.run(None, {"hist": hist, "mask": mask})[0]
    alone = np.concatenate(
        [sess.run(None, {"hist": hist[i : i + 1], "mask": mask[i : i + 1]})[0] for i in range(4)]
    )
    return ParityResult(diffs, float(np.abs(batched[:4] - alone).max()))


def check_classifier(clf: Any, onnx_path: Path, x: NDArray[np.float32]) -> dict[str, Any]:
    label, proba = session(onnx_path).run(None, {"embedding": x.astype(np.float32)})
    diff = float(np.abs(clf.predict_proba(x) - proba).max())
    mismatches = int((clf.predict(x) != label).sum())
    return {
        "tolerance": TOLERANCE,
        "max_abs_diff": diff,
        "label_mismatches": mismatches,
        "n": len(x),
        "ok": diff <= TOLERANCE and mismatches == 0,
    }
