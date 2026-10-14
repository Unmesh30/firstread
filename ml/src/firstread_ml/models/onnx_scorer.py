"""Score impressions with an exported user encoder through ONNX Runtime.

Used by the real-data eval gate, so the gate measures the exact artifact we
release rather than the PyTorch weights it came from.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from firstread_core.features import HISTORY_LEN, gather_history, history_index_matrix
from firstread_ml.baselines import FitContext, score_by_user_vectors
from firstread_ml.export.parity import session
from firstread_ml.mind.data import Impressions

BATCH = 1024


class ORTUserEncoderScorer:
    name = "onnx_release"
    needs_embeddings = True

    def __init__(self, onnx_path: Path, threads: int = 4) -> None:
        self.sess = session(onnx_path, threads)

    def fit(self, ctx: FitContext) -> None:
        """Nothing to learn: the weights are fixed. Keeps the embedding table."""
        if ctx.emb is None:
            raise ValueError("onnx_release needs embeddings")
        self.emb = ctx.emb

    def user_vectors(self, imps: Impressions, rows: slice) -> NDArray[np.float32]:
        idx, mask = history_index_matrix(imps.histories(rows), HISTORY_LEN)
        hist = gather_history(self.emb, idx, mask)
        out = [
            self.sess.run(None, {"hist": hist[i : i + BATCH], "mask": mask[i : i + BATCH]})[0]
            for i in range(0, len(mask), BATCH)
        ]
        return np.concatenate(out) if out else np.zeros((0, self.emb.shape[1]), np.float32)

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return score_by_user_vectors(imps, self.emb, self.user_vectors)
