"""Baselines every model must beat (SPEC section 7.5).

Each scorer is fitted on the training period only and scores every candidate
of every test impression independently.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import l2_normalize
from firstread_core.features import HISTORY_LEN, gather_history, history_index_matrix
from firstread_core.features import mean_history_vector as _mean_history
from firstread_ml.mind.data import Impressions
from firstread_ml.mind.splits import click_counts

SCORE_CHUNK = 8192  # impressions per batch when scoring with embeddings


@dataclass
class FitContext:
    """Everything a scorer may learn from: training-period data only, plus
    first-seen times (leak-free; see ``splits.first_seen``)."""

    train: Impressions
    n_news: int
    first_seen: NDArray[np.int64]
    emb: NDArray[np.float32] | None = None
    seed: int = 0
    embedding_id: str = ""  # embedding cache id, part of model checkpoint keys
    model_dir: Path | None = None  # where trainable scorers cache checkpoints
    tag: str = ""  # e.g. "small-holdout", for run names
    _population: NDArray[np.float32] | None = field(default=None, repr=False)

    @property
    def population_vector(self) -> NDArray[np.float32]:
        """Normalized mean embedding of all training clicks: the vector for a
        user with no history."""
        if self._population is None:
            if self.emb is None:
                raise ValueError("population vector needs embeddings")
            clicked = self.train.cand[self.train.label == 1]
            self._population = l2_normalize(self.emb[clicked].mean(axis=0))
        return self._population


class Scorer(Protocol):
    name: str
    needs_embeddings: bool

    def fit(self, ctx: FitContext) -> None: ...

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        """One score per flat candidate (``imps.cand``); higher ranks first."""
        ...


class RandomScorer:
    name = "random"
    needs_embeddings = False

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, ctx: FitContext) -> None:
        self.seed = ctx.seed

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return np.random.default_rng(self.seed + 1).random(len(imps.cand))


class PopularityScorer:
    """Clicks in the training period. Articles new at test time all score 0."""

    name = "popularity"
    needs_embeddings = False

    def fit(self, ctx: FitContext) -> None:
        self.counts = click_counts(ctx.train, ctx.n_news)

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return self.counts[imps.cand].astype(np.float64)


class RecencyScorer:
    """Newest first, by first-seen time (MIND has no publish time)."""

    name = "recency"
    needs_embeddings = False

    def fit(self, ctx: FitContext) -> None:
        self.first_seen = ctx.first_seen

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return self.first_seen[imps.cand].astype(np.float64)


class MeanHistoryScorer:
    """Cosine between a candidate and the mean of the user's last 50 clicks.

    Zero training: the content-only baseline. Empty histories use the
    population vector.
    """

    name = "mean_history"
    needs_embeddings = True

    def fit(self, ctx: FitContext) -> None:
        if ctx.emb is None:
            raise ValueError("mean_history needs embeddings")
        self.emb = ctx.emb
        self.population = ctx.population_vector

    def user_vectors(self, imps: Impressions, rows: slice) -> NDArray[np.float32]:
        hists = imps.histories(rows)
        idx, mask = history_index_matrix(hists, HISTORY_LEN)
        return _mean_history(gather_history(self.emb, idx, mask), mask, self.population)

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return score_by_user_vectors(imps, self.emb, self.user_vectors)


def score_by_user_vectors(
    imps: Impressions,
    emb: NDArray[np.float32],
    user_vectors: "UserVectorFn",
) -> NDArray[np.float64]:
    """Dot product of each impression's user vector with its candidates, in chunks."""
    out = np.empty(len(imps.cand), dtype=np.float64)
    for start in range(0, imps.n, SCORE_CHUNK):
        rows = slice(start, min(start + SCORE_CHUNK, imps.n))
        users = user_vectors(imps, rows)
        lo, hi = imps.cand_offsets[rows.start], imps.cand_offsets[rows.stop]
        local_imp = np.repeat(np.arange(users.shape[0]), imps.cand_len[rows])
        out[lo:hi] = np.einsum("ij,ij->i", users[local_imp], emb[imps.cand[lo:hi]])
    return out


class UserVectorFn(Protocol):
    def __call__(self, imps: Impressions, rows: slice) -> NDArray[np.float32]: ...


BASELINES: dict[str, type] = {
    "random": RandomScorer,
    "popularity": PopularityScorer,
    "recency": RecencyScorer,
    "mean_history": MeanHistoryScorer,
}
