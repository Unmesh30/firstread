"""Evaluation slices. Each keeps a subset of every impression's candidates.

* ``overall``: all candidates.
* ``cold_candidates``: candidates never clicked in training (not a positive in
  any train impression, not in any train history).
* ``cold_users``: impressions whose user has <= 2 history clicks (0 included).
* ``category:<c>``: candidates in MIND category c.

After restricting, an impression stays in the slice only if it still has at
least one clicked and one non-clicked candidate.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from firstread_ml.mind.data import Impressions

COLD_USER_MAX_HISTORY = 2
LOW_N = 100  # slices with fewer impressions are flagged as low-n


@dataclass(frozen=True)
class Slice:
    name: str
    idx: NDArray[np.int64]  # indices into the flat candidate arrays
    offsets: NDArray[np.int64]
    extra: dict[str, int]

    @property
    def n_impressions(self) -> int:
        return len(self.offsets) - 1


def restrict(
    imps: Impressions, keep: NDArray[np.bool_], name: str, extra: dict[str, int] | None = None
) -> Slice:
    cand_imp = imps.cand_imp
    kept = np.flatnonzero(keep)
    k_imp = cand_imp[kept]
    pos = np.bincount(k_imp, weights=imps.label[kept], minlength=imps.n)
    tot = np.bincount(k_imp, minlength=imps.n)
    valid = (pos > 0) & (pos < tot)
    kept = kept[valid[k_imp]]
    sizes = np.bincount(cand_imp[kept], minlength=imps.n)[valid]
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return Slice(name, kept, offsets, extra or {})


def build_slices(
    test: Impressions,
    clicked_in_train: NDArray[np.bool_],
    article_category: NDArray[np.int32],
    category_names: list[str],
) -> list[Slice]:
    every = np.ones(len(test.cand), dtype=np.bool_)
    hist_len = test.hist_len
    cold_user = hist_len[test.cand_imp] <= COLD_USER_MAX_HISTORY
    empty_hist = hist_len[test.cand_imp] == 0
    cold_users = restrict(test, cold_user, "cold_users")
    n_empty = int(np.unique(test.cand_imp[cold_users.idx][empty_hist[cold_users.idx]]).size)
    slices = [
        restrict(test, every, "overall"),
        restrict(test, ~clicked_in_train[test.cand], "cold_candidates"),
        Slice(cold_users.name, cold_users.idx, cold_users.offsets, {"empty_history": n_empty}),
    ]
    cats = article_category[test.cand]
    for c, cname in enumerate(category_names):
        slices.append(restrict(test, cats == c, f"category:{cname}"))
    return slices
