"""Time-based splits and per-article statistics derived from impression logs."""

import numpy as np
from numpy.typing import NDArray

from firstread_ml.mind.data import Impressions

DAY = 86_400


def holdout_split(train: Impressions) -> tuple[Impressions, Impressions]:
    """Split train into (fit, holdout): holdout is the last calendar day (UTC).

    We tune on holdout and report on dev only once, so dev numbers stay honest.
    """
    last_day_start = (int(train.time.max()) // DAY) * DAY
    is_holdout = train.time >= last_day_start
    if is_holdout.all():
        raise ValueError("train spans a single day; cannot make a time-based holdout")
    return train.take(~is_holdout), train.take(is_holdout)


def clicked_articles(train: Impressions, n_news: int) -> NDArray[np.bool_]:
    """Articles clicked during training: a positive in any train impression or
    present in any train user's history. Everything else is "cold"."""
    out = np.zeros(n_news, dtype=np.bool_)
    out[train.cand[train.label == 1]] = True
    out[train.hist] = True
    return out


def click_counts(train: Impressions, n_news: int) -> NDArray[np.int64]:
    """Number of clicks per article in train impressions (the popularity baseline)."""
    return np.bincount(train.cand[train.label == 1], minlength=n_news).astype(np.int64)


def first_seen(parts: list[Impressions], n_news: int) -> NDArray[np.int64]:
    """Earliest impression time at which each article appears as a candidate.

    MIND has no publish time, so this stands in for it (recency baseline).
    No leakage: an article that is a candidate at time t has first-seen <= t.
    Articles never shown as candidates get the earliest time minus one day.
    """
    big = np.iinfo(np.int64).max
    out = np.full(n_news, big, dtype=np.int64)
    for p in parts:
        np.minimum.at(out, p.cand, np.repeat(p.time, p.cand_len))
    seen = out != big
    floor = int(out[seen].min()) - DAY if seen.any() else 0
    out[~seen] = floor
    return out
