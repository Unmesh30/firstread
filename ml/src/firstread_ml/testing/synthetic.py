"""Seeded generator of tiny MIND-format TSVs with planted structure.

Each article belongs to one topic; its title is ``"topicK <word> <word>"``.
Each user prefers one or two topics and mostly clicks articles from them, and
some articles are inherently more popular. Dev contains articles that never
appear in train (cold-start candidates) and users with empty histories (cold
users). ``hash_backend`` embeds titles so same-topic articles are similar,
without downloading a model.

Nothing here is MIND data: it only mimics the file format.
"""

import hashlib
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_DIM
from firstread_ml.mind.manifest import split_name

VARIANT = "synthetic"
N_TOPICS = 6
TRAIN_DAYS = 5  # the last of these is the holdout day
START = datetime(2019, 11, 9, 0, 0, 0)


def _fmt_time(t: datetime) -> str:
    return f"{t.month}/{t.day}/{t.year} {t.strftime('%I:%M:%S %p').lstrip('0')}"


def generate(
    raw_dir: Path,
    seed: int = 0,
    n_news: int = 600,
    n_users: int = 400,
    n_train_imps: int = 2500,
    n_dev_imps: int = 800,
) -> Path:
    """Write ``MINDsynthetic_{train,dev}/{news,behaviors}.tsv`` under ``raw_dir``."""
    rng = np.random.default_rng(seed)
    topic = rng.integers(0, N_TOPICS, n_news)
    popularity = rng.lognormal(0.0, 0.7, n_news)
    # Articles "appear" over time; the last 20% appear only on the dev day.
    appear_day = np.sort(rng.integers(0, TRAIN_DAYS, n_news))
    appear_day[int(n_news * 0.8) :] = TRAIN_DAYS
    rng.shuffle(appear_day)

    prefs = [rng.choice(N_TOPICS, size=rng.integers(1, 3), replace=False) for _ in range(n_users)]
    affinity = np.full((n_users, N_TOPICS), 0.15)
    for u, p in enumerate(prefs):
        affinity[u, p] = 1.0
    # A third of users start with an empty history; others have 1-30 old clicks.
    hist_len = np.where(rng.random(n_users) < 0.33, 0, rng.integers(1, 31, n_users))
    old_pool = np.flatnonzero(appear_day == 0)
    histories = []
    for u in range(n_users):
        w = affinity[u, topic[old_pool]] * popularity[old_pool]
        k = min(int(hist_len[u]), len(old_pool))
        histories.append(rng.choice(old_pool, size=k, replace=False, p=w / w.sum()))

    def impressions(n: int, days: list[int]) -> list[str]:
        rows = []
        for i in range(n):
            day = int(rng.choice(days))
            u = int(rng.integers(n_users))
            pool = np.flatnonzero(appear_day <= day)
            fresh = pool[appear_day[pool] >= day - 1]
            size = int(rng.integers(5, 21))
            cands = rng.choice(fresh if len(fresh) >= size else pool, size=size, replace=False)
            p_click = 0.35 * affinity[u, topic[cands]] * np.minimum(popularity[cands], 3.0) / 3.0
            labels = (rng.random(size) < p_click).astype(int)
            labels[int(np.argmax(p_click))] = 1  # at least one click
            if labels.all():
                labels[int(np.argmin(p_click))] = 0  # at least one non-click
            t = START + timedelta(days=day, seconds=int(rng.integers(0, 86_400)))
            hist = " ".join(f"N{j}" for j in histories[u])
            imp = " ".join(f"N{c}-{lab}" for c, lab in zip(cands, labels, strict=True))
            rows.append(f"{i + 1}\tU{u}\t{_fmt_time(t)}\t{hist}\t{imp}")
        return rows

    def news_rows(ids: Iterable[int]) -> list[str]:
        words = [f"w{j}" for j in range(50)]
        return [
            f"N{i}\ttopic{topic[i]}\tsub{topic[i]}\ttopic{topic[i]} "
            f"{words[(i * 7) % 50]} {words[(i * 13) % 50]}\tSynthetic abstract {i}.\t"
            f"https://example.com/{i}\t[]\t[]"
            for i in ids
        ]

    train_ids = np.flatnonzero(appear_day < TRAIN_DAYS)
    for split, imps, ids in (
        ("train", impressions(n_train_imps, list(range(TRAIN_DAYS))), train_ids),
        ("dev", impressions(n_dev_imps, [TRAIN_DAYS]), range(n_news)),
    ):
        d = raw_dir / split_name(VARIANT, split)
        d.mkdir(parents=True, exist_ok=True)
        (d / "news.tsv").write_text("\n".join(news_rows(ids)) + "\n")
        (d / "behaviors.tsv").write_text("\n".join(imps) + "\n")
    return raw_dir


def _word_vector(word: str) -> NDArray[np.float64]:
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=EMBED_DIM)


def hash_backend(texts: list[str]) -> Iterable[NDArray[Any]]:
    """Deterministic fake embedder: weighted sum of per-word random vectors.

    The first word (the topic) dominates, so same-topic texts are similar.
    """
    for text in texts:
        words = text.replace(".", " ").split()
        vec = np.zeros(EMBED_DIM)
        for j, w in enumerate(words):
            vec += (3.0 if j == 0 else 0.5) * _word_vector(w)
        yield vec
