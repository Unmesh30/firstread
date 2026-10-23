"""The streaming dev report gives the same numbers whatever the chunk size (ADR 0014)."""

import numpy as np

from firstread_core.ranking import RankingConfig
from firstread_ml.eval import rank_tune as rt
from firstread_ml.mind.data import Impressions

D = 384


def synthetic(n_imp: int = 40, n_news: int = 60, seed: int = 0) -> tuple[Impressions, rt.DevInputs]:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_news, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    hist_len = rng.integers(0, 8, n_imp)  # some cold users (<= 2), some empty
    cand_len = rng.integers(3, 12, n_imp)
    hist = rng.integers(0, n_news, hist_len.sum()).astype(np.int32)
    cand = np.concatenate([rng.choice(n_news, k, replace=False) for k in cand_len]).astype(np.int32)
    label = np.zeros(len(cand), np.int8)
    off = np.concatenate([[0], np.cumsum(cand_len)])
    for i in range(n_imp):  # one click and at least one non-click per impression
        label[off[i] + rng.integers(0, cand_len[i])] = 1
    test = Impressions(
        imp_id=np.arange(n_imp, dtype=np.int64),
        time=np.sort(rng.integers(100_000, 400_000, n_imp)).astype(np.int64),
        hist_offsets=np.concatenate([[0], np.cumsum(hist_len)]).astype(np.int64),
        hist=hist,
        cand_offsets=off.astype(np.int64),
        cand=cand,
        label=label,
    )
    clicks = rng.integers(0, n_news, 300)
    trend = rt.TrendIndex.build(
        clicks.astype(np.int64), rng.integers(0, 400_000, 300).astype(np.int64),
        int(test.time.min()), int(test.time.max()),
    )  # fmt: skip
    topics = ["sports", "health", "world"]
    centroids = {t: emb[k] for k, t in enumerate(topics)}

    def encode(h: np.ndarray, m: np.ndarray) -> np.ndarray:
        s: np.ndarray = ((h * m[..., None]).sum(axis=1) + 1e-3).astype(np.float32)
        return s

    inp = rt.DevInputs(
        emb=emb,
        first_seen=rng.integers(0, 100_000, n_news).astype(np.int64),
        cold_article=rng.random(n_news) < 0.5,
        category=rng.integers(0, 3, n_news).astype(np.int32),
        category_names=["a", "b", "c"],
        trend=trend,
        population=emb[5],
        centroids=centroids,
        stated={i: topics[: 1 + i % 3] for i in range(0, n_imp, 3)},
        encode=encode,
    )
    return test, inp


def config() -> RankingConfig:
    return RankingConfig.from_dict(
        dict(version="rank-x", model_norm="raw", w_fresh=0.05, w_trend=0.025,
             half_life_hours=3.0, trend_half_life_hours=6.0, mmr_lambda=0.7, story_cap=2,
             cold_start_k=0.0, cold_start_k_source="tuned", rerank_size=150, slate_size=100,
             page_size=20, cold_segment={"max_history": 2, "w_fresh": 0.05, "w_trend": 0.1})
    )  # fmt: skip


def test_chunked_equals_unchunked() -> None:
    test, inp = synthetic()
    sample = np.arange(0, test.n, 2)
    whole = rt.stream_dev(test, inp, config(), sample, chunk=10_000, progress=lambda _: None)
    for chunk in (1, 3, 7):
        parts = rt.stream_dev(test, inp, config(), sample, chunk=chunk, progress=lambda _: None)
        assert parts == whole, chunk


def test_stream_covers_every_row_and_slice() -> None:
    test, inp = synthetic()
    rows = rt.stream_dev(test, inp, config(), np.arange(test.n), chunk=5, progress=lambda _: None)
    assert {"model_only", "blend", "sample:blend", "sample:blend+mmr(λ=0.7)",
            "sample:blend+mmr(λ=1.0)", "stated:k=0.0", "stated:no_topics"} <= set(rows)  # fmt: skip
    assert rows["blend"]["overall"]["impressions"] == test.n
    assert {"category:a", "category:b", "category:c"} <= set(rows["blend"])
    assert "ild@10" in rows["sample:blend+mmr(λ=0.7)"]
    assert rt.peak_rss_gb() > 0
