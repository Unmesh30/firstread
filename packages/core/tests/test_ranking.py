"""Ranking functions shared by offline tuning and the API (ADR 0014)."""

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from firstread_core.ranking import (
    RankingConfig,
    ReasonInputs,
    blend,
    cold_start_vector,
    freshness,
    load_config,
    mmr,
    model_version,
    normalize,
    reasons,
    topic_weight,
    trend_feature,
)
from firstread_core.ranking import config as ranking_config

RNG = np.random.default_rng(0)


def unit_rows(n: int, d: int = 384) -> np.ndarray:
    v = RNG.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---- components ----


def test_freshness_halves_every_half_life_and_clamps_negative_ages() -> None:
    f = freshness(np.array([0.0, 6.0, 12.0, -3.0]), 6.0)
    assert f.tolist() == pytest.approx([1.0, 0.5, 0.25, 1.0])
    with pytest.raises(ValueError, match="half_life"):
        freshness(1.0, 0)


@pytest.mark.parametrize("method", ["raw", "zscore", "percentile"])
def test_grouped_equals_one_request_at_a_time(method: str) -> None:
    x = RNG.normal(size=17)
    x[3] = x[4]  # a tie
    offsets = np.array([0, 5, 6, 17])
    grouped = normalize(x, method, offsets)  # type: ignore[arg-type]
    single = np.concatenate(
        [normalize(x[a:b], method) for a, b in itertools.pairwise(offsets)]  # type: ignore[arg-type]
    )
    assert np.allclose(grouped, single)


def test_zscore_and_percentile_properties() -> None:
    x = np.array([1.0, 2.0, 2.0, 5.0])
    z = normalize(x, "zscore")
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)
    assert normalize(np.array([3.0, 3.0]), "zscore").tolist() == [0.0, 0.0]  # no spread
    p = normalize(x, "percentile")
    assert p.tolist() == pytest.approx([0.125, 0.5, 0.5, 0.875])  # ties share a rank
    with pytest.raises(ValueError, match="unknown normalization"):
        normalize(x, "softmax")  # type: ignore[arg-type]


def test_trend_feature_is_zero_when_not_trending_and_rank_only() -> None:
    raw = np.array([0.0, 10.0, 0.5, 0.0])
    t = trend_feature(raw)
    assert t[0] == 0.0
    assert t[3] == 0.0
    assert t[1] > t[2] > 0
    assert np.allclose(trend_feature(raw * 1000), t)  # only ranks matter


def test_blend_is_linear_in_its_weights() -> None:
    m, f, t = RNG.normal(size=5), RNG.random(5), RNG.random(5)
    base = blend(m, f, t, "raw", 0.0, 0.0)
    assert np.allclose(base, m)
    assert np.allclose(blend(m, f, t, "raw", 0.2, 0.1) - base, 0.2 * f + 0.1 * t)


# ---- MMR ----


def test_lambda_one_is_a_stable_relevance_sort() -> None:
    rel = np.array([0.3, 0.9, 0.9, 0.1, 0.5])
    res = mmr(rel, unit_rows(5), 1.0)
    assert res.order == [1, 2, 4, 0, 3]
    assert res.capped == []
    assert res.diversity_picks == set()


def test_output_is_a_duplicate_free_subset_and_respects_the_cap() -> None:
    rel = RNG.random(30)
    groups = [i % 4 for i in range(30)]  # 4 stories
    res = mmr(rel, unit_rows(30), 0.7, groups, cap=2)
    assert len(res.order) == len(set(res.order)) == 8  # 4 groups x 2
    assert sorted(res.order + res.capped) == list(range(30))
    for g in range(4):
        assert sum(1 for i in res.order if groups[i] == g) <= 2
    assert [rel[i] for i in res.capped] == sorted((rel[i] for i in res.capped), reverse=True)


def test_diversity_rises_as_lambda_falls() -> None:
    base = unit_rows(1)[0]
    near = base + 0.05 * unit_rows(10)  # ten near-copies of one story
    far = unit_rows(10)
    vecs = np.concatenate([near, far])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    rel = np.concatenate([np.linspace(1.0, 0.9, 10), np.linspace(0.8, 0.7, 10)])

    def ild(order: list[int]) -> float:
        top = vecs[order[:6]]
        s = top @ top.T
        n = len(top)
        return float((1 - s)[~np.eye(n, dtype=bool)].mean())

    values = [ild(mmr(rel, vecs, lam, k=6).order) for lam in (1.0, 0.8, 0.5, 0.2)]
    assert values == sorted(values)
    assert values[-1] > values[0]


def test_mmr_is_deterministic_and_handles_edge_cases() -> None:
    rel, v = RNG.random(12), unit_rows(12)
    assert mmr(rel, v, 0.6).order == mmr(rel, v, 0.6).order
    assert mmr(np.zeros(0), np.zeros((0, 384)), 0.5).order == []
    assert mmr(np.ones(3), unit_rows(3), 0.5).order[0] == 0  # flat relevance
    with pytest.raises(ValueError, match="lam"):
        mmr(rel, v, 1.5)


# ---- cold start ----


def test_topic_weight_and_cold_start_limits() -> None:
    assert topic_weight(0, 3) == 1.0
    assert topic_weight(3, 3) == 0.5
    assert topic_weight(5, 0) == 0.0
    cents = {"sports": unit_rows(1)[0], "health": unit_rows(1)[0]}
    pop, click = unit_rows(1)[0], unit_rows(1)[0]
    only_topics = cold_start_vector(cents, ["sports"], None, 0, 3, pop)
    assert np.allclose(only_topics, cents["sports"], atol=1e-6)
    mostly_clicks = cold_start_vector(cents, ["sports"], click, 10_000, 3, pop)
    assert float(mostly_clicks @ click) > 0.999
    fallback = cold_start_vector({}, ["weather"], None, 0, 3, pop)
    assert np.allclose(fallback, pop, atol=1e-6)  # no centroid -> population vector
    mixed = cold_start_vector(cents, ["sports", "health"], click, 2, 3, pop)
    assert np.linalg.norm(mixed) == pytest.approx(1.0, abs=1e-6)


# ---- reasons ----


def test_reasons_follow_the_largest_components() -> None:
    warm = ReasonInputs(model=0.6, fresh=0.01, trend=0.3, age_hours=5, outlets=12,
                        topic="sports", followed=("sports",), topic_weight=0.1,
                        nearest_read="Team wins final")  # fmt: skip
    got = reasons(warm)
    assert [r.code for r in got] == ["similar_to_read", "trending"]
    assert got[0].text == "Because you read “Team wins final”"
    assert got[1].text == "Covered by 12 outlets in the last 24 h"
    cold = ReasonInputs(model=0.5, fresh=0.2, trend=0.0, age_hours=0.3, outlets=0,
                        topic="health", followed=("health",), topic_weight=0.9)  # fmt: skip
    assert [r.code for r in reasons(cold)] == ["topic_match", "new"]
    plain = ReasonInputs(model=0.1, fresh=0.0, trend=0.0, age_hours=30, outlets=0,
                         topic="autos", followed=("autos",))  # fmt: skip
    assert [r.text for r in reasons(plain)] == ["From Autos, a topic you follow"]


# ---- config ----

# Each released ranking config's SHA-256 (ADR 0014): editing a file means a new version.
# rank-1: tuned on MIND-small holdout. rank-2: rank-1 with the owner's k = 3. rank-3 (the
# default): rank-2 with the owner's w_fresh 0.1 / w_trend 0.05, a tie on the holdout J.
PINNED_SHA: dict[str, str] = {
    "rank-1": "cd25c71fe10fd0a0d1787dadbda417fb13b6b8c92c86e3dbbbfc7373294e6f23",
    "rank-2": "3befa2936a3f1d5f205a4aee091a0159d173374b8863a0c750d992aab1427b51",
    "rank-3": "9a1fee30535564926b9ff1724d070cc8ed189e30b4a9d60f13caaf740c64a8a2",
}


def test_rank_2_differs_from_rank_1_only_in_k() -> None:
    one, two = load_config("rank-1"), load_config("rank-2")
    assert model_version("v0.2.0", two) == "v0.2.0+rank-2"
    assert (two.cold_start_k, two.cold_start_k_source) == (3.0, "judgment")
    assert (one.cold_start_k, one.cold_start_k_source) == (0.0, "tuned")
    same = {
        k: v
        for k, v in vars(one).items()
        if k not in {"version", "cold_start_k", "cold_start_k_source"}
    }
    assert same == {k: v for k, v in vars(two).items() if k in same}


def test_rank_3_is_the_default_and_differs_from_rank_2_only_in_the_shared_weights() -> None:
    two, three = load_config("rank-2"), load_config()
    assert three.version == "rank-3"
    assert model_version("v0.2.0", three) == "v0.2.0+rank-3"
    assert (three.w_fresh, three.w_trend) == (0.1, 0.05)
    assert three.cold_segment == two.cold_segment  # cold readers keep rank-2's weights
    same = {k: v for k, v in vars(two).items() if k not in {"version", "w_fresh", "w_trend"}}
    assert same == {k: v for k, v in vars(three).items() if k in same}


def test_every_ranking_config_is_pinned_and_valid() -> None:
    files = sorted(Path(ranking_config.HERE).glob("rank-*.yaml"))
    assert {f.stem for f in files} == set(PINNED_SHA)
    for f in files:
        assert hashlib.sha256(f.read_bytes()).hexdigest() == PINNED_SHA[f.stem], f.name
        cfg = load_config(f.stem)
        assert model_version("v0.2.0", cfg) == f"v0.2.0+{f.stem}"


def make(**over: object) -> RankingConfig:
    base = dict(version="rank-x", model_norm="raw", w_fresh=0.1, w_trend=0.1,
                half_life_hours=6.0, trend_half_life_hours=6.0, mmr_lambda=0.8, story_cap=2,
                cold_start_k=3.0, cold_start_k_source="judgment", rerank_size=150,
                slate_size=100, page_size=20)  # fmt: skip
    return RankingConfig.from_dict(base | over)


def test_config_validation_and_segment_weights() -> None:
    cfg = make(cold_segment={"max_history": 2, "w_fresh": 0.3, "w_trend": 0.0})
    assert cfg.weights_for(1) == (0.3, 0.0)
    assert cfg.weights_for(3) == (0.1, 0.1)
    assert make().weights_for(0) == (0.1, 0.1)
    for bad in ({"model_norm": "x"}, {"mmr_lambda": 2}, {"w_fresh": -1}, {"page_size": 500},
                {"cold_start_k_source": "guess"}):  # fmt: skip
        with pytest.raises(ValueError, match=r"must|unknown|need"):
            make(**bad)
