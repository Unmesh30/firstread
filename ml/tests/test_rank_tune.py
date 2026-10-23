"""The ranking study's proxies and pre-registered rules (ADR 0014), on tiny data."""

from dataclasses import replace

import numpy as np
import pytest

from firstread_ml.eval import rank_tune as rt
from firstread_ml.testing.builders import make_impressions

H = 3600


def timed(imps, times):  # type: ignore[no-untyped-def]
    return replace(imps, time=np.asarray(times, np.int64))


def test_trend_counts_only_earlier_clicks_inside_the_window_with_decay() -> None:
    # Article 7 is clicked at t=0 and t=10h; article 8 at t=0.
    clicks = timed(make_impressions([[], []], [[7, 8], [7, 9]], [[1, 1], [1, 0]]), [0, 10 * H])
    # Queries: article 7 at 10h (the 10h click is NOT strictly before), at 16h, at 30h;
    # article 8 at 10h; article 9 (never clicked).
    test = timed(
        make_impressions([[], [], [], []], [[7, 8], [7], [7, 9], [8]], [[1, 0], [1], [1, 0], [1]]),
        [10 * H, 16 * H, 30 * H, 25 * H],
    )
    got = rt.trend_scores(test, [clicks], window_h=24, half_life_h=6)
    want_7_at_10 = 2 ** (-10 / 6)  # only the t=0 click
    want_7_at_16 = 2 ** (-16 / 6) + 2 ** (-6 / 6)
    want_7_at_30 = 2 ** (-20 / 6)  # the t=0 click is outside 24 h
    assert got == pytest.approx(
        [want_7_at_10, 2 ** (-10 / 6), want_7_at_16, want_7_at_30, 0.0, 0.0]
    )


def row(norm: str, wf: float, wt: float, overall: float, cold_c: float, cold_u: float,
        h: float = 3.0) -> dict:  # type: ignore[type-arg]  # fmt: skip
    return {
        "norm": norm, "half_life": h, "w_fresh": wf, "w_trend": wt, "overall": overall,
        "cold_candidates": cold_c, "cold_users": cold_u, "J": (overall + cold_c) / 2,
    }  # fmt: skip


def test_blend_rule_guard_tie_and_simplicity() -> None:
    rows = [
        row("raw", 0, 0, 0.40, 0.50, 0.36),  # model only
        row("raw", 0.4, 0.4, 0.43, 0.56, 0.30),  # best J but hurts cold users > 1%
        row("zscore", 0.1, 0.0, 0.42, 0.54, 0.36),  # J 0.48
        row("raw", 0.05, 0.05, 0.4200, 0.5390, 0.36),  # tied, same weight, lower J
        row("raw", 0.1, 0.0, 0.4210, 0.5385, 0.36, h=6),  # tied, same weight, longer h
        row("raw", 0.1, 0.0, 0.4210, 0.5385, 0.36, h=3),  # tied, same weight, higher J: wins
    ]
    choice = rt.choose_blend(rows)
    assert choice["norm"] == "raw"
    assert choice["w_fresh"] == 0.1
    assert choice["half_life"] == 3


def test_lambda_rule_picks_the_most_diverse_within_one_percent() -> None:
    sweep = [{"lambda": lam, "overall": o} for lam, o in
             [(1.0, 0.400), (0.9, 0.399), (0.8, 0.3961), (0.7, 0.3959), (0.5, 0.37)]]  # fmt: skip
    assert rt.choose_lambda(sweep)["lambda"] == 0.8


def test_k_rule_and_soundness_fallback() -> None:
    sweep = [
        {"k": 0.0, "stated_overall": 0.40, "stated_cold_users": 0.30},
        {"k": 3.0, "stated_overall": 0.399, "stated_cold_users": 0.34},
        {"k": 10.0, "stated_overall": 0.39, "stated_cold_users": 0.36},  # overall drops > 0.5%
    ]
    assert rt.choose_k(sweep, n_cold=800) == (3.0, "tuned")
    assert rt.choose_k(sweep, n_cold=100) == (rt.K_JUDGMENT, "judgment")


def test_config_from_choices_validates_and_carries_the_segment() -> None:
    choice = {"norm": "zscore", "half_life": 6.0, "w_fresh": 0.05, "w_trend": 0.1}
    segment = {"adopt": True, "best": {"w_fresh": 0.2, "w_trend": 0.0}}
    cfg = rt.config_from(choice, 0.8, 3.0, "tuned", segment)
    assert cfg["version"] == "rank-1"
    assert cfg["cold_segment"] == {"max_history": 2, "w_fresh": 0.2, "w_trend": 0.0}
    assert "cold_segment" not in rt.config_from(choice, 0.8, 3.0, "tuned", {"adopt": False})


def test_stated_topics_need_three_fit_clicks_and_rank_by_count() -> None:
    fit = make_impressions([[], [], []], [[0, 1], [2, 3], [4, 5]], [[1, 1], [1, 1], [1, 0]])
    test = make_impressions([[], []], [[0], [1]], [[1], [1]])
    topics = np.array(["sports", "sports", "health", "exclude", "world", "us"])
    fit_users = np.array([7, 7, 8])
    test_users = np.array([7, 8])
    got = rt.stated_topics(fit, test, fit_users, test_users, topics)
    assert got == {0: ["sports", "health"]}  # user 8 has 1 click: no proxy
