"""Versioned, append-only history in Redis (ADR 0016)."""

import uuid

import numpy as np

from firstread_api import redis_state as rs


def entry(version: int, article: uuid.UUID | None = None) -> rs.HistoryEntry:
    v = np.random.default_rng(version).normal(size=384).astype(np.float16)
    return rs.HistoryEntry(version, article or uuid.uuid4(), uuid.uuid4(), "sports", "title", v)


def test_pack_round_trip_and_markers() -> None:
    e = entry(7)
    back = rs.unpack(rs.pack(e))
    assert (back.version, back.article_id, back.cluster, back.topic) == (
        7, e.article_id, e.cluster, "sports",
    )  # fmt: skip
    assert back.vector is not None
    assert e.vector is not None
    assert np.array_equal(back.vector, e.vector)
    m = rs.unpack(rs.pack(rs.marker(12)))
    assert m.marker
    assert m.version == 12


def test_consistency_rules() -> None:
    full = [entry(v) for v in range(1, 6)]
    assert rs.is_consistent(full, 5)
    assert not rs.is_consistent(full, 6)  # a write never reached Redis
    assert not rs.is_consistent(full[:2] + full[3:], 5)  # a gap
    assert rs.is_consistent([], 0)
    assert not rs.is_consistent([], 3)  # lost key
    rebuilt = [entry(2), entry(9), rs.marker(10), entry(11), entry(12)]
    assert rs.is_consistent(rebuilt, 12)
    assert not rs.is_consistent(rebuilt, 13)
    assert not rs.is_consistent([entry(2), rs.marker(10), entry(12)], 12)  # 11 missing
    window = [entry(v) for v in range(41, 41 + rs.HIST_KEEP_RAW)]  # marker trimmed away
    assert rs.is_consistent(window, 40 + rs.HIST_KEEP_RAW)
    assert not rs.is_consistent(window[1:], 40 + rs.HIST_KEEP_RAW)  # short and not from 1


def test_latest_deduplicates_by_article_and_caps() -> None:
    a = uuid.uuid4()
    entries = [entry(1, a), entry(2), rs.marker(2), entry(3, a)]
    out = rs.latest(entries)
    assert [e.version for e in out] == [2, 3]
    many = [entry(v) for v in range(1, 80)]
    assert len(rs.latest(many)) == 50
    assert rs.latest(many)[-1].version == 79


def test_hidden_sentinel_tells_missing_from_empty() -> None:
    assert rs.hidden_from([]) is None
    assert rs.hidden_from([rs.SENTINEL]) == set()
    a = uuid.uuid4()
    assert rs.hidden_from([rs.SENTINEL, str(a).encode()]) == {a}
