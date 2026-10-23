"""The pool snapshot blob shared by ingestion and the API (ADR 0012)."""

import uuid
import zlib
from datetime import UTC, datetime

import numpy as np
import pytest

from firstread_core import pools
from firstread_core.personas import PersonaSample, PersonaSummary, parse_all
from firstread_core.pools import ArticleInfo, PoolSnapshot, SnapshotError, TrendingEntry
from firstread_core.topics import TOPIC_IDS

NOW = datetime(2026, 10, 15, 12, 20, tzinfo=UTC)


def full_snapshot(version: str = "v1") -> PoolSnapshot:
    """Every pool at its cap, all topics, all centroids: the largest blob in practice."""
    rng = np.random.default_rng(0)

    def ids(n: int) -> list[uuid.UUID]:
        return [uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | k, version=4) for k in range(n)]

    fresh = ids(pools.FRESH_CAP)
    topics = {t: ids(pools.TOPIC_CAP) for t in TOPIC_IDS}
    trending = [
        TrendingEntry(a, 10.0 - k / 100, 5 + k % 50, (uuid.uuid4(), uuid.uuid4()))
        for k, a in enumerate(ids(pools.TRENDING_CAP))
    ]
    members = set(fresh) | {i for v in topics.values() for i in v} | {t.id for t in trending}
    articles = {
        a: ArticleInfo(uuid.uuid4(), "sports", 1_760_000_000 + k, "title")
        for k, a in enumerate(members)
    }
    centroids = {t: rng.normal(size=384).astype(np.float32) for t in TOPIC_IDS}
    stats = {"articles_24h": 4855, "sources_24h": 1200, "median_lag_s": 912.0}
    # Five demo personas (ADR 0018) with long sample headlines.
    personas = [
        PersonaSummary(
            f"persona-{k}", "A persona name", "A one-line description of who this reader is.",
            ("business", "tech_science", "politics"), 20, 1_760_000_000,
            tuple(PersonaSample(f"{k}{j} " + "w" * 197, "A Source Name", "business")
                  for j in range(3)),
        ).to_dict()
        for k in range(5)
    ]  # fmt: skip
    return PoolSnapshot(version, NOW, fresh, topics, trending, articles, centroids, stats, personas)


def test_round_trip_keeps_everything() -> None:
    snap = full_snapshot()
    back = pools.decode(pools.encode(snap), "v1")
    assert back.version == "v1"
    assert back.built_at == NOW
    assert back.fresh == snap.fresh
    assert back.topics == snap.topics
    assert back.trending[0].clusters == snap.trending[0].clusters
    assert back.trending[0].score == pytest.approx(snap.trending[0].score, abs=1e-4)
    assert back.articles == snap.articles
    assert np.allclose(back.centroids["sports"], snap.centroids["sports"], atol=1e-2)  # fp16
    assert back.stats == snap.stats
    assert parse_all(back.personas) == parse_all(snap.personas)
    assert len(parse_all(back.personas)) == 5


def test_blob_at_the_caps_stays_under_the_limit() -> None:
    size = len(pools.encode(full_snapshot()))
    assert size <= pools.MAX_BLOB_BYTES, size


def test_version_inside_the_blob_must_match_pools_ver() -> None:
    blob = pools.encode(full_snapshot("v1"))
    with pytest.raises(SnapshotError, match="version"):
        pools.decode(blob, "v2")


@pytest.mark.parametrize(
    "blob",
    [b"garbage", zlib.compress(b"[1, 2]"), zlib.compress(b'{"schema_version": 99}')],
)
def test_unreadable_or_unknown_blobs_are_rejected(blob: bytes) -> None:
    with pytest.raises(SnapshotError):
        pools.decode(blob)


def test_every_pool_member_needs_article_info() -> None:
    snap = full_snapshot()
    snap.articles.pop(snap.fresh[0])
    with pytest.raises(SnapshotError, match="article info"):
        pools.encode(snap)


def test_topic_centroids_prefer_confident_rows_and_skip_empty_topics() -> None:
    e = np.eye(384, dtype=np.float32)[:4]
    topics = ["sports", "sports", "sports", "health"]
    good = np.array([True, False, False, True])
    out = pools.topic_centroids(e, topics, good, min_rows=1)
    assert np.allclose(out["sports"], e[0])  # only the confident row
    assert set(out) == {"sports", "health"}  # no rows -> absent (population fallback)
    mixed = pools.topic_centroids(e, topics, good, min_rows=2)
    assert np.isclose(np.linalg.norm(mixed["sports"]), 1.0)  # too few confident: all rows
    assert mixed["sports"][1] > 0


def test_malformed_persona_summaries_are_skipped() -> None:
    good = full_snapshot().personas[0]
    bad_topics = {**good, "topics": ["nope"]}
    missing = {k: v for k, v in good.items() if k != "samples"}
    assert [p.id for p in parse_all([good, bad_topics, missing, {"id": 3}])] == [good["id"]]
