"""The candidate-pool snapshot that ingestion publishes and the API reads (ADR 0012).

Ingestion builds one :class:`PoolSnapshot` per run and writes it to Redis as a single
compressed blob, then points ``pools:ver`` at it:

1. ``SET pools:blob:{ver} <blob>`` (no TTL: an ingestion gap of any length leaves the
   last good snapshot readable);
2. ``SET pools:ver {ver}``;
3. ``EXPIRE pools:blob:{previous} 120``: a grace period for a reader that fetched the
   old version just before the switch.

The version and :data:`SCHEMA_VERSION` are also inside the blob, so a reader can
check that the blob it fetched is the one ``pools:ver`` named, and that it knows the
format. The blob is zlib-compressed JSON (no pickle): ids are base64url of their 16
bytes, times are epoch seconds, centroids are float16 bytes.
"""

from __future__ import annotations

import base64
import json
import uuid
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_DIM
from firstread_core.topics import TOPIC_IDS

SCHEMA_VERSION = 1
VER_KEY = "pools:ver"
BLOB_PREFIX = "pools:blob:"
PREVIOUS_GRACE_S = 120
# Caps the API relies on (and the size test checks): 500 fresh, 300 per topic, 200 trending.
FRESH_CAP, TOPIC_CAP, TRENDING_CAP = 500, 300, 200
MAX_BLOB_BYTES = 256 * 1024


def blob_key(version: str) -> str:
    return f"{BLOB_PREFIX}{version}"


class SnapshotError(ValueError):
    """The blob is unreadable, of an unknown schema, or not the version asked for."""


def _id(u: uuid.UUID) -> str:
    return base64.urlsafe_b64encode(u.bytes).rstrip(b"=").decode()


def _uuid(s: str) -> uuid.UUID:
    return uuid.UUID(bytes=base64.urlsafe_b64decode(s + "=="))


@dataclass(frozen=True)
class ArticleInfo:
    """What the API needs about a pool member before it hydrates the full row."""

    cluster: uuid.UUID
    category: str
    first_seen: int  # epoch seconds
    text_source: str  # "title" | "title_summary"


@dataclass(frozen=True)
class TrendingEntry:
    id: uuid.UUID  # the story's representative article
    score: float
    outlets: int  # distinct outlets that covered the story in the trending window
    clusters: tuple[uuid.UUID, ...]  # story clusters merged into this entry


@dataclass
class PoolSnapshot:
    version: str
    built_at: datetime
    fresh: list[uuid.UUID]  # newest first
    topics: dict[str, list[uuid.UUID]]  # per app topic, newest first
    trending: list[TrendingEntry]  # best first
    articles: dict[uuid.UUID, ArticleInfo]  # every pool member
    centroids: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    personas: list[dict[str, Any]] = field(default_factory=list)

    def member_ids(self) -> set[uuid.UUID]:
        out = set(self.fresh) | {t.id for t in self.trending}
        for ids in self.topics.values():
            out.update(ids)
        return out


def encode(snap: PoolSnapshot) -> bytes:
    """Compressed blob; raises if a pool member has no :class:`ArticleInfo`."""
    missing = snap.member_ids() - set(snap.articles)
    if missing:
        raise SnapshotError(f"{len(missing)} pool members have no article info")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "version": snap.version,
        "built_at": snap.built_at.astimezone(UTC).isoformat(),
        "fresh": [_id(i) for i in snap.fresh],
        "topics": {t: [_id(i) for i in ids] for t, ids in snap.topics.items()},
        "trending": [
            [_id(t.id), round(t.score, 4), t.outlets, [_id(c) for c in t.clusters]]
            for t in snap.trending
        ],
        "articles": {
            _id(a): [_id(i.cluster), i.category, i.first_seen, i.text_source]
            for a, i in snap.articles.items()
        },
        "centroids": {
            t: base64.b64encode(np.asarray(v, np.float16).tobytes()).decode()
            for t, v in snap.centroids.items()
        },
        "stats": snap.stats,
        "personas": snap.personas,
    }
    raw = json.dumps(doc, separators=(",", ":"), sort_keys=True, default=str).encode()
    return zlib.compress(raw, 6)


def decode(blob: bytes, expected_version: str | None = None) -> PoolSnapshot:
    """Parse a blob; raises :class:`SnapshotError` on anything unexpected."""
    try:
        doc = json.loads(zlib.decompress(blob))
    except (zlib.error, ValueError) as exc:
        raise SnapshotError(f"unreadable pool snapshot: {type(exc).__name__}") from exc
    if not isinstance(doc, dict):
        raise SnapshotError("unreadable pool snapshot: not an object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError(f"unknown pool snapshot schema {doc.get('schema_version')!r}")
    if expected_version is not None and doc.get("version") != expected_version:
        raise SnapshotError("pool snapshot version differs from pools:ver")
    try:
        topics = {t: [_uuid(i) for i in ids] for t, ids in doc["topics"].items()}
        unknown = set(topics) - set(TOPIC_IDS)
        if unknown:
            raise SnapshotError(f"unknown topics in snapshot: {sorted(unknown)}")
        centroids = {}
        for t, b64 in doc["centroids"].items():
            v = np.frombuffer(base64.b64decode(b64), np.float16).astype(np.float32)
            if v.shape != (EMBED_DIM,):
                raise SnapshotError(f"centroid {t} has shape {v.shape}")
            centroids[t] = v
        return PoolSnapshot(
            version=str(doc["version"]),
            built_at=datetime.fromisoformat(doc["built_at"]),
            fresh=[_uuid(i) for i in doc["fresh"]],
            topics=topics,
            trending=[
                TrendingEntry(_uuid(a), float(s), int(n), tuple(_uuid(c) for c in cs))
                for a, s, n, cs in doc["trending"]
            ],
            articles={
                _uuid(a): ArticleInfo(_uuid(c), str(cat), int(ts), str(src))
                for a, (c, cat, ts, src) in doc["articles"].items()
            },
            centroids=centroids,
            stats=dict(doc["stats"]),
            personas=list(doc["personas"]),
        )
    except SnapshotError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"malformed pool snapshot: {type(exc).__name__}") from exc


def topic_centroids(
    embeddings: NDArray[np.floating[Any]],
    topics: Sequence[str],
    weights: NDArray[np.bool_] | None = None,
    min_rows: int = 20,
) -> dict[str, NDArray[np.float32]]:
    """Normalized mean embedding per topic (the cold-start topic vectors, SPEC §7.4).

    ``weights`` marks the preferred rows (e.g. confident model labels). A topic uses
    only those when it has at least ``min_rows`` of them, otherwise all of its rows.
    Topics with no rows are absent: callers fall back to the population vector.
    Shared by the live snapshot (ingestion) and offline tuning on MIND (ml).
    """
    from firstread_core.embedding import l2_normalize

    emb = np.asarray(embeddings, np.float32)
    labels = np.asarray(topics)
    good = np.ones(len(labels), np.bool_) if weights is None else np.asarray(weights, np.bool_)
    out: dict[str, NDArray[np.float32]] = {}
    for t in TOPIC_IDS:
        rows = labels == t
        if not rows.any():
            continue
        pick = rows & good if (rows & good).sum() >= min_rows else rows
        out[t] = l2_normalize(emb[pick].mean(axis=0))
    return out


def info_map(
    ids: Sequence[uuid.UUID],
    clusters: Sequence[uuid.UUID],
    categories: Sequence[str],
    first_seen: Sequence[int],
    text_sources: Sequence[str],
    wanted: set[uuid.UUID],
) -> dict[uuid.UUID, ArticleInfo]:
    """:class:`ArticleInfo` for the ``wanted`` ids, from parallel columns."""
    out: dict[uuid.UUID, ArticleInfo] = {}
    for i, c, cat, ts, src in zip(ids, clusters, categories, first_seen, text_sources, strict=True):
        if i in wanted:
            out[i] = ArticleInfo(c, cat, int(ts), src)
    return out


def summarize(snap: PoolSnapshot) -> Mapping[str, int]:
    """Pool sizes for run summaries."""
    return {
        "fresh": len(snap.fresh),
        "trending": len(snap.trending),
        **{f"topic:{t}": len(v) for t, v in snap.topics.items()},
    }
