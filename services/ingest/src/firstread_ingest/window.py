"""A runner-side copy of the last 72 h of articles: the egress fix (ADR 0012).

Dedupe, the pools, the topic centroids and (later) the personas all need the recent
window: about 14,500 rows with their vectors. Reading it from Postgres on every run
would cost ~14 MB of egress per run (~40 GB a month against Supabase Free's 5 GB).
Instead each run keeps the window in a :class:`WindowSnapshot`, saved to a file that
GitHub Actions caches between runs, and fetches only what changed.

**The snapshot is only a cache.** :func:`sync` trusts it only when all of these hold:

* the file exists and has this module's :data:`FORMAT_VERSION`, the applied schema
  (SHA of ``schema_migrations``) and the embedding model id;
* after fetching rows created since its watermark, one check query over the window
  agrees on ``count(*)``, ``max(created_at)`` and an MD5 of the sorted ids.

Anything else means a full reload from Postgres (the old ``load_recent`` cost), with
the reason recorded. Rows are immutable after insert (ADR 0009: clusters are never
re-merged) and there is one writer (the ``ingest`` concurrency group), so a watermark
delta is exact; the checksum catches anything else, such as a stray local run.

Vectors are kept as the exact float16 values Postgres stores (``halfvec``), so dedupe
sees bit-identical inputs either way.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_core.embedding import EMBED_DIM
from firstread_core.topics import TOPIC_IDS
from firstread_ingest.dedupe import Recent

FORMAT_VERSION = 1
WINDOW = timedelta(hours=72)  # dedupe fingerprints and topic pools look back 72 h

TEXT_SOURCES = ("title", "title_summary")
CATEGORY_SOURCES = ("model", "feed_hint", "geo_rule")
ORIGINS = ("gdelt", "rss")

COLUMNS = (
    "id, story_cluster_id, text_source, embedding, url_canonical, title_fingerprint, "
    "first_seen_at, created_at, category, category_source, category_confidence, origin, "
    "source_domain, (image_url is not null) as has_image"
)


def _us(t: datetime) -> int:
    return int(t.timestamp() * 1_000_000)


def _from_us(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, UTC)


@dataclass
class WindowSnapshot:
    """Parallel columns, one row per article, sorted by (first_seen_at, id)."""

    ids: list[uuid.UUID]
    clusters: list[uuid.UUID]
    text_source: NDArray[np.int8]
    embedding: NDArray[np.float16]  # [n, 384], the stored halfvec values
    url_canonical: list[str]
    fingerprint: list[str]
    first_seen_us: NDArray[np.int64]
    created_us: NDArray[np.int64]
    category: NDArray[np.int8]  # index into TOPIC_IDS
    category_source: NDArray[np.int8]
    confidence: NDArray[np.float32]
    origin: NDArray[np.int8]
    source_domain: list[str]
    has_image: NDArray[np.bool_]
    schema_sha: str
    embedding_model: str

    @classmethod
    def empty(cls, schema_sha: str, embedding_model: str) -> WindowSnapshot:
        return cls(
            [], [], np.zeros(0, np.int8), np.zeros((0, EMBED_DIM), np.float16), [], [],
            np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int8),
            np.zeros(0, np.int8), np.zeros(0, np.float32), np.zeros(0, np.int8), [],
            np.zeros(0, np.bool_), schema_sha, embedding_model,
        )  # fmt: skip

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def watermark_us(self) -> int:
        return int(self.created_us.max()) if len(self) else 0

    # ---- building ----
    @classmethod
    def from_rows(
        cls, rows: Sequence[Any], schema_sha: str, embedding_model: str
    ) -> WindowSnapshot:
        snap = cls.empty(schema_sha, embedding_model)
        return snap.merged(rows)

    def merged(self, rows: Sequence[Any]) -> WindowSnapshot:
        """This snapshot plus ``rows`` (asyncpg records with :data:`COLUMNS`), re-sorted.

        A row whose id is already present is ignored (rows never change after insert).
        """
        have = set(self.ids)
        new = [r for r in rows if r["id"] not in have]
        if not new:
            return self
        add = WindowSnapshot(
            ids=[r["id"] for r in new],
            clusters=[r["story_cluster_id"] for r in new],
            text_source=np.array([TEXT_SOURCES.index(r["text_source"]) for r in new], np.int8),
            embedding=np.stack([np.asarray(r["embedding"].to_numpy(), np.float16) for r in new]),
            url_canonical=[r["url_canonical"] for r in new],
            fingerprint=[r["title_fingerprint"] for r in new],
            first_seen_us=np.array([_us(r["first_seen_at"]) for r in new], np.int64),
            created_us=np.array([_us(r["created_at"]) for r in new], np.int64),
            category=np.array([TOPIC_IDS.index(r["category"]) for r in new], np.int8),
            category_source=np.array(
                [CATEGORY_SOURCES.index(r["category_source"]) for r in new], np.int8
            ),
            confidence=np.array([r["category_confidence"] for r in new], np.float32),
            origin=np.array([ORIGINS.index(r["origin"]) for r in new], np.int8),
            source_domain=[r["source_domain"] for r in new],
            has_image=np.array([bool(r["has_image"]) for r in new], np.bool_),
            schema_sha=self.schema_sha,
            embedding_model=self.embedding_model,
        )
        return self._concat(add)._sorted()

    def _concat(self, other: WindowSnapshot) -> WindowSnapshot:
        return WindowSnapshot(
            ids=self.ids + other.ids,
            clusters=self.clusters + other.clusters,
            text_source=np.concatenate([self.text_source, other.text_source]),
            embedding=np.concatenate([self.embedding, other.embedding]),
            url_canonical=self.url_canonical + other.url_canonical,
            fingerprint=self.fingerprint + other.fingerprint,
            first_seen_us=np.concatenate([self.first_seen_us, other.first_seen_us]),
            created_us=np.concatenate([self.created_us, other.created_us]),
            category=np.concatenate([self.category, other.category]),
            category_source=np.concatenate([self.category_source, other.category_source]),
            confidence=np.concatenate([self.confidence, other.confidence]),
            origin=np.concatenate([self.origin, other.origin]),
            source_domain=self.source_domain + other.source_domain,
            has_image=np.concatenate([self.has_image, other.has_image]),
            schema_sha=self.schema_sha,
            embedding_model=self.embedding_model,
        )

    def take(self, idx: NDArray[np.int64] | Sequence[int]) -> WindowSnapshot:
        ix = np.asarray(idx, np.int64)
        pick = ix.tolist()
        return WindowSnapshot(
            ids=[self.ids[i] for i in pick],
            clusters=[self.clusters[i] for i in pick],
            text_source=self.text_source[ix],
            embedding=self.embedding[ix],
            url_canonical=[self.url_canonical[i] for i in pick],
            fingerprint=[self.fingerprint[i] for i in pick],
            first_seen_us=self.first_seen_us[ix],
            created_us=self.created_us[ix],
            category=self.category[ix],
            category_source=self.category_source[ix],
            confidence=self.confidence[ix],
            origin=self.origin[ix],
            source_domain=[self.source_domain[i] for i in pick],
            has_image=self.has_image[ix],
            schema_sha=self.schema_sha,
            embedding_model=self.embedding_model,
        )

    def _sorted(self) -> WindowSnapshot:
        order = sorted(range(len(self)), key=lambda i: (self.first_seen_us[i], self.ids[i].bytes))
        return self.take(order)

    def since(self, cut: datetime) -> WindowSnapshot:
        """Rows with ``first_seen_at >= cut`` (the window boundary)."""
        return self.take(np.flatnonzero(self.first_seen_us >= _us(cut)))

    # ---- views used by the pipeline ----
    def recent(self, now: datetime, window: timedelta, fingerprint_window: timedelta) -> Recent:
        """Dedupe's view, with the same boundaries as the old ``store.load_recent``."""
        all_rows = self.since(now - max(window, fingerprint_window))
        emb = all_rows.since(now - window)
        fp_cut = _us(now - fingerprint_window)
        return Recent(
            ids=list(emb.ids),
            clusters=list(emb.clusters),
            text_sources=[TEXT_SOURCES[i] for i in emb.text_source],  # type: ignore[misc]
            embeddings=emb.embedding.astype(np.float32),
            url_cluster=dict(zip(all_rows.url_canonical, all_rows.clusters, strict=True)),
            fingerprint_cluster={
                fp: c
                for fp, c, ts in zip(
                    all_rows.fingerprint, all_rows.clusters, all_rows.first_seen_us, strict=True
                )
                if ts >= fp_cut
            },
        )

    def checksum(self) -> tuple[int, int, str]:
        """(count, max created_at in µs, md5 of sorted ids) — the check query's shape."""
        # uuid order is byte order in Postgres too, so no text collation is involved.
        ids = ",".join(str(i) for i in sorted(self.ids, key=lambda u: u.bytes))
        return len(self), self.watermark_us, hashlib.md5(ids.encode()).hexdigest()  # noqa: S324

    def categories(self) -> list[str]:
        return [TOPIC_IDS[i] for i in self.category]

    def text_sources(self) -> list[str]:
        return [TEXT_SOURCES[i] for i in self.text_source]

    # ---- persistence ----
    def save(self, path: Path) -> None:
        """Write atomically (tmp + rename) as an .npz with a JSON header."""
        header = {
            "format_version": FORMAT_VERSION,
            "schema_sha": self.schema_sha,
            "embedding_model": self.embedding_model,
            "rows": len(self),
            "watermark": _from_us(self.watermark_us).isoformat() if len(self) else None,
        }
        buf = io.BytesIO()
        # Uncompressed: fp16 vectors barely compress, and the Actions cache compresses.
        np.savez(
            buf,
            header=np.frombuffer(json.dumps(header).encode(), np.uint8),
            ids=np.frombuffer(b"".join(i.bytes for i in self.ids), np.uint8),
            clusters=np.frombuffer(b"".join(c.bytes for c in self.clusters), np.uint8),
            text_source=self.text_source,
            embedding=self.embedding,
            url_canonical=_pack(self.url_canonical),
            fingerprint=_pack(self.fingerprint),
            first_seen_us=self.first_seen_us,
            created_us=self.created_us,
            category=self.category,
            category_source=self.category_source,
            confidence=self.confidence,
            origin=self.origin,
            source_domain=_pack(self.source_domain),
            has_image=self.has_image,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple[WindowSnapshot, dict[str, Any]]:
        """Read a saved snapshot; raises :class:`SnapshotUnusableError` on any problem."""
        try:
            with np.load(path, allow_pickle=False) as z:
                header = json.loads(z["header"].tobytes())
                if header.get("format_version") != FORMAT_VERSION:
                    raise SnapshotUnusableError("format")
                ids = _uuids(z["ids"])
                snap = cls(
                    ids=ids,
                    clusters=_uuids(z["clusters"]),
                    text_source=z["text_source"],
                    embedding=z["embedding"],
                    url_canonical=_unpack(z["url_canonical"]),
                    fingerprint=_unpack(z["fingerprint"]),
                    first_seen_us=z["first_seen_us"],
                    created_us=z["created_us"],
                    category=z["category"],
                    category_source=z["category_source"],
                    confidence=z["confidence"],
                    origin=z["origin"],
                    source_domain=_unpack(z["source_domain"]),
                    has_image=z["has_image"],
                    schema_sha=str(header["schema_sha"]),
                    embedding_model=str(header["embedding_model"]),
                )
        except SnapshotUnusableError:
            raise
        except FileNotFoundError as exc:
            raise SnapshotUnusableError("missing") from exc
        except Exception as exc:  # corrupt or truncated file
            raise SnapshotUnusableError("corrupt") from exc
        n = len(snap.ids)
        cols = (snap.clusters, snap.text_source, snap.embedding, snap.url_canonical,
                snap.fingerprint, snap.first_seen_us, snap.created_us, snap.category,
                snap.category_source, snap.confidence, snap.origin, snap.source_domain,
                snap.has_image)  # fmt: skip
        if any(len(c) != n for c in cols) or snap.embedding.shape[1:] != (EMBED_DIM,):
            raise SnapshotUnusableError("corrupt")
        return snap, header


class SnapshotUnusableError(Exception):
    """Why a saved snapshot can't be trusted: a full reload follows."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _pack(strings: Sequence[str]) -> NDArray[np.uint8]:
    """Strings as one uint8 array: 4-byte little-endian lengths, then utf-8 bytes."""
    parts = [s.encode() for s in strings]
    lens = np.array([len(p) for p in parts], "<u4").tobytes()
    head = np.array([len(parts)], "<u4").tobytes()
    return np.frombuffer(head + lens + b"".join(parts), np.uint8)


def _unpack(arr: NDArray[np.uint8]) -> list[str]:
    raw = arr.tobytes()
    n = int(np.frombuffer(raw[:4], "<u4")[0])
    lens = np.frombuffer(raw[4 : 4 + 4 * n], "<u4")
    out, pos = [], 4 + 4 * n
    for ln in lens.tolist():
        out.append(raw[pos : pos + ln].decode())
        pos += ln
    return out


def _uuids(arr: NDArray[np.uint8]) -> list[uuid.UUID]:
    raw = arr.tobytes()
    return [uuid.UUID(bytes=raw[i : i + 16]) for i in range(0, len(raw), 16)]


# ---- Postgres side ----

_CHECK = (
    "select count(*) as n, max(created_at) as max_created, "
    "coalesce(md5(string_agg(id::text, ',' order by id)), md5('')) as ids_md5 "
    "from public.articles where first_seen_at >= $1"
)


async def schema_sha(conn: Any) -> str:
    """SHA of the applied migrations: a schema change invalidates every snapshot."""
    rows = await conn.fetch("select filename, sha256 from public.schema_migrations order by 1")
    joined = ";".join(f"{r['filename']}:{r['sha256']}" for r in rows)
    return hashlib.sha256(joined.encode()).hexdigest()


async def fetch_all(conn: Any, cut: datetime) -> list[Any]:
    rows: list[Any] = await conn.fetch(
        f"select {COLUMNS} from public.articles where first_seen_at >= $1",  # noqa: S608
        cut,
    )
    return rows


async def fetch_delta(conn: Any, cut: datetime, watermark_us: int) -> list[Any]:
    rows: list[Any] = await conn.fetch(
        f"select {COLUMNS} from public.articles "  # noqa: S608 (constant column list)
        "where first_seen_at >= $1 and created_at > $2",
        cut,
        _from_us(watermark_us),
    )
    return rows


async def check(conn: Any, cut: datetime) -> tuple[int, int, str]:
    row = await conn.fetchrow(_CHECK, cut)
    max_created = _us(row["max_created"]) if row["max_created"] is not None else 0
    return int(row["n"]), max_created, str(row["ids_md5"])


@dataclass
class SyncResult:
    snapshot: WindowSnapshot
    reloaded: bool
    reason: str  # "delta" when the cache was trusted, else why it was reloaded
    fetched_rows: int


async def sync(
    conn: Any,
    snap: WindowSnapshot | None,
    now: datetime,
    embedding_model: str,
    *,
    reason_if_none: str = "missing",
) -> SyncResult:
    """Bring ``snap`` up to date with Postgres, or reload it. See the module docstring."""
    cut = now - WINDOW
    sha = await schema_sha(conn)
    reason = None
    if snap is None:
        reason = reason_if_none
    elif snap.schema_sha != sha:
        reason = "schema"
    elif snap.embedding_model != embedding_model:
        reason = "embedding_model"
    if reason is None and snap is not None:
        delta = await fetch_delta(conn, cut, snap.watermark_us)
        updated = snap.merged(delta).since(cut)
        if updated.checksum() == await check(conn, cut):
            return SyncResult(updated, False, "delta", len(delta))
        reason = "checksum"
    rows = await fetch_all(conn, cut)
    full = WindowSnapshot.from_rows(rows, sha, embedding_model)
    return SyncResult(full, True, reason or "unknown", len(rows))


def load_file(path: Path | None) -> tuple[WindowSnapshot | None, str]:
    """The saved snapshot (or None) and, if None, why."""
    if path is None:
        return None, "no_file"
    try:
        snap, _ = WindowSnapshot.load(path)
    except SnapshotUnusableError as exc:
        return None, exc.reason
    return snap, "loaded"
