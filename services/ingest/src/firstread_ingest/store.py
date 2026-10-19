"""Postgres reads and writes for ingestion. Everything else in the pipeline is pure.

* Vectors are **read** with pgvector's asyncpg codec (binary, registered for the
  ``extensions`` schema) and **written** as text cast to ``extensions.halfvec(384)``
  inside one ``insert ... select from unnest(...)``, so a batch is one statement.
* ``on conflict (url_canonical) do nothing returning`` makes the upsert idempotent
  and tells us which rows were actually new.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from firstread_core.embedding import EMBED_DIM
from firstread_ingest.dedupe import Recent
from firstread_ingest.sources.rss import FeedState


async def register_codecs(conn: Any) -> None:
    from pgvector.asyncpg import register_vector

    await register_vector(conn, schema="extensions")


# ---- egress accounting (Supabase Free: 5 GB/month, pooler egress included; ADR 0011) ----

_ROW_OVERHEAD = 11  # DataRow message: type byte, length, field count
_FIELD_OVERHEAD = 4  # per-field length prefix


def value_bytes(v: Any) -> int:
    """Approximate wire size of one result value (binary protocol)."""
    if v is None:
        return 0
    if isinstance(v, bytes | bytearray | memoryview):
        return len(v)
    if isinstance(v, str):
        return len(v.encode())
    if isinstance(v, bool):
        return 1
    if isinstance(v, int | float | datetime):
        return 8
    if isinstance(v, uuid.UUID):
        return 16
    if isinstance(v, list | tuple):
        return 20 + sum(value_bytes(x) + _FIELD_OVERHEAD for x in v)
    dim = getattr(v, "dimensions", None)  # pgvector HalfVector / Vector
    if callable(dim):
        return 4 + 2 * int(dim())
    return 8


def rows_bytes(rows: Any) -> int:
    if rows is None:
        return 0
    if not isinstance(rows, list):
        rows = [rows]
    total = 0
    for r in rows:
        values = r.values() if hasattr(r, "values") else [r]
        total += _ROW_OVERHEAD + sum(value_bytes(v) + _FIELD_OVERHEAD for v in values)
    return total


class Metered:
    """A connection wrapper that estimates result bytes (Supabase bills egress).

    Everything else passes straight through to the asyncpg connection.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.bytes = 0
        self.queries = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def fetch(self, *args: Any, **kwargs: Any) -> Any:
        rows = await self._conn.fetch(*args, **kwargs)
        self.queries += 1
        self.bytes += rows_bytes(rows)
        return rows

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:
        row = await self._conn.fetchrow(*args, **kwargs)
        self.queries += 1
        self.bytes += rows_bytes(row)
        return row

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        v = await self._conn.fetchval(*args, **kwargs)
        self.queries += 1
        self.bytes += _ROW_OVERHEAD + value_bytes(v) + _FIELD_OVERHEAD
        return v

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        status = await self._conn.execute(*args, **kwargs)
        self.queries += 1
        self.bytes += len(str(status)) + 8  # CommandComplete
        return status


@dataclass(frozen=True)
class ArticleRow:
    id: uuid.UUID
    url: str
    url_canonical: str
    title: str
    source_name: str
    source_domain: str
    image_url: str | None
    language: str
    source_country: str | None
    geo: str
    category: str
    category_confidence: float
    category_source: str
    published_at: datetime | None
    first_seen_at: datetime
    embedding: np.ndarray
    embedding_model: str
    text_source: str
    title_fingerprint: str
    story_cluster_id: uuid.UUID
    origin: str


def _vec_text(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v.tolist()) + "]"


async def load_recent(
    conn: Any, now: datetime, window: timedelta, fingerprint_window: timedelta
) -> Recent:
    rows = await conn.fetch(
        "select id, story_cluster_id, text_source, embedding, url_canonical, title_fingerprint, "
        "first_seen_at from public.articles where first_seen_at >= $1",
        now - max(window, fingerprint_window),
    )
    emb_rows = [r for r in rows if r["first_seen_at"] >= now - window]
    embeddings = (
        np.stack([np.asarray(r["embedding"].to_numpy(), np.float32) for r in emb_rows])
        if emb_rows
        else np.zeros((0, EMBED_DIM), np.float32)
    )
    fp_cut = now - fingerprint_window
    return Recent(
        ids=[r["id"] for r in emb_rows],
        clusters=[r["story_cluster_id"] for r in emb_rows],
        text_sources=[r["text_source"] for r in emb_rows],
        embeddings=embeddings,
        url_cluster={r["url_canonical"]: r["story_cluster_id"] for r in rows},
        fingerprint_cluster={
            r["title_fingerprint"]: r["story_cluster_id"]
            for r in rows
            if r["first_seen_at"] >= fp_cut
        },
    )


_INSERT = """
insert into public.articles (
  id, url, url_canonical, title, source_name, source_domain, image_url, language,
  source_country, geo, category, category_confidence, category_source, published_at,
  first_seen_at, embedding, embedding_model, text_source, title_fingerprint,
  story_cluster_id, origin)
select id, url, url_canonical, title, source_name, source_domain, image_url, language,
  source_country, geo, category, category_confidence, category_source, published_at,
  first_seen_at, embedding::extensions.halfvec(384), embedding_model, text_source,
  title_fingerprint, story_cluster_id, origin
from unnest(
  $1::uuid[], $2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::text[],
  $8::text[], $9::text[], $10::text[], $11::text[], $12::real[], $13::text[],
  $14::timestamptz[], $15::timestamptz[], $16::text[], $17::text[], $18::text[],
  $19::text[], $20::uuid[], $21::text[]
) as t(id, url, url_canonical, title, source_name, source_domain, image_url, language,
  source_country, geo, category, category_confidence, category_source, published_at,
  first_seen_at, embedding, embedding_model, text_source, title_fingerprint,
  story_cluster_id, origin)
on conflict (url_canonical) do nothing
returning url_canonical
"""


async def insert_articles(conn: Any, rows: Sequence[ArticleRow]) -> set[str]:
    """Insert new articles; return the canonical URLs that were actually inserted."""
    if not rows:
        return set()
    cols = [
        [r.id for r in rows],
        [r.url for r in rows],
        [r.url_canonical for r in rows],
        [r.title for r in rows],
        [r.source_name for r in rows],
        [r.source_domain for r in rows],
        [r.image_url for r in rows],
        [r.language for r in rows],
        [r.source_country for r in rows],
        [r.geo for r in rows],
        [r.category for r in rows],
        [r.category_confidence for r in rows],
        [r.category_source for r in rows],
        [r.published_at for r in rows],
        [r.first_seen_at for r in rows],
        [_vec_text(r.embedding) for r in rows],
        [r.embedding_model for r in rows],
        [r.text_source for r in rows],
        [r.title_fingerprint for r in rows],
        [r.story_cluster_id for r in rows],
        [r.origin for r in rows],
    ]
    out = await conn.fetch(_INSERT, *cols)
    return {r["url_canonical"] for r in out}


async def insert_coverage(conn: Any, rows: Sequence[tuple[uuid.UUID, str, datetime]]) -> None:
    """(story_cluster_id, source_domain, first_seen_at); the first sighting wins."""
    if not rows:
        return
    await conn.execute(
        "insert into public.story_coverage (story_cluster_id, source_domain, first_seen_at) "
        "select * from unnest($1::uuid[], $2::text[], $3::timestamptz[]) "
        "on conflict (story_cluster_id, source_domain) do nothing",
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
    )


async def load_feed_states(conn: Any) -> dict[str, FeedState]:
    rows = await conn.fetch(
        "select feed_id, etag, last_modified, last_status, consecutive_failures "
        "from public.feed_state"
    )
    return {
        r["feed_id"]: FeedState(
            r["etag"], r["last_modified"], r["last_status"], r["consecutive_failures"]
        )
        for r in rows
    }


async def save_feed_states(conn: Any, states: dict[str, FeedState], now: datetime) -> None:
    if not states:
        return
    ids = list(states)
    await conn.execute(
        "insert into public.feed_state (feed_id, etag, last_modified, last_status, "
        "last_fetched_at, consecutive_failures) "
        "select f, e, m, s, $5, c from unnest($1::text[], $2::text[], $3::text[], $4::int[], "
        "$6::int[]) as t(f, e, m, s, c) "
        "on conflict (feed_id) do update set etag = excluded.etag, "
        "last_modified = excluded.last_modified, last_status = excluded.last_status, "
        "last_fetched_at = excluded.last_fetched_at, "
        "consecutive_failures = excluded.consecutive_failures",
        ids,
        [states[i].etag for i in ids],
        [states[i].last_modified for i in ids],
        [states[i].last_status for i in ids],
        now,
        [states[i].consecutive_failures for i in ids],
    )


async def start_run(conn: Any, now: datetime) -> uuid.UUID:
    run_id: uuid.UUID = await conn.fetchval(
        "insert into public.ingestion_runs (started_at) values ($1) returning id", now
    )
    return run_id


async def finish_run(conn: Any, run_id: uuid.UUID, summary: dict[str, Any], now: datetime) -> None:
    await conn.execute(
        "update public.ingestion_runs set finished_at = $2, window_start = $3, window_end = $4, "
        "fetched = $5, inserted = $6, duplicates = $7, near_duplicates = $8, clustered = $9, "
        "errors = $10, gdelt_calls = $11, status = $12, details = $13::jsonb, "
        "redis_commands = $14 where id = $1",
        run_id,
        now,
        summary.get("window_start"),
        summary.get("window_end"),
        summary["fetched"],
        summary["inserted"],
        summary["duplicates"],
        summary["near_duplicates"],
        summary["clustered"],
        summary["errors"],
        summary.get("gdelt_calls", 0),
        summary["status"],
        json.dumps(summary["details"], default=str),
        summary.get("redis_commands", 0),
    )


async def load_cursor(conn: Any, source: str) -> str | None:
    value: str | None = await conn.fetchval(
        "select cursor from public.source_cursor where source = $1", source
    )
    return value


async def save_cursor(conn: Any, source: str, cursor: str, now: datetime) -> None:
    await conn.execute(
        "insert into public.source_cursor (source, cursor, updated_at) values ($1, $2, $3) "
        "on conflict (source) do update set cursor = excluded.cursor, "
        "updated_at = excluded.updated_at",
        source,
        cursor,
        now,
    )


RETENTION_BATCH = 5000


async def _delete_batched(conn: Any, table: str, column: str, before: datetime) -> int:
    """Delete rows with ``column < before`` in batches (by ctid), so no statement runs long."""
    total = 0
    while True:
        n = await conn.fetchval(
            f"with gone as (delete from public.{table} where ctid in ("  # noqa: S608 (constants)
            f"select ctid from public.{table} where {column} < $1 limit {RETENTION_BATCH}) "
            "returning 1) select count(*) from gone",
            before,
        )
        total += int(n)
        if n < RETENTION_BATCH:
            return total


async def apply_retention(
    conn: Any,
    now: datetime,
    articles: timedelta,
    coverage: timedelta,
    runs: timedelta,
    feed_ids: Sequence[str] | None = None,
) -> dict[str, int]:
    """Delete expired rows; return how many went from each table."""
    out = {
        "articles": await _delete_batched(conn, "articles", "first_seen_at", now - articles),
        "story_coverage": await _delete_batched(
            conn, "story_coverage", "first_seen_at", now - coverage
        ),
        "ingestion_runs": await _delete_batched(conn, "ingestion_runs", "started_at", now - runs),
    }
    if feed_ids is not None:
        gone = await conn.fetch(
            "delete from public.feed_state where not (feed_id = any($1::text[])) returning 1",
            list(feed_ids),
        )
        out["feed_state"] = len(gone)
    return out


async def database_size(conn: Any) -> dict[str, int]:
    row = await conn.fetchrow(
        "select pg_database_size(current_database()) as db, "
        "pg_total_relation_size('public.articles') as articles, "
        "pg_total_relation_size('public.story_coverage') as story_coverage"
    )
    return dict(row)
