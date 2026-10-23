"""Postgres queries for profiles, history and events (ADR 0016).

Every query is named, so ``db_bytes_estimated_total{query}`` shows what each one
costs in Supabase egress (ADR 0011). Vectors are copied server-side
(``INSERT ... SELECT``) wherever possible; the only vectors that cross the wire are
the newly clicked articles', so they can be appended to the Redis history.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from firstread_api.auth import Principal
from firstread_api.metrics import DB_BYTES, DB_QUERIES
from firstread_api.redis_state import HistoryEntry
from firstread_api.schemas import EventIn
from firstread_core.db import rows_bytes
from firstread_core.features import HISTORY_LEN

POSITIVE = {"click": "click", "like": "like"}  # event type -> history kind


class Q:
    """Named, metered queries on one connection."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def _count(self, name: str, result: Any) -> None:
        DB_QUERIES.labels(name).inc()
        DB_BYTES.labels(name).inc(rows_bytes(result) if result is not None else 16)

    async def fetch(self, name: str, sql: str, *args: Any) -> list[Any]:
        rows: list[Any] = await self.conn.fetch(sql, *args)
        self._count(name, rows)
        return rows

    async def fetchrow(self, name: str, sql: str, *args: Any) -> Any:
        row = await self.conn.fetchrow(sql, *args)
        self._count(name, row)
        return row

    async def fetchval(self, name: str, sql: str, *args: Any) -> Any:
        v = await self.conn.fetchval(sql, *args)
        self._count(name, None)
        return v

    async def execute(self, name: str, sql: str, *args: Any) -> str:
        status: str = await self.conn.execute(sql, *args)
        self._count(name, None)
        return status

    def transaction(self) -> Any:
        return self.conn.transaction()


# ---- profiles ----


async def ensure_profile(q: Q, user: Principal, local_auth: bool) -> None:
    """Create the profile on first sight; touch ``last_seen_at`` at most once a day.

    Locally (no Supabase Auth) the shim ``auth.users`` row is created first, so the
    foreign keys hold. On Supabase, Auth creates that row at sign-in.
    """
    if local_auth:
        await q.execute(
            "auth_user_local",
            "insert into auth.users (id, is_anonymous) values ($1, $2) on conflict do nothing",
            user.user_id,
            user.is_anonymous,
        )
    await q.execute(
        "profile_ensure",
        "insert into public.profiles (user_id, is_anonymous) values ($1, $2) "
        "on conflict (user_id) do update set last_seen_at = now(), "
        "is_anonymous = excluded.is_anonymous "
        "where profiles.last_seen_at < now() - interval '1 day' "
        "or profiles.is_anonymous is distinct from excluded.is_anonymous",
        user.user_id,
        user.is_anonymous,
    )


async def set_topics(q: Q, uid: uuid.UUID, topics: Sequence[str]) -> None:
    await q.execute(
        "profile_topics",
        "update public.profiles set topics = $2, updated_at = now() where user_id = $1",
        uid,
        list(topics),
    )


async def me(q: Q, uid: uuid.UUID) -> dict[str, Any]:
    prof = await q.fetchrow(
        "me_profile",
        "select topics, hidden_sources, is_anonymous, active_persona, "
        "(select count(*) from public.user_history h where h.user_id = p.user_id) as n "
        "from public.profiles p where user_id = $1",
        uid,
    )
    recent = await q.fetch(
        "me_recent",
        "select article_id, title, source_name, category, clicked_at from public.user_history "
        "where user_id = $1 order by version desc limit 5",
        uid,
    )
    return {"profile": dict(prof), "recent": [dict(r) for r in recent]}


async def delete_user_data(q: Q, uid: uuid.UUID) -> None:
    """Every app row of this reader. The auth user itself is deleted in P5 (service key)."""
    async with q.transaction():
        for table in ("served_slates", "events", "user_history", "profiles"):
            await q.execute(
                f"delete_{table}",
                f"delete from public.{table} where user_id = $1",  # noqa: S608 (constant names)
                uid,
            )


# ---- events ----


@dataclass
class EventsOutcome:
    accepted: list[uuid.UUID] = field(default_factory=list)
    duplicate: list[uuid.UUID] = field(default_factory=list)
    rejected: list[uuid.UUID] = field(default_factory=list)
    history: list[HistoryEntry] = field(default_factory=list)  # for the Redis append
    hidden: list[uuid.UUID] = field(default_factory=list)
    seen: list[uuid.UUID] = field(default_factory=list)
    hidden_sources_added: list[str] = field(default_factory=list)


_INSERT_EVENTS = """
insert into public.events (id, user_id, article_id, type, scope, position, dwell_ms,
                           request_id, model_version, client_ts)
select e.id, $1, e.article_id, e.type, e.scope, e.position, e.dwell_ms, e.request_id, $2, e.ts
from unnest($3::uuid[], $4::uuid[], $5::text[], $6::text[], $7::int[], $8::int[], $9::text[],
            $10::timestamptz[])
  as e(id, article_id, type, scope, position, dwell_ms, request_id, ts)
on conflict (id) do nothing
returning id
"""

_UPSERT_HISTORY = """
insert into public.user_history (user_id, article_id, version, kind, clicked_at, title,
  source_name, source_domain, url, category, story_cluster_id, text_source, embedding)
select $1, a.id, v.version, v.kind, v.ts, a.title, a.source_name, a.source_domain, a.url,
  a.category, a.story_cluster_id, a.text_source, a.embedding
from unnest($2::uuid[], $3::bigint[], $4::text[], $5::timestamptz[]) as v(id, version, kind, ts)
join public.articles a on a.id = v.id
on conflict (user_id, article_id) do update set version = excluded.version,
  kind = excluded.kind, clicked_at = excluded.clicked_at
returning article_id, version, story_cluster_id, category, text_source, embedding
"""

_TRIM_HISTORY = f"""
delete from public.user_history where user_id = $1 and version < (
  select min(version) from (
    select version from public.user_history where user_id = $1
    order by version desc limit {HISTORY_LEN}
  ) newest
)
"""  # noqa: S608 (constant)


def entry_from_row(r: Any) -> HistoryEntry:
    return HistoryEntry(
        version=int(r["version"]),
        article_id=r["article_id"],
        cluster=r["story_cluster_id"],
        topic=r["category"],
        text_source=r["text_source"],
        vector=np.asarray(r["embedding"].to_numpy(), np.float16),
    )


async def process_events(
    q: Q, uid: uuid.UUID, events: Sequence[EventIn], model_version: str
) -> EventsOutcome:
    """Insert a batch idempotently; side effects only for rows actually inserted."""
    out = EventsOutcome()
    wanted = sorted({e.article_id for e in events if e.article_id is not None})
    known: dict[uuid.UUID, str] = {}
    if wanted:
        rows = await q.fetch(
            "events_articles",
            "select id, source_domain from public.articles where id = any($1::uuid[])",
            wanted,
        )
        known = {r["id"]: r["source_domain"] for r in rows}
    ok = []
    for e in events:
        if e.article_id is not None and e.article_id not in known:
            out.rejected.append(e.event_id)
        else:
            ok.append(e)
    if not ok:
        return out
    now = datetime.now(UTC)
    async with q.transaction():
        inserted_rows = await q.fetch(
            "events_insert",
            _INSERT_EVENTS,
            uid,
            model_version,
            [e.event_id for e in ok],
            [e.article_id for e in ok],
            [e.type for e in ok],
            [e.scope for e in ok],
            [e.position for e in ok],
            [e.dwell_ms for e in ok],
            [e.request_id for e in ok],
            [e.client_ts for e in ok],
        )
        inserted = {r["id"] for r in inserted_rows}
        new = [e for e in ok if e.event_id in inserted]
        out.accepted = [e.event_id for e in new]
        out.duplicate = [e.event_id for e in ok if e.event_id not in inserted]
        # One history write per article: the batch's last positive event wins.
        positives: dict[uuid.UUID, EventIn] = {}
        for e in new:
            if e.type in POSITIVE and e.article_id is not None:
                positives.pop(e.article_id, None)
                positives[e.article_id] = e
        if positives:
            n = len(positives)
            top = await q.fetchval(
                "profile_version",
                "update public.profiles set hist_version = hist_version + $2, updated_at = now() "
                "where user_id = $1 returning hist_version",
                uid,
                n,
            )
            versions = list(range(int(top) - n + 1, int(top) + 1))
            items = list(positives.values())
            rows = await q.fetch(
                "history_upsert",
                _UPSERT_HISTORY,
                uid,
                [e.article_id for e in items],
                versions,
                [POSITIVE[e.type] for e in items],
                [e.client_ts or now for e in items],
            )
            out.history = [entry_from_row(r) for r in rows]
            await q.execute("history_trim", _TRIM_HISTORY, uid)
        hide_sources = sorted(
            {known[e.article_id] for e in new
             if e.type == "hide" and e.scope == "source" and e.article_id is not None}
        )  # fmt: skip
        if hide_sources:
            await q.execute(
                "profile_hidden_sources",
                "update public.profiles set hidden_sources = (select array(select distinct x "
                "from unnest(hidden_sources || $2::text[]) x order by x)), updated_at = now() "
                "where user_id = $1",
                uid,
                hide_sources,
            )
            out.hidden_sources_added = hide_sources
    out.hidden = [
        e.article_id for e in new
        if e.type == "hide" and e.scope != "source" and e.article_id is not None
    ]  # fmt: skip
    out.seen = [e.article_id for e in new if e.type == "impression" and e.article_id is not None]
    return out


# ---- rebuild sources (when the Redis copy is missing or has a gap) ----


async def history_for_rebuild(q: Q, uid: uuid.UUID) -> tuple[list[HistoryEntry], int]:
    rows = await q.fetch(
        "history_rebuild",
        "select article_id, version, story_cluster_id, category, text_source, embedding "
        "from public.user_history where user_id = $1 order by version",
        uid,
    )
    version = await q.fetchval(
        "profile_version_read", "select hist_version from public.profiles where user_id = $1", uid
    )
    return [entry_from_row(r) for r in rows], int(version or 0)


async def hidden_for_rebuild(q: Q, uid: uuid.UUID) -> list[uuid.UUID]:
    rows = await q.fetch(
        "hidden_rebuild",
        "select distinct article_id from public.events where user_id = $1 and type = 'hide' "
        "and coalesce(scope, 'article') = 'article' and article_id is not null",
        uid,
    )
    return [r["article_id"] for r in rows]
