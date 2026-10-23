"""Per-reader state in Redis: append-only, versioned, rebuilt from Postgres (ADR 0016).

Keys (all expire; Postgres holds the durable copy):

* ``u:{uid}:hist`` (list, 7 d): history entries, appended with ``RPUSH`` +
  ``LTRIM -60 -1``. Each entry carries the ``hist_version`` Postgres gave it in the
  same transaction as the event. Read: de-duplicated by article (latest wins), ≤ 50.
* ``u:{uid}:hidden`` (set, 7 d): hidden article ids via ``SADD``, plus a sentinel
  member ``_`` written on rebuild, so "missing" can be told from "empty".
* ``u:{uid}:seen`` (set, 3 d): viewed article ids (impression events).
* ``u:{uid}`` (hash, 7 d) and ``u:{uid}:slate`` (30 min): the user-vector cache and the
  feed slate (P3d).

No update is read-modify-write, so two overlapping event batches can't drop a click.

**Consistency:** a rebuild writes a *marker* entry holding the profile's
``hist_version`` at that moment, after the rebuilt rows. Every later click appends
entries with consecutive versions. The list is trusted only if the entries after the
last marker run exactly from ``marker + 1`` to ``profiles.hist_version`` (or, once the
marker has been trimmed away, form 60 consecutive versions ending there). Anything
else (a failed push, a lost key, a persona switch) means: rebuild from
``user_history``.
"""

from __future__ import annotations

import struct
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from firstread_api.metrics import REDIS_COMMANDS
from firstread_core.embedding import EMBED_DIM
from firstread_core.features import HISTORY_LEN
from firstread_core.topics import TOPIC_IDS

HIST_KEEP_RAW = 60  # raw entries kept; > HISTORY_LEN so re-clicks don't shrink history
USER_TTL_S = 7 * 86_400
SEEN_TTL_S = 3 * 86_400
SLATE_TTL_S = 30 * 60
SENTINEL = b"_"
TEXT_SOURCES = ("title", "title_summary")
_HEAD = struct.Struct("<q16s16sBBB")  # version, article, cluster, topic, text_source, flags
FLAG_MARKER = 1


def hist_key(uid: uuid.UUID) -> str:
    return f"u:{uid}:hist"


def hidden_key(uid: uuid.UUID) -> str:
    return f"u:{uid}:hidden"


def seen_key(uid: uuid.UUID) -> str:
    return f"u:{uid}:seen"


def user_key(uid: uuid.UUID) -> str:
    return f"u:{uid}"


def slate_key(uid: uuid.UUID) -> str:
    return f"u:{uid}:slate"


def all_keys(uid: uuid.UUID) -> list[str]:
    return [user_key(uid), hist_key(uid), hidden_key(uid), seen_key(uid), slate_key(uid)]


# ---- command counting (Upstash bills every command, pipelined or not; ADR 0011) ----


class CountingPipeline:
    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe
        self._names: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._pipe, name)
        if not callable(attr):
            return attr

        def queue(*args: Any, **kwargs: Any) -> CountingPipeline:
            attr(*args, **kwargs)
            self._names.append(name)
            return self

        return queue

    async def execute(self) -> list[Any]:
        for n in self._names:
            REDIS_COMMANDS.labels(n).inc()
        self._names = []
        out: list[Any] = await self._pipe.execute()
        return out


class CountingRedis:
    """A redis.asyncio client wrapper that counts commands by name."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def pipeline(self, transaction: bool = False) -> CountingPipeline:
        return CountingPipeline(self._client.pipeline(transaction=transaction))

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name in ("aclose", "close", "connection_pool") or not callable(attr):
            return attr

        async def call(*args: Any, **kwargs: Any) -> Any:
            REDIS_COMMANDS.labels(name).inc()
            return await attr(*args, **kwargs)

        return call


# ---- history entries ----


@dataclass(frozen=True)
class HistoryEntry:
    version: int
    article_id: uuid.UUID
    cluster: uuid.UUID
    topic: str
    text_source: str
    vector: NDArray[np.float16] | None  # None for markers
    marker: bool = False


def pack(e: HistoryEntry) -> bytes:
    head = _HEAD.pack(
        e.version,
        e.article_id.bytes,
        e.cluster.bytes,
        TOPIC_IDS.index(e.topic) if not e.marker else 0,
        TEXT_SOURCES.index(e.text_source) if not e.marker else 0,
        FLAG_MARKER if e.marker else 0,
    )
    if e.marker:
        return head
    vec = np.asarray(e.vector, np.float16)
    if vec.shape != (EMBED_DIM,):
        raise ValueError(f"history vector must be {EMBED_DIM}-d")
    return head + vec.tobytes()


def unpack(raw: bytes) -> HistoryEntry:
    version, aid, cluster, topic, src, flags = _HEAD.unpack_from(raw)
    if flags & FLAG_MARKER:
        return HistoryEntry(version, uuid.UUID(int=0), uuid.UUID(int=0), "", "", None, True)
    vec = np.frombuffer(raw, np.float16, count=EMBED_DIM, offset=_HEAD.size)
    return HistoryEntry(
        version, uuid.UUID(bytes=aid), uuid.UUID(bytes=cluster), TOPIC_IDS[topic],
        TEXT_SOURCES[src], vec,
    )  # fmt: skip


def marker(version: int) -> HistoryEntry:
    return HistoryEntry(version, uuid.UUID(int=0), uuid.UUID(int=0), "", "", None, True)


def is_consistent(entries: Sequence[HistoryEntry], hist_version: int) -> bool:
    """Whether the raw list provably holds every history write up to ``hist_version``."""
    if hist_version == 0:
        return not [e for e in entries if not e.marker]
    last = max((i for i, e in enumerate(entries) if e.marker), default=None)
    if last is not None:
        tail = [e.version for e in entries[last + 1 :]]
        start = entries[last].version + 1
        return tail == list(range(start, hist_version + 1))
    versions = [e.version for e in entries]
    if not versions or versions != list(range(versions[0], versions[0] + len(versions))):
        return False
    # No marker: either the full history since version 1, or a full trimmed window.
    return versions[-1] == hist_version and (versions[0] == 1 or len(versions) >= HIST_KEEP_RAW)


def latest(entries: Iterable[HistoryEntry], limit: int = HISTORY_LEN) -> list[HistoryEntry]:
    """De-duplicate by article (latest version wins), oldest first, at most ``limit``."""
    best: dict[uuid.UUID, HistoryEntry] = {}
    for e in entries:
        if not e.marker and (e.article_id not in best or e.version > best[e.article_id].version):
            best[e.article_id] = e
    return sorted(best.values(), key=lambda e: e.version)[-limit:]


# ---- writes (append-only) ----


def append_history(pipe: CountingPipeline, uid: uuid.UUID, entries: Sequence[HistoryEntry]) -> None:
    """``RPUSH`` + ``LTRIM`` + ``EXPIRE``: 3 commands for any number of new entries."""
    if not entries:
        return
    key = hist_key(uid)
    pipe.rpush(key, *[pack(e) for e in sorted(entries, key=lambda e: e.version)])
    pipe.ltrim(key, -HIST_KEEP_RAW, -1)
    pipe.expire(key, USER_TTL_S)


def add_members(pipe: CountingPipeline, key: str, ids: Iterable[uuid.UUID], ttl: int) -> None:
    """``SADD`` + ``EXPIRE``: 2 commands."""
    members = [str(i).encode() for i in ids]
    if members:
        pipe.sadd(key, *members)
        pipe.expire(key, ttl)


async def rebuild_history(
    redis: CountingRedis, uid: uuid.UUID, rows: Sequence[HistoryEntry], hist_version: int
) -> None:
    """Replace the list with Postgres' rows plus a marker (``DEL`` + ``RPUSH`` +
    ``EXPIRE``). A click landing between the commands only causes another rebuild."""
    pipe = redis.pipeline()
    pipe.delete(hist_key(uid))
    pipe.rpush(hist_key(uid), *[pack(e) for e in sorted(rows, key=lambda e: e.version)],
               pack(marker(hist_version)))  # fmt: skip
    pipe.expire(hist_key(uid), USER_TTL_S)
    await pipe.execute()


async def rebuild_hidden(redis: CountingRedis, uid: uuid.UUID, ids: Iterable[uuid.UUID]) -> None:
    pipe = redis.pipeline()
    pipe.sadd(hidden_key(uid), SENTINEL, *[str(i).encode() for i in ids])
    pipe.expire(hidden_key(uid), USER_TTL_S)
    await pipe.execute()


def hidden_from(members: Iterable[bytes]) -> set[uuid.UUID] | None:
    """Parse ``SMEMBERS``: None when the sentinel is missing (key lost: rebuild)."""
    raw = set(members)
    if SENTINEL not in raw:
        return None
    return {uuid.UUID(m.decode()) for m in raw - {SENTINEL}}


async def delete_user(redis: CountingRedis, uid: uuid.UUID) -> None:
    await redis.delete(*all_keys(uid))
