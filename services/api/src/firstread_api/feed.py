"""The feed pipeline: seven stages, each with a deadline and a tested way to fail (ADR 0017).

1. **auth**: the JWT (the route's dependency, timed there). 401, or 503 if the
   JWKS can't be fetched.
2. **state**: Redis reads ∥ the profile row; version check and rebuilds. Redis down:
   history and hides from Postgres, no seen set → ``degraded: redis``.
3. **encode**: the cached vector, else the micro-batched encoder; then the
   cold-start blend. Encoder error or timeout: mean of history → ``encoder``.
4. **cand**: ANN ∥ pool cosines, both in SQL. ANN down: pools only (``ann``); no
   snapshot yet: ANN only (``pools``); both down: 503.
5. **hydrate**: the top ``rerank_size`` by the pre-hydrate blend, from the LRU. 503.
6. **rank**: read-story filter (τ_story), story groups (τ_feed, complete linkage),
   MMR with the story cap, reasons.
7. **persist**: the ``served_slates`` row and the Redis slate, before responding.
   Postgres down: 503; Redis down: ``redis``.

The whole request has its own deadline (``API_FEED_TIMEOUT_S``, 1 s) → 503. Nothing
runs after the response (Cloud Run bills CPU only during requests).

**Pages.** The first page (no cursor) ranks a new slate. A cursor page is served from
the stored slate: reading it costs 2 Redis commands (``GET`` slate, ``SMEMBERS``
hidden). Only when the reader clicked (``hist_version`` moved), hid something, or
the slate ran low does it read the rest of the state and re-rank, excluding
everything already served.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from firstread_api import redis_state as rs
from firstread_api import repo
from firstread_api.articles_cache import Article
from firstread_api.candidates import Candidate, Exclusions, ann, pool_scores
from firstread_api.deps import Services
from firstread_api.errors import ApiError
from firstread_api.metrics import DEGRADED, FEED_STAGE, HISTORY_REBUILDS
from firstread_api.slate import (
    MAX_SERVED,
    Cursor,
    Slate,
    SlateItem,
    cursor_expired,
    new_slate_id,
)
from firstread_core import pools, stories
from firstread_core.embedding import l2_normalize
from firstread_core.features import mean_history_vector, pad_history
from firstread_core.ranking import (
    ReasonInputs,
    blend,
    cold_start_vector,
    freshness,
    mmr,
    reasons,
    topic_weight,
    trend_feature,
)

log = structlog.get_logger("firstread_api.feed")

WINDOW = timedelta(hours=72)  # candidates are at most this old (SPEC §8)
ANN_LIMIT = 200
FRESH_TAKE, TRENDING_TAKE, TOPIC_TAKE = 200, 200, 100
SHOWN_FOR_CAP = 100  # the newest served items that count toward the per-story cap
STAGES = ("auth", "state", "encode", "cand", "hydrate", "rank", "persist")


def unavailable(message: str) -> ApiError:
    return ApiError(503, message, "unavailable", headers={"Retry-After": "5"})


# ---- timing ----


@dataclass
class Timings:
    ms: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        self.ms[stage] = self.ms.get(stage, 0.0) + seconds * 1000

    def observe(self) -> None:
        """One histogram observation per stage that ran (a stage may run in parts)."""
        for stage, ms in self.ms.items():
            FEED_STAGE.labels(stage).observe(ms / 1000)

    @asynccontextmanager
    async def stage(self, name: str) -> AsyncIterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def header(self) -> str:
        """``Server-Timing``: one ``name;dur=<ms>`` per stage that ran, in pipeline order."""
        return ", ".join(f"{s};dur={self.ms[s]:.1f}" for s in STAGES if s in self.ms)

    def rounded(self) -> dict[str, float]:
        return {s: round(self.ms[s], 2) for s in STAGES if s in self.ms}


async def within[R](seconds: float, aw: Awaitable[R]) -> R:
    async with asyncio.timeout(seconds):
        return await aw


# ---- state ----


@dataclass
class Profile:
    hist_version: int
    topics: list[str]
    hidden_sources: list[str]


@dataclass
class State:
    profile: Profile
    hidden: set[uuid.UUID]
    history: list[rs.HistoryEntry] = field(default_factory=list)  # oldest first, ≤ 50
    seen: set[uuid.UUID] = field(default_factory=set)
    cached_vec: NDArray[np.float32] | None = None
    full: bool = False  # history, seen and the cached vector were read

    def hidden_sig(self) -> str:
        """Changes whenever a hidden article or source is added (re-rank the slate)."""
        h = hashlib.sha256()
        for x in sorted(str(i) for i in self.hidden):
            h.update(x.encode())
        h.update(b"|" + ",".join(sorted(self.profile.hidden_sources)).encode())
        return h.hexdigest()[:16]


_PROFILE = "select hist_version, topics, hidden_sources from public.profiles where user_id = $1"
_SERVED = (
    "select page, article_ids from public.served_slates where user_id = $1 and slate_id = $2 "
    "order by page"
)


def vec_key(hist_version: int, at: float) -> str:
    return f"{hist_version}:{int(at)}"


def parse_vec(
    raw: list[Any], hist_version: int, ttl_s: float, now: float
) -> NDArray[np.float32] | None:
    """The cached encoder output if it was computed for this history version < ttl ago."""
    vec, key = raw if len(raw) == 2 else (None, None)
    if vec is None or key is None:
        return None
    try:
        version, at = (int(x) for x in key.decode().split(":"))
    except ValueError:
        return None
    if version != hist_version or now - at >= ttl_s or len(vec) != 4 * 384:
        return None
    return np.frombuffer(vec, np.float32).copy()


class FeedPipeline:
    def __init__(self, svc: Services, uid: uuid.UUID, request_id: str) -> None:
        self.svc = svc
        self.s = svc.settings
        self.cfg = svc.ranking
        self.uid = uid
        self.request_id = request_id
        self.t = Timings()
        self.degraded: list[str] = []
        self.redis_ok = True
        self.new_vec: NDArray[np.float32] | None = None

    def degrade(self, reason: str, error: BaseException | None = None) -> None:
        if reason not in self.degraded:
            self.degraded.append(reason)
            DEGRADED.labels(reason).inc()
            log.warning("feed_degraded", reason=reason,
                        error=type(error).__name__ if error else None)  # fmt: skip

    async def _q[R](self, deadline: float, fn: Callable[[repo.Q], Awaitable[R]]) -> R:
        """Run ``fn`` on a pooled connection. The deadline covers the query, not the wait
        for a connection (a cold pool opening one is bounded by the request deadline)."""
        async with self.svc.pool.acquire() as conn:
            return await within(deadline, fn(repo.Q(conn)))

    # -- stage 2 --

    async def _profile(self) -> Profile:
        row = await self._q(
            self.s.api_state_timeout_s, lambda q: q.fetchrow("feed_profile", _PROFILE, self.uid)
        )
        if row is None:  # ensure_profile ran just before; only a concurrent delete gets here
            raise unavailable("profile not found")
        return Profile(int(row["hist_version"]), list(row["topics"]), list(row["hidden_sources"]))

    async def _redis(self, *queue: tuple[str, tuple[Any, ...]]) -> list[Any]:
        pipe = self.svc.redis.pipeline()
        for name, args in queue:
            getattr(pipe, name)(*args)
        return await within(self.s.api_state_timeout_s, pipe.execute())

    async def light_state(self, cursor: Cursor) -> tuple[State, Slate | None]:
        """For a cursor page: the slate and hidden set (2 commands) ∥ the profile row."""
        uid = self.uid
        reads = self._redis(("get", (rs.slate_key(uid),)), ("smembers", (rs.hidden_key(uid),)))
        got, profile = await asyncio.gather(reads, self._profile(), return_exceptions=True)
        if isinstance(profile, BaseException):
            raise profile
        slate: Slate | None = None
        hidden: set[uuid.UUID] | None = None
        if isinstance(got, BaseException):
            self.redis_ok = False
            self.degrade("redis", got)
        else:
            raw_slate, members = got
            hidden = rs.hidden_from(members)
            if raw_slate is not None:
                try:
                    slate = Slate.loads(raw_slate)
                except Exception:  # an unreadable slate is a missing slate
                    slate = None
        if slate is not None and slate.id != cursor.slate_id:
            slate = None  # a newer first page replaced it
        if slate is not None and len(slate.served) < min(cursor.offset, MAX_SERVED):
            # Behind the cursor: a page was served and recorded in served_slates but
            # its slate write was lost (Redis failed then). Served ids come from
            # Postgres, or the next page would repeat what the reader just saw.
            slate = None
        if slate is None:
            slate = await self._slate_from_postgres(cursor)
        if hidden is None:
            hidden = await self._rebuild_hidden()
        return State(profile, hidden), slate

    async def _redis_or_error(self, queue: list[tuple[str, tuple[Any, ...]]]) -> Any:
        if not self.redis_ok:
            return ConnectionError("Redis already failed in this request")
        try:
            return await self._redis(*queue)
        except Exception as exc:
            return exc

    async def full_state(self, light: State | None = None) -> State:
        """History, seen ids, the cached vector (+ hidden ids unless ``light`` has them):
        3 or 4 commands, in parallel with the profile row (unless ``light`` has it)."""
        uid = self.uid
        queue: list[tuple[str, tuple[Any, ...]]] = [
            ("hmget", (rs.user_key(uid), "vec", "vec_key")),
            ("lrange", (rs.hist_key(uid), 0, -1)),
            ("smembers", (rs.seen_key(uid),)),
        ]
        if light is None:
            queue.append(("smembers", (rs.hidden_key(uid),)))
            got, profile = await asyncio.gather(self._redis_or_error(queue), self._profile())
        else:
            got, profile = await self._redis_or_error(queue), light.profile
        if isinstance(got, BaseException):
            if self.redis_ok:
                self.redis_ok = False
                self.degrade("redis", got)
            rows, version = await self._history_from_postgres()
            hidden = light.hidden if light else set(await self._hidden_from_postgres())
            profile.hist_version = version  # the same snapshot as the rows
            return State(profile, hidden, rs.latest(rows), set(), None, full=True)
        raw_vec, raw_hist, seen_members = got[0], got[1], got[2]
        entries = [rs.unpack(x) for x in raw_hist]
        if not rs.is_consistent(entries, profile.hist_version):
            HISTORY_REBUILDS.labels("history").inc()
            entries, profile.hist_version = await self._history_from_postgres()
            await self._quiet(
                rs.rebuild_history(self.svc.redis, uid, entries, profile.hist_version)
            )
        if light is not None:
            hidden = light.hidden
        else:
            parsed = rs.hidden_from(got[3])
            hidden = parsed if parsed is not None else await self._rebuild_hidden()
        seen = {uuid.UUID(m.decode()) for m in seen_members}
        vec = parse_vec(list(raw_vec), profile.hist_version, self.s.api_vec_ttl_s, time.time())
        return State(profile, hidden, rs.latest(entries), seen, vec, full=True)

    async def _history_from_postgres(self) -> tuple[list[rs.HistoryEntry], int]:
        async with self.svc.pool.acquire() as conn:
            return await repo.history_for_rebuild(repo.Q(conn), self.uid)

    async def _hidden_from_postgres(self) -> list[uuid.UUID]:
        async with self.svc.pool.acquire() as conn:
            return await repo.hidden_for_rebuild(repo.Q(conn), self.uid)

    async def _rebuild_hidden(self) -> set[uuid.UUID]:
        ids = await self._hidden_from_postgres()
        if self.redis_ok:
            HISTORY_REBUILDS.labels("hidden").inc()
            await self._quiet(rs.rebuild_hidden(self.svc.redis, self.uid, ids))
        return set(ids)

    async def _slate_from_postgres(self, cursor: Cursor) -> Slate:
        """The served ids of a slate whose Redis copy is gone (expired or Redis down)."""
        async with self.svc.pool.acquire() as conn:
            rows = await repo.Q(conn).fetch(
                "feed_served", _SERVED, self.uid, uuid.UUID(cursor.slate_id)
            )
        if not rows:
            raise cursor_expired()
        # Once each (a repeat within 60 s records the same ids again), oldest first.
        served = list(dict.fromkeys(str(a) for r in rows for a in r["article_ids"]))
        return Slate(cursor.slate_id, -1, "", [], served, {}, int(rows[-1]["page"]) + 1)

    async def _quiet(self, aw: Awaitable[Any]) -> None:
        """A best-effort Redis write: a failure only costs a rebuild later."""
        try:
            await within(self.s.api_state_timeout_s, aw)
        except Exception as exc:
            log.warning("feed_redis_write_failed", error=type(exc).__name__)

    # -- stage 3 --

    async def user_vector(
        self, st: State, snap: pools.PoolSnapshot | None
    ) -> tuple[NDArray[np.float32], float]:
        """The reader's vector and the share of it that came from their chosen topics."""
        batcher = self.svc.batcher
        if batcher is None:
            raise unavailable("model not loaded")
        n = len(st.history)
        click: NDArray[np.float32] | None = None
        if n:
            click = st.cached_vec
            if click is None:
                vecs = np.stack([np.asarray(e.vector, np.float32) for e in st.history])
                hist, mask = pad_history(vecs)
                try:
                    out = await within(self.s.api_encode_timeout_s, batcher.encode(hist, mask))
                    click = self.new_vec = np.asarray(out, np.float32)
                except Exception as exc:  # timeout, ONNX error, shutting down
                    self.degrade("encoder", exc)
                    click = mean_history_vector(hist, mask)
        centroids = snap.centroids if snap is not None else {}
        k = self.cfg.cold_start_k
        u = cold_start_vector(centroids, st.profile.topics, click, n, k, batcher.encoder.population)
        return l2_normalize(u), topic_weight(n, k) if st.profile.topics else 0.0

    # -- stage 4 --

    async def candidates(
        self,
        u: NDArray[np.float32],
        st: State,
        snap: pools.PoolSnapshot | None,
        served: Sequence[str],
    ) -> list[Candidate]:
        clicked = {e.article_id for e in st.history}
        excl_ids = st.seen | clicked | st.hidden | {uuid.UUID(i) for i in served}
        excl = Exclusions(
            ids=sorted(excl_ids),
            sources=sorted(st.profile.hidden_sources),
            clusters=sorted({e.cluster for e in st.history}),
            hidden=sorted(st.hidden),
        )
        since = datetime.now(UTC) - WINDOW
        deadline = self.s.api_candidates_timeout_s

        async def run_ann() -> list[Candidate]:
            return await self._q(deadline, lambda q: ann(
                q, u, since, excl, ANN_LIMIT, self.s.api_ann_mode, self.s.api_ann_ef_search
            ))  # fmt: skip

        pool_tags: dict[uuid.UUID, set[str]] = {}
        if snap is not None:
            picks: list[tuple[str, Sequence[uuid.UUID]]] = [
                ("fresh", snap.fresh[:FRESH_TAKE]),
                ("trending", [t.id for t in snap.trending[:TRENDING_TAKE]]),
                *((f"topic:{t}", snap.topics.get(t, [])[:TOPIC_TAKE]) for t in st.profile.topics),
            ]
            for tag, ids in picks:
                for i in ids:
                    if i not in excl_ids:
                        pool_tags.setdefault(i, set()).add(tag)

        async def run_pools() -> list[Candidate]:
            ids = sorted(pool_tags)
            return await self._q(deadline, lambda q: pool_scores(q, u, ids, excl))

        if snap is None:
            self.degrade("pools")
            ann_res: Any = await asyncio.gather(run_ann(), return_exceptions=True)
            ann_res, pool_res = ann_res[0], None
        else:
            ann_res, pool_res = await asyncio.gather(run_ann(), run_pools(), return_exceptions=True)
        if isinstance(ann_res, BaseException):
            self.degrade("ann", ann_res)
            ann_res = None
        if isinstance(pool_res, BaseException):
            self.degrade("pools", pool_res)
            pool_res = None
        if ann_res is None and pool_res is None:
            raise unavailable("no candidate source answered")
        merged: dict[uuid.UUID, Candidate] = {}
        for c in [*(ann_res or []), *(pool_res or [])]:
            have = merged.get(c.id)
            if have is None:
                merged[c.id] = c
            else:
                have.sources |= c.sources
        for c in merged.values():
            c.sources = (c.sources - {"pool"}) | pool_tags.get(c.id, set())
        return list(merged.values())

    # -- the blend (before hydration: every candidate carries age, story and source) --

    def score(
        self, cands: Sequence[Candidate], snap: pools.PoolSnapshot | None, n_history: int
    ) -> dict[str, NDArray[np.float64]]:
        trend_by_cluster: dict[uuid.UUID, tuple[float, int]] = {}
        if snap is not None:
            for t in snap.trending:
                for c in t.clusters:
                    trend_by_cluster.setdefault(c, (t.score, t.outlets))
        now = datetime.now(UTC)
        model = np.array([c.score for c in cands], np.float64)
        age_h = np.array([(now - c.first_seen_at).total_seconds() / 3600 for c in cands])
        raw_trend = np.array([trend_by_cluster.get(c.cluster, (0.0, 0))[0] for c in cands])
        outlets = np.array([trend_by_cluster.get(c.cluster, (0.0, 0))[1] for c in cands])
        w_fresh, w_trend = self.cfg.weights_for(n_history)
        fresh = freshness(age_h, self.cfg.half_life_hours)
        trend = trend_feature(raw_trend)
        final = blend(model, fresh, trend, self.cfg.model_norm, w_fresh, w_trend)
        return {
            "final": final,
            "model": final - w_fresh * fresh - w_trend * trend,  # the normalized model term
            "fresh": w_fresh * fresh,
            "trend": w_trend * trend,
            "age_h": age_h,
            "outlets": outlets.astype(np.float64),
        }

    # -- stage 6 --

    async def rank(
        self,
        cands: Sequence[Candidate],
        comp: dict[str, NDArray[np.float64]],
        rows: dict[uuid.UUID, Article],
        st: State,
        w_topic: float,
        served: Sequence[Article] = (),
    ) -> list[SlateItem]:
        read_clusters = {e.cluster for e in st.history}
        keep = [
            i for i, c in enumerate(cands)
            if c.id in rows and rows[c.id].story_cluster_id not in read_clusters
        ]  # fmt: skip
        if not keep:
            return []
        keep.sort(key=lambda i: -comp["final"][i])
        vecs = np.stack([np.asarray(rows[cands[i].id].vector, np.float32) for i in keep])
        srcs = [rows[cands[i].id].text_source for i in keep]
        hist_vecs = (
            np.stack([np.asarray(e.vector, np.float32) for e in st.history])
            if st.history else np.zeros((0, vecs.shape[1]), np.float32)
        )  # fmt: skip
        hist_srcs = [e.text_source for e in st.history]
        near = stories.near_any(vecs, srcs, hist_vecs, hist_srcs)  # type: ignore[arg-type]
        sel = [k for k in range(len(keep)) if not near[k]]
        keep, vecs, srcs = [keep[k] for k in sel], vecs[sel], [srcs[k] for k in sel]
        if not keep:
            return []
        # Story groups over what this reader was already shown (leading) + the candidates,
        # so the 2-per-story cap holds across pages, not only within one ranking pass.
        shown = [a for a in served if a.story_cluster_id not in read_clusters]
        clusters = [a.story_cluster_id for a in shown] + [
            rows[cands[i].id].story_cluster_id for i in keep
        ]
        all_vecs = np.concatenate([
            np.stack([np.asarray(a.vector, np.float32) for a in shown])
            if shown else np.zeros((0, vecs.shape[1]), np.float32), vecs,
        ])  # fmt: skip
        all_groups = stories.group(all_vecs, [a.text_source for a in shown] + srcs, clusters)  # type: ignore[arg-type]
        groups = all_groups[len(shown) :]
        used = Counter(all_groups[: len(shown)])
        final = comp["final"][keep]
        res = mmr(final, vecs, self.cfg.mmr_lambda, groups, cap=self.cfg.story_cap,
                  k=self.cfg.slate_size)  # fmt: skip
        order = []
        for k in res.order:
            if used[groups[k]] < self.cfg.story_cap:
                used[groups[k]] += 1
                order.append(k)
        nearest: dict[int, uuid.UUID] = {}
        titles: dict[uuid.UUID, str] = {}
        if len(hist_vecs) and w_topic < 0.5:  # only then can "similar to read" show
            sims = vecs[order] @ hist_vecs.T
            nearest = {k: st.history[int(np.argmax(sims[r]))].article_id
                       for r, k in enumerate(order)}  # fmt: skip
            titles = await self._titles(set(nearest.values()))
        items = []
        for k in order:
            i = keep[k]
            a = rows[cands[i].id]
            why = reasons(ReasonInputs(
                model=float(comp["model"][i]), fresh=float(comp["fresh"][i]),
                trend=float(comp["trend"][i]), age_hours=float(comp["age_h"][i]),
                outlets=int(comp["outlets"][i]), topic=a.category, followed=st.profile.topics,
                topic_weight=w_topic, nearest_read=titles.get(nearest.get(k, uuid.UUID(int=0))),
                diversity_pick=k in res.diversity_picks,
            ))  # fmt: skip
            items.append(SlateItem(
                str(a.id), round(float(comp["final"][i]), 6), round(float(comp["model"][i]), 6),
                round(float(comp["fresh"][i]), 6), round(float(comp["trend"][i]), 6),
                [[r.code, r.text] for r in why],
            ))  # fmt: skip
        return items

    async def _titles(self, ids: set[uuid.UUID]) -> dict[uuid.UUID, str]:
        """Titles of history items named in "Because you read …" (best effort)."""
        try:
            rows = await self._q(self.s.api_hydrate_timeout_s, lambda q: q.fetch(
                "feed_history_titles",
                "select article_id, title from public.user_history "
                "where user_id = $1 and article_id = any($2::uuid[])",
                self.uid, sorted(ids),
            ))  # fmt: skip
        except Exception as exc:
            log.warning("feed_titles_failed", error=type(exc).__name__)
            return {}
        return {r["article_id"]: r["title"] for r in rows}

    async def hydrate(self, ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Article]:
        try:
            return await self._q(
                self.s.api_hydrate_timeout_s, lambda q: self.svc.articles.get_many(q, ids)
            )
        except Exception as exc:
            log.warning("feed_hydrate_failed", error=type(exc).__name__)
            raise unavailable("articles unavailable") from None

    # -- a full ranking pass: stages 3 to 6 --

    async def rank_slate(self, st: State, served: Sequence[str]) -> list[SlateItem]:
        async with self.t.stage("cand"):
            snap = await self.svc.pools.get()
        async with self.t.stage("encode"):
            u, w_topic = await self.user_vector(st, snap)
        async with self.t.stage("cand"):
            cands = await self.candidates(u, st, snap, served)
        if not cands:
            return []
        comp = self.score(cands, snap, len(st.history))
        top = np.argsort(-comp["final"], kind="stable")[: self.cfg.rerank_size]
        shown = [uuid.UUID(i) for i in served[max(len(served) - SHOWN_FOR_CAP, 0) :]]
        async with self.t.stage("hydrate"):  # one query for the misses of both
            rows = await self.hydrate([*(cands[i].id for i in top), *shown])
        async with self.t.stage("rank"):
            sub = [cands[i] for i in top]
            sub_comp = {k: v[top] for k, v in comp.items()}
            shown_rows = [rows[i] for i in shown if i in rows]
            return await self.rank(sub, sub_comp, rows, st, w_topic, shown_rows)

    # -- stage 7 --

    async def persist(self, slate: Slate, page: list[SlateItem], write_redis: bool) -> None:
        model_version = self.s.model_version

        async def insert(q: repo.Q) -> None:
            await q.execute(
                "feed_served_insert",
                "insert into public.served_slates (request_id, slate_id, user_id, page, "
                "model_version, article_ids, components) values ($1, $2, $3, $4, $5, $6, $7)",
                self.request_id, uuid.UUID(slate.id), self.uid, slate.page - 1, model_version,
                [uuid.UUID(i.id) for i in page], [x for i in page for x in i.components()],
            )  # fmt: skip

        async def to_redis() -> None:
            pipe = self.svc.redis.pipeline()
            pipe.set(rs.slate_key(self.uid), slate.dumps(), ex=rs.SLATE_TTL_S)
            if self.new_vec is not None:
                key = rs.user_key(self.uid)
                pipe.hset(key, mapping={
                    "vec": np.asarray(self.new_vec, np.float32).tobytes(),
                    "vec_key": vec_key(slate.hist_version, time.time()),
                })  # fmt: skip
                pipe.expire(key, rs.USER_TTL_S)
            await pipe.execute()

        # Postgres first: a page that isn't recorded must not be marked served in Redis.
        try:
            await self._q(self.s.api_persist_timeout_s, insert)
        except Exception as exc:
            log.warning("feed_persist_failed", error=type(exc).__name__)
            raise unavailable("could not record the page") from None
        if write_redis and self.redis_ok:
            try:
                await within(self.s.api_persist_timeout_s, to_redis())
            except Exception as exc:  # the next page falls back to served_slates
                self.degrade("redis", exc)

    # -- the request --

    async def run(
        self, cursor: Cursor | None, limit: int
    ) -> tuple[list[tuple[SlateItem, Article]], str | None]:
        """One page: its items with their rows, and the next cursor (None at the end)."""
        now = time.time()
        page_ttl = self.s.api_page_cache_s
        slate: Slate | None = None
        async with self.t.stage("state"):
            if cursor is not None:
                st, slate = await self.light_state(cursor)
            else:
                st = await self.full_state()
        offset = cursor.offset if cursor is not None else 0
        if slate is not None:
            cached = slate.cached_page(offset, now, page_ttl)
            if cached is not None:  # a repeat within 60 s: the same page again
                async with self.t.stage("hydrate"):
                    rows = await self.hydrate([uuid.UUID(i.id) for i in cached])
                slate.page += 1
                async with self.t.stage("persist"):  # only the row events join on
                    await self.persist(slate, cached, write_redis=False)
                return self._page(cached, rows), self._next(slate, offset, cached)
        stale = (
            slate is None
            or slate.hist_version != st.profile.hist_version
            or slate.hidden_sig != st.hidden_sig()
            or len(slate.remaining()) < limit
        )
        if stale:
            if not st.full:
                async with self.t.stage("state"):
                    st = await self.full_state(st)
            items = await self.rank_slate(st, slate.served if slate is not None else [])
            if slate is None:
                slate = Slate(new_slate_id(), st.profile.hist_version, st.hidden_sig(), items)
            else:
                slate.rerank(items, st.profile.hist_version, st.hidden_sig())
        assert slate is not None  # noqa: S101 (narrowing: set by either branch above)
        page = slate.remaining()[:limit]
        async with self.t.stage("hydrate"):
            rows = await self.hydrate([uuid.UUID(i.id) for i in page])
        page = [i for i in page if uuid.UUID(i.id) in rows]  # deleted by retention meanwhile
        if not page:
            return [], None
        slate.serve(offset, page, now, page_ttl)
        async with self.t.stage("persist"):
            await self.persist(slate, page, write_redis=True)
        return self._page(page, rows), self._next(slate, offset, page)

    @staticmethod
    def _page(
        items: Sequence[SlateItem], rows: dict[uuid.UUID, Article]
    ) -> list[tuple[SlateItem, Article]]:
        return [(i, rows[uuid.UUID(i.id)]) for i in items if uuid.UUID(i.id) in rows]

    @staticmethod
    def _next(slate: Slate, offset: int, page: Sequence[SlateItem]) -> str | None:
        return Cursor(slate.id, offset + len(page)).encode() if page else None
