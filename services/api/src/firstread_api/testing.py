"""Test support for the API (imported only by tests and the local load test).

* :class:`Signer`: a locally generated P-256 key, its JWKS, and tokens with any claims.
* :func:`live_api`: the real app on a throwaway database (all migrations) and Redis
  db 14, with the lifespan running, plus raw handles to check side effects.
* :func:`insert_articles`: minimal article rows with unit vectors.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jwt
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec

from firstread_api.config import Settings
from firstread_core.db import apply_migrations, connect, register_vector_codec
from firstread_core.dbtest import MIGRATIONS, temp_database

ISSUER = "https://test-project.supabase.co/auth/v1"
REDIS_TEST_DB = 14


class Signer:
    """Signs tokens like Supabase Auth (ES256) with a key generated for the test run."""

    def __init__(self, kid: str = "test-key-1") -> None:
        self.kid = kid
        self.private = ec.generate_private_key(ec.SECP256R1())

    def jwk(self) -> dict[str, Any]:
        public = jwt.algorithms.ECAlgorithm.to_jwk(self.private.public_key(), as_dict=True)
        return {**public, "kid": self.kid, "alg": "ES256", "use": "sig"}

    def write_jwks(self, path: Path, extra: Sequence[dict[str, Any]] = ()) -> Path:
        path.write_text(json.dumps({"keys": [self.jwk(), *extra]}))
        return path

    def token(
        self,
        sub: uuid.UUID | str,
        *,
        anonymous: bool = True,
        exp_in: int = 3600,
        headers: dict[str, Any] | None = None,
        **claims: Any,
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": str(sub),
            "aud": "authenticated",
            "role": "authenticated",
            "iss": ISSUER,
            "iat": now,
            "exp": now + exp_in,
            "is_anonymous": anonymous,
            **claims,
        }
        return jwt.encode(payload, self.private, algorithm="ES256",
                          headers={"kid": self.kid, **(headers or {})})  # fmt: skip


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "app_env": "test",
        "auth_issuer": ISSUER,
        "readyz_timeout_s": 0.5,
        "api_load_model": False,  # feed tests turn it on (it loads the pinned encoder)
        "_env_file": None,
    }
    return Settings(**(base | overrides))


def redis_url(base: str | None = None) -> str:
    url = base or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return url.rsplit("/", 1)[0] + f"/{REDIS_TEST_DB}"


@dataclass
class LiveApi:
    client: httpx.AsyncClient
    app: Any
    dsn: str
    redis: Any  # raw redis.asyncio client on the test db
    signer: Signer

    def auth(self, uid: uuid.UUID, **kw: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.signer.token(uid, **kw)}"}

    async def db(self) -> Any:
        conn = await connect(self.dsn)
        await register_vector_codec(conn)
        return conn


@asynccontextmanager
async def live_api(tmp_path: Path, **overrides: Any) -> AsyncIterator[LiveApi]:
    from redis.asyncio import Redis

    from firstread_api.main import create_app

    signer = Signer()
    jwks = signer.write_jwks(tmp_path / "jwks.json")
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
        finally:
            await conn.close()
        from pydantic import SecretStr

        settings = make_settings(
            database_url=SecretStr(dsn),
            redis_url=SecretStr(redis_url()),
            auth_local_jwks_file=jwks,
            **overrides,
        )
        redis = Redis.from_url(redis_url())
        await redis.flushdb()
        app = create_app(settings)
        try:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                    yield LiveApi(c, app, dsn, redis, signer)
        finally:
            await redis.flushdb()
            await redis.aclose()


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=384).astype(np.float32)
    return v / np.linalg.norm(v)


async def insert_articles(
    conn: Any, n: int, *, topic: str = "sports", start: int = 0
) -> list[uuid.UUID]:
    """``n`` minimal stored articles (seeded vectors); returns their ids."""
    ids = [uuid.uuid4() for _ in range(n)]
    for k, aid in enumerate(ids):
        v = unit(start + k)
        await conn.execute(
            "insert into public.articles (id, url, url_canonical, title, source_name, "
            "source_domain, geo, category, category_confidence, category_source, first_seen_at, "
            "embedding, embedding_model, text_source, title_fingerprint, story_cluster_id, origin) "
            "values ($1, $2, $2, $3, 'Example', $4, 'us', $5, 0.9, 'model', now(), "
            "$6::text::extensions.halfvec(384), 'm', 'title', $7, $1, 'gdelt')",
            aid,
            f"https://e{start + k}.com/a",
            f"Headline {start + k}",
            f"e{(start + k) % 3}.com",
            topic,
            "[" + ",".join(f"{x:.6f}" for x in v) + "]",
            f"fp{start + k}-{aid}",
        )
    return ids


# ---- feed fixtures (ADR 0017) ----


@dataclass
class Seeded:
    id: uuid.UUID
    vector: np.ndarray
    cluster: uuid.UUID
    domain: str
    category: str
    first_seen: Any  # datetime
    text_source: str = "title"


def topic_vectors(topics: Sequence[str], n: int, seed: int = 0) -> dict[str, list[np.ndarray]]:
    """``n`` unit vectors per topic around a random topic direction (pairwise cos ≈ 0.4,
    well below every story threshold, so no two of them group or count as read)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {}
    for t in topics:
        d = rng.normal(size=384)
        d /= np.linalg.norm(d)
        vs = []
        for _ in range(n):
            e = rng.normal(size=384)
            e /= np.linalg.norm(e)
            v = 0.6 * d + 0.8 * e
            vs.append((v / np.linalg.norm(v)).astype(np.float32))
        out[t] = vs
    return out


async def insert_article(
    conn: Any,
    vector: np.ndarray,
    *,
    category: str = "sports",
    domain: str = "example.com",
    cluster: uuid.UUID | None = None,
    age_hours: float = 1.0,
    text_source: str = "title",
    title: str | None = None,
) -> Seeded:
    from datetime import UTC, datetime, timedelta

    aid = uuid.uuid4()
    cluster = cluster or aid
    first_seen = datetime.now(UTC) - timedelta(hours=age_hours)
    await conn.execute(
        "insert into public.articles (id, url, url_canonical, title, source_name, "
        "source_domain, geo, category, category_confidence, category_source, first_seen_at, "
        "embedding, embedding_model, text_source, title_fingerprint, story_cluster_id, origin) "
        "values ($1, $2, $2, $3, $4, $4, 'us', $5, 0.9, 'model', $6, "
        "$7::text::extensions.halfvec(384), 'm', $8, $9, $10, 'gdelt')",
        aid, f"https://{domain}/{aid}", title or f"Headline {aid.hex[:8]}", domain, category,
        first_seen, "[" + ",".join(f"{x:.6f}" for x in vector) + "]", text_source,
        f"fp-{aid}", cluster,
    )  # fmt: skip
    return Seeded(aid, vector, cluster, domain, category, first_seen, text_source)


async def publish_pools(
    redis: Any,
    articles: Sequence[Seeded],
    trending: Sequence[tuple[Seeded, float, int]] = (),
    version: str = "test-1",
) -> Any:
    """Publish a pool snapshot built from ``articles`` the way ingestion does (blob, then
    ``pools:ver``); ``trending`` = (representative, score, outlets)."""
    from datetime import UTC, datetime

    from firstread_core import pools

    newest = sorted(articles, key=lambda a: a.first_seen, reverse=True)
    topics: dict[str, list[uuid.UUID]] = {}
    for a in newest:
        topics.setdefault(a.category, []).append(a.id)
    snap = pools.PoolSnapshot(
        version=version,
        built_at=datetime.now(UTC),
        fresh=[a.id for a in newest],
        topics=topics,
        trending=[pools.TrendingEntry(a.id, s, n, (a.cluster,)) for a, s, n in trending],
        articles={
            a.id: pools.ArticleInfo(
                a.cluster, a.category, int(a.first_seen.timestamp()), a.text_source
            )
            for a in articles
        },
        centroids=pools.topic_centroids(
            np.stack([a.vector for a in articles]), [a.category for a in articles], min_rows=1
        ),
    )
    await redis.set(pools.blob_key(version), pools.encode(snap))
    await redis.set(pools.VER_KEY, version)
    return snap
