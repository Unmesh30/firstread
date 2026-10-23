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
