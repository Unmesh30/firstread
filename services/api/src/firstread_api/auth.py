"""Supabase JWT verification against the project's JWKS (ADR 0015).

* Keys come only from the configured ``AUTH_JWKS_URL`` (or, locally, a file), fetched
  with an async httpx client. Token headers (``jku``, ``x5u``, ``jwk``) are ignored,
  so a token can never choose where its key comes from (CVE-2026-48522 was
  PyJWKClient fetching any URI scheme; we don't use PyJWKClient at all).
* The JWKS is cached for at most ``AUTH_JWKS_TTL_S`` (≤ 10 min, Supabase's own edge
  cache). An unknown ``kid`` triggers at most one refetch per
  ``AUTH_JWKS_MIN_REFRESH_S``. Fetches are single-flight. There's no stale-if-error:
  past the TTL an unreachable JWKS is a 503, not a silently old key set.
* Algorithms are an allowlist (ES256, optionally RS256); HS256 and ``none`` are never
  accepted, and the key's type must match the algorithm.
* Claims: ``exp`` (with leeway), ``iat``, ``iss``, ``aud == authenticated``,
  ``role == authenticated``, ``sub`` a UUID. Anonymous users are role
  ``authenticated`` with ``is_anonymous: true``.
* Local and test only: ``X-Dev-User: <uuid>`` without a token (``AUTH_DEV_BYPASS``).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
import structlog
from fastapi import Request

from firstread_api.config import Settings
from firstread_api.errors import ApiError
from firstread_api.metrics import AUTH_FAILURES, JWKS_REFRESH

log = structlog.get_logger("firstread_api.auth")
DEV_HEADER = "X-Dev-User"
KTY_FOR_ALG = {"ES256": "EC", "RS256": "RSA"}


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    is_anonymous: bool


class AuthUnavailableError(Exception):
    """The JWKS can't be fetched and the cached copy has expired."""


class JWKSCache:
    """The project's signing keys, fetched lazily inside requests (Cloud Run gives no
    CPU outside requests, so there are no background refreshes)."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client
        self.keys: dict[str, jwt.PyJWK] = {}
        self.fetched_at = -1e18
        self.last_unknown_kid_fetch = -1e18  # rate-limits refetches for unseen kids
        self.lock = asyncio.Lock()

    def _fresh(self, now: float) -> bool:
        return now - self.fetched_at < self.settings.auth_jwks_ttl_s

    async def _load(self) -> dict[str, Any]:
        path = self.settings.auth_local_jwks_file
        if path is not None:
            data: dict[str, Any] = json.loads(path.read_text())
            return data
        url = self.settings.auth_jwks_url
        if not url:
            raise AuthUnavailableError("AUTH_JWKS_URL is not set")
        client = self.client or httpx.AsyncClient(timeout=self.settings.auth_http_timeout_s)
        try:
            resp = await client.get(url, follow_redirects=False)
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
            return body
        finally:
            if self.client is None:
                await client.aclose()

    async def _refresh(self) -> None:
        try:
            data = await self._load()
        except AuthUnavailableError:
            JWKS_REFRESH.labels("error").inc()
            raise
        except Exception as exc:
            JWKS_REFRESH.labels("error").inc()
            log.warning("jwks_fetch_failed", error=type(exc).__name__)
            raise AuthUnavailableError(type(exc).__name__) from exc
        keys: dict[str, jwt.PyJWK] = {}
        for k in data.get("keys", []):
            kid, alg = k.get("kid"), k.get("alg")
            if not kid or alg not in self.settings.algorithms:
                continue
            try:
                keys[kid] = jwt.PyJWK(k, algorithm=alg)
            except jwt.PyJWTError:
                continue
        self.keys = keys
        self.fetched_at = time.monotonic()
        JWKS_REFRESH.labels("ok").inc()

    async def get(self, kid: str) -> jwt.PyJWK | None:
        """The key for ``kid``; refetches when the cache expired, or once per interval
        for a kid we haven't seen (key rotation)."""
        now = time.monotonic()
        if self._fresh(now) and kid in self.keys:
            return self.keys[kid]
        async with self.lock:
            now = time.monotonic()
            if not self._fresh(now):
                await self._refresh()
            elif (
                kid not in self.keys
                and now - self.last_unknown_kid_fetch >= self.settings.auth_jwks_min_refresh_s
            ):
                self.last_unknown_kid_fetch = now
                await self._refresh()
            return self.keys.get(kid)


class Verifier:
    def __init__(self, settings: Settings, jwks: JWKSCache) -> None:
        self.settings = settings
        self.jwks = jwks

    def _reject(self, reason: str, message: str = "invalid token") -> ApiError:
        AUTH_FAILURES.labels(reason).inc()
        return ApiError(401, message, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    async def verify(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise self._reject("malformed") from None
        alg, kid = header.get("alg"), header.get("kid")
        if alg not in self.settings.algorithms:
            raise self._reject("algorithm")
        if not isinstance(kid, str) or not kid:
            raise self._reject("kid")
        try:
            key = await self.jwks.get(kid)
        except AuthUnavailableError:
            raise ApiError(503, "authentication is temporarily unavailable",
                           "auth_unavailable", headers={"Retry-After": "5"}) from None  # fmt: skip
        if key is None:
            raise self._reject("unknown_kid")
        if KTY_FOR_ALG.get(alg) != key.key_type:
            raise self._reject("key_type")
        try:
            claims = jwt.decode(
                token,
                key=key.key,
                algorithms=[alg],
                audience=self.settings.auth_audience,
                issuer=self.settings.auth_issuer,
                leeway=self.settings.auth_leeway_s,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except jwt.ExpiredSignatureError:
            raise self._reject("expired", "token expired") from None
        except jwt.PyJWTError:
            raise self._reject("claims") from None
        if claims.get("role") != "authenticated":
            raise self._reject("role")
        try:
            sub = uuid.UUID(str(claims["sub"]))
        except ValueError:
            raise self._reject("sub") from None
        return Principal(sub, bool(claims.get("is_anonymous", False)))


async def authenticate(request: Request, settings: Settings, verifier: Verifier) -> Principal:
    if settings.auth_dev_bypass and settings.app_env in ("local", "test"):
        dev = request.headers.get(DEV_HEADER)
        if dev:
            try:
                return Principal(uuid.UUID(dev), True)
            except ValueError:
                raise ApiError(401, f"{DEV_HEADER} must be a UUID") from None
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        AUTH_FAILURES.labels("missing").inc()
        raise ApiError(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    return await verifier.verify(token.strip())
