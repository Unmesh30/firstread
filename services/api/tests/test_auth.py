"""Supabase JWT verification (ADR 0015): every rejection path, JWKS caching rules."""

import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest

from firstread_api.auth import JWKSCache, Verifier
from firstread_api.errors import ApiError
from firstread_api.testing import ISSUER, Signer, make_settings

UID = uuid.UUID("11111111-2222-4333-8444-555555555555")


def verifier(signer: Signer, tmp_path: Path, **kw: Any) -> Verifier:
    s = make_settings(auth_local_jwks_file=signer.write_jwks(tmp_path / "jwks.json"), **kw)
    return Verifier(s, JWKSCache(s))


async def test_a_valid_anonymous_token(signer: Signer, tmp_path: Path) -> None:
    p = await verifier(signer, tmp_path).verify(signer.token(UID))
    assert p.user_id == UID
    assert p.is_anonymous


async def rejected(v: Verifier, token: str) -> ApiError:
    with pytest.raises(ApiError) as e:
        await v.verify(token)
    return e.value


@pytest.mark.parametrize(
    "claims",
    [
        {"exp_in": -120},  # expired beyond the 30 s leeway
        {"iss": "https://evil.example/auth/v1"},
        {"aud": "anon"},
        {"role": "anon"},
        {"role": "service_role"},
        {"sub": "not-a-uuid"},
    ],
)
async def test_bad_claims_are_rejected(signer: Signer, tmp_path: Path, claims: dict) -> None:  # type: ignore[type-arg]
    sub = claims.pop("sub", UID)
    err = await rejected(verifier(signer, tmp_path), signer.token(sub, **claims))
    assert err.status == 401


async def test_missing_claims_are_rejected(signer: Signer, tmp_path: Path) -> None:
    token = jwt.encode({"sub": str(UID), "aud": "authenticated", "role": "authenticated"},
                       signer.private, algorithm="ES256", headers={"kid": signer.kid})  # fmt: skip
    assert (await rejected(verifier(signer, tmp_path), token)).status == 401


async def test_alg_confusion_hs256_with_the_public_key_is_rejected(
    signer: Signer, tmp_path: Path
) -> None:
    # The classic attack: sign HS256 with the *public* key as the HMAC secret. PyJWT won't
    # even encode that, so build it by hand.
    import hashlib
    import hmac

    from cryptography.hazmat.primitives import serialization

    pem = signer.private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    body = f"{b64({'alg': 'HS256', 'typ': 'JWT', 'kid': signer.kid})}.{b64(claims_for(UID))}"
    sig = base64.urlsafe_b64encode(hmac.new(pem, body.encode(), hashlib.sha256).digest())
    forged = f"{body}.{sig.rstrip(b'=').decode()}"
    assert (await rejected(verifier(signer, tmp_path), forged)).status == 401


def claims_for(uid: uuid.UUID) -> dict:  # type: ignore[type-arg]
    import time

    now = int(time.time())
    return {
        "sub": str(uid), "aud": "authenticated", "role": "authenticated",
        "iss": ISSUER, "iat": now, "exp": now + 600,
    }  # fmt: skip


def b64(d: dict) -> str:  # type: ignore[type-arg]
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


async def test_alg_none_is_rejected(signer: Signer, tmp_path: Path) -> None:
    token = f"{b64({'alg': 'none', 'kid': signer.kid})}.{b64({'sub': str(UID)})}."
    assert (await rejected(verifier(signer, tmp_path), token)).status == 401


async def test_a_jku_header_is_ignored(signer: Signer, tmp_path: Path) -> None:
    attacker = Signer(kid=signer.kid)  # same kid, different key, "hosted" elsewhere
    token = attacker.token(UID, headers={"jku": "file:///etc/passwd"})
    assert (await rejected(verifier(signer, tmp_path), token)).status == 401
    good = signer.token(UID, headers={"jku": "https://evil.example/jwks.json"})
    assert (await verifier(signer, tmp_path).verify(good)).user_id == UID


class Counting(httpx.AsyncBaseTransport):
    def __init__(self, jwks: dict) -> None:  # type: ignore[type-arg]
        self.jwks, self.calls, self.fail = jwks, 0, False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail:
            return httpx.Response(503)
        return httpx.Response(200, json=self.jwks)


def http_verifier(transport: Counting, **kw: Any) -> tuple[Verifier, JWKSCache]:
    s = make_settings(auth_jwks_url="https://test-project.supabase.co/auth/v1/.well-known/jwks.json",
                      **kw)  # fmt: skip
    cache = JWKSCache(s, httpx.AsyncClient(transport=transport))
    return Verifier(s, cache), cache


async def test_unknown_kid_refetches_once_per_interval(signer: Signer) -> None:
    t = Counting({"keys": [signer.jwk()]})
    v, _ = http_verifier(t)
    await v.verify(signer.token(UID))
    assert t.calls == 1
    stranger = Signer(kid="rotated-key")
    assert (await rejected(v, stranger.token(UID))).status == 401
    assert t.calls == 2  # one refetch for the unknown kid
    assert (await rejected(v, stranger.token(UID))).status == 401
    assert t.calls == 2  # not again within AUTH_JWKS_MIN_REFRESH_S
    await v.verify(signer.token(UID))
    assert t.calls == 2  # cached


async def test_rotation_is_picked_up_after_the_interval(signer: Signer) -> None:
    new = Signer(kid="new-key")
    t = Counting({"keys": [signer.jwk()]})
    v, _ = http_verifier(t, auth_jwks_min_refresh_s=0)
    await v.verify(signer.token(UID))
    t.jwks = {"keys": [signer.jwk(), new.jwk()]}
    assert (await v.verify(new.token(UID))).user_id == UID
    assert t.calls == 2


async def test_ttl_expiry_refetches_and_an_unreachable_jwks_is_a_503(signer: Signer) -> None:
    t = Counting({"keys": [signer.jwk()]})
    v, cache = http_verifier(t)
    await v.verify(signer.token(UID))
    cache.fetched_at -= 601  # past the 10-minute cap
    t.fail = True
    err = await rejected(v, signer.token(UID))
    assert err.status == 503
    assert err.code == "auth_unavailable"


async def test_keys_with_disallowed_algorithms_are_dropped(signer: Signer) -> None:
    hs = {"kty": "oct", "kid": signer.kid, "alg": "HS256", "k": "c2VjcmV0"}
    v, _ = http_verifier(Counting({"keys": [hs]}))
    assert (await rejected(v, signer.token(UID))).status == 401


async def test_a_failed_fetch_backs_off_instead_of_queueing(signer: Signer) -> None:
    t = Counting({"keys": [signer.jwk()]})
    v, cache = http_verifier(t)
    await v.verify(signer.token(UID))
    cache.fetched_at -= 601  # cache expired
    t.fail = True
    assert (await rejected(v, signer.token(UID))).status == 503
    assert t.calls == 2  # one failed fetch
    results = await asyncio.gather(*(v.verify(signer.token(UID)) for _ in range(20)),
                                   return_exceptions=True)  # fmt: skip
    assert all(isinstance(r, ApiError) and r.status == 503 for r in results)
    assert t.calls == 2  # no fetch during the backoff: 503 at once
    cache.failed_at -= 11  # backoff over; JWKS reachable again
    t.fail = False
    assert (await v.verify(signer.token(UID))).user_id == UID
    assert t.calls == 3
