"""HTTP standards without a database: errors, request ids, CORS, rate limits, metrics."""

import uuid

from fastapi.testclient import TestClient

from firstread_api.main import create_app
from firstread_api.testing import Signer, make_settings

UID = uuid.uuid4()


def test_topics_are_public_and_carry_the_envelope(client: TestClient) -> None:
    body = client.get("/v1/topics").json()
    assert len(body["topics"]) == 13
    assert body["topics"][0] == {"id": "us", "name": "U.S."}
    assert body["model_version"] == "v0.2.0+rank-2"
    assert body["request_id"]


def test_every_error_has_one_shape(client: TestClient) -> None:
    r = client.get("/v1/me")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"
    body = r.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["request_id"] == r.headers["X-Request-ID"]
    r = client.get("/v1/nothing-here")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_validation_errors_use_the_error_shape(client: TestClient, signer: Signer) -> None:
    headers = {"Authorization": f"Bearer {signer.token(UID)}"}
    r = client.post("/v1/events", json={"events": []}, headers=headers)
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"]
    too_many = [{"event_id": str(uuid.uuid4()), "type": "impression",
                 "article_id": str(uuid.uuid4())} for _ in range(101)]  # fmt: skip
    assert client.post("/v1/events", json={"events": too_many}, headers=headers).status_code == 422
    bad = [{"event_id": str(uuid.uuid4()), "type": "click"}]  # no article_id
    assert client.post("/v1/events", json={"events": bad}, headers=headers).status_code == 422
    for topics in (["sports", "health"], ["sports", "sports", "health"], ["sports", "x", "us"]):
        r = client.put("/v1/me/topics", json={"topics": topics}, headers=headers)
        assert r.status_code == 422, topics


def test_request_id_is_always_generated_by_the_server(client: TestClient) -> None:
    """A client can't choose request_id (it keys served slates in P3d): even a valid
    inbound X-Request-ID is only logged as client_request_id."""
    for inbound in ("req-12345678", "short", "<script>alert(1)</script>", None):
        headers = {"X-Request-ID": inbound} if inbound else {}
        r = client.get("/v1/topics", headers=headers)
        rid = r.headers["X-Request-ID"]
        assert rid != inbound
        assert len(rid) == 32
        assert r.json()["request_id"] == rid
    first = client.get("/v1/topics", headers={"X-Request-ID": "retry-12345678"}).json()
    second = client.get("/v1/topics", headers={"X-Request-ID": "retry-12345678"}).json()
    assert first["request_id"] != second["request_id"]  # a retry gets a new id


def test_only_a_valid_client_id_is_kept_for_tracing() -> None:
    from firstread_api.logging import client_request_id

    assert client_request_id("trace-abc-123") == "trace-abc-123"
    for bad in (None, "short", "x" * 65, "bad id", "<script>"):
        assert client_request_id(bad) is None


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    pre = {
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Authorization",
    }
    r = client.options("/v1/topics", headers={"Origin": "https://unmesh30.github.io", **pre})
    assert r.headers["access-control-allow-origin"] == "https://unmesh30.github.io"
    r = client.options("/v1/topics", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in r.headers


def test_ip_rate_limit_returns_429_with_retry_after(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(create_app(make_settings(rl_ip_per_min=2))) as c:
        assert c.get("/v1/topics").status_code == 200
        assert c.get("/v1/topics").status_code == 200
        r = c.get("/v1/topics")
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "rate_limited"
        assert int(r.headers["Retry-After"]) >= 1
        assert c.get("/healthz").status_code == 200  # probes are never limited


def test_dev_bypass_only_when_enabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(create_app(make_settings())) as c:
        assert c.get("/v1/me", headers={"X-Dev-User": str(UID)}).status_code == 401


def test_metrics_use_route_templates_and_need_a_token_outside_local(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pydantic import SecretStr

    with TestClient(create_app(make_settings(metrics_token=SecretStr("t0ken")))) as c:
        c.get("/v1/topics")
        assert c.get("/metrics").status_code == 401
        body = c.get("/metrics", headers={"Authorization": "Bearer t0ken"}).text
        assert 'route="/v1/topics"' in body
        assert "redis_commands_total" in body
