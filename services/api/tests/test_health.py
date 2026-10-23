import asyncio

import pytest
from fastapi.testclient import TestClient

from firstread_api.health import Check
from firstread_api.routes.ops import get_readiness_checks


async def _ok() -> None:
    return None


async def _fail() -> None:
    raise ConnectionRefusedError("connection to postgresql://user:secret@db refused")


async def _hang() -> None:
    await asyncio.sleep(10)


def _override(client: TestClient, checks: dict[str, Check]) -> None:
    client.app.dependency_overrides[get_readiness_checks] = lambda: checks  # type: ignore[attr-defined]


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_all_ok(client: TestClient) -> None:
    _override(client, {"postgres": _ok, "redis": _ok})
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ready", "checks": {"postgres": "ok", "redis": "ok"}, "model_tag": None
    }  # fmt: skip


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ({"postgres": _fail, "redis": _ok}, {"postgres": "error", "redis": "ok"}),
        ({"postgres": _ok, "redis": _fail}, {"postgres": "ok", "redis": "error"}),
        ({"postgres": _hang, "redis": _ok}, {"postgres": "timeout", "redis": "ok"}),
    ],
)
def test_readyz_failures_return_503(
    client: TestClient, checks: dict[str, Check], expected: dict[str, str]
) -> None:
    _override(client, checks)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready", "checks": expected, "model_tag": None}


def test_readyz_does_not_leak_error_details(client: TestClient) -> None:
    _override(client, {"postgres": _fail, "redis": _ok})
    assert "secret" not in client.get("/readyz").text


async def test_the_model_check_fails_when_the_encoder_is_missing() -> None:
    from firstread_api.health import model_check

    await model_check(True)()
    with pytest.raises(RuntimeError):
        await model_check(False)()
