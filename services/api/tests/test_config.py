from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from firstread_api.config import Settings
from firstread_api.testing import make_settings

PROD = {
    "app_env": "prod",
    "auth_jwks_url": "https://ref.supabase.co/auth/v1/.well-known/jwks.json",
    "auth_issuer": "https://ref.supabase.co/auth/v1",
    "metrics_token": SecretStr("m"),
    "log_uid_salt": SecretStr("s"),
    "api_model_dir": Path("/app/models/v0.2.0"),
}


def make(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg, arg-type]


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/1")
    s = make()
    assert s.redis_url.get_secret_value() == "redis://example:6379/1"
    assert s.model_version == "v0.2.0+rank-3"


def test_secrets_are_masked_in_repr() -> None:
    s = make(database_url="postgresql://u:hunter2@h/db")
    assert "hunter2" not in repr(s)


def test_a_complete_prod_config_starts() -> None:
    assert make(**PROD).app_env == "prod"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"auth_dev_bypass": True}, "AUTH_DEV_BYPASS"),
        ({"auth_local_jwks_file": Path("jwks.json")}, "AUTH_LOCAL_JWKS_FILE"),
        ({"auth_jwks_url": "http://ref.supabase.co/jwks"}, "https"),
        ({"auth_issuer": ""}, "AUTH_ISSUER"),
        ({"metrics_token": None}, "METRICS_TOKEN"),
        ({"log_uid_salt": SecretStr("local-dev-salt-not-secret")}, "LOG_UID_SALT"),
        ({"api_model_dir": None}, "API_MODEL_DIR"),
        ({"api_load_model": False}, "API_LOAD_MODEL"),
    ],
)
def test_prod_refuses_local_only_or_missing_settings(override: dict, message: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValidationError, match=message):
        make(**(PROD | override))


@pytest.mark.parametrize("algs", ["HS256", "none", "ES256,HS256", ""])
def test_only_asymmetric_algorithms_are_allowed(algs: str) -> None:
    with pytest.raises(ValidationError, match="AUTH_ALGORITHMS"):
        make(auth_algorithms=algs)


def test_jwks_ttl_is_capped_at_ten_minutes() -> None:
    with pytest.raises(ValidationError, match="TTL"):
        make(auth_jwks_ttl_s=601)


def test_every_setting_is_documented_in_env_example() -> None:
    example = (Path(__file__).resolve().parents[3] / ".env.example").read_text()
    names = {line.split("=", 1)[0] for line in example.splitlines() if "=" in line
             and not line.startswith("#")}  # fmt: skip
    missing = sorted(f.upper() for f in Settings.model_fields if f.upper() not in names)
    assert not missing, f"document these in .env.example: {missing}"


@pytest.mark.parametrize(
    "override",
    [{"redis_socket_timeout_s": 5}, {"redis_connect_timeout_s": 1}, {"redis_retries": 10}],
)
def test_the_redis_client_cannot_be_configured_to_hang(override: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="less than or equal"):
        make_settings(**override)


def test_the_redis_client_uses_the_fail_fast_settings() -> None:
    from firstread_api.main import redis_client

    client = redis_client(make_settings())
    kw = client.connection_pool.connection_kwargs
    assert kw["socket_timeout"] == 0.5
    assert kw["socket_connect_timeout"] == 0.5
    assert kw["retry"].get_retries() == 1
