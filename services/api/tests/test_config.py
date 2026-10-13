import pytest

from firstread_api.config import Settings


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/1")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.app_env == "prod"
    assert s.redis_url.get_secret_value() == "redis://example:6379/1"


def test_secrets_are_masked_in_repr() -> None:
    s = Settings(_env_file=None, database_url="postgresql://u:hunter2@h/db")  # type: ignore[call-arg, arg-type]
    assert "hunter2" not in repr(s)
