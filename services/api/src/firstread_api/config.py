"""Typed settings loaded from environment variables (and `.env` locally)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API configuration. Every field is documented in `.env.example`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Secrets: SecretStr keeps them out of reprs and logs.
    database_url: SecretStr = SecretStr("postgresql://firstread:firstread@localhost:5432/firstread")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    readyz_timeout_s: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
