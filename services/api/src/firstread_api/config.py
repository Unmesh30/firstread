"""Typed settings loaded from environment variables (and `.env` locally).

Production refuses to start with anything that only makes sense locally: the auth
dev bypass, a local JWKS file, a non-https JWKS URL, or a missing metrics token or
log salt (ADR 0015).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from firstread_core.ranking.config import DEFAULT_VERSION as RANKING_VERSION

ALLOWED_ALGORITHMS = frozenset({"ES256", "RS256"})  # never HS256 or "none"
MAX_JWKS_TTL_S = 600  # Supabase's edge caches the JWKS for 10 minutes
DEV_SALT = "local-dev-salt-not-secret"


class Settings(BaseSettings):
    """API configuration. Every field is documented in `.env.example`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Secrets: SecretStr keeps them out of reprs and logs.
    database_url: SecretStr = SecretStr("postgresql://firstread:firstread@localhost:5432/firstread")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    readyz_timeout_s: float = 2.0
    db_pool_max: int = 5

    # What every response's model_version names (ADR 0014): model release + ranking.
    api_model_tag: str = "v0.2.0"
    api_ranking_version: str = RANKING_VERSION

    # ---- auth (ADR 0015) ----
    auth_jwks_url: str | None = None  # https://<ref>.supabase.co/auth/v1/.well-known/jwks.json
    auth_issuer: str | None = None  # https://<ref>.supabase.co/auth/v1
    auth_audience: str = "authenticated"
    auth_algorithms: str = "ES256"  # comma-separated; RS256 only if added on purpose
    auth_jwks_ttl_s: int = MAX_JWKS_TTL_S
    auth_jwks_min_refresh_s: int = 60  # at most one refetch per interval on an unknown kid
    auth_leeway_s: int = 30
    auth_http_timeout_s: float = 2.0
    # Local and test only: accept `X-Dev-User: <uuid>` without a token.
    auth_dev_bypass: bool = False
    # Local and test only: read the JWKS from a file (the load test signs its own tokens).
    auth_local_jwks_file: Path | None = None

    # ---- HTTP ----
    api_cors_origins: str = (
        "https://unmesh30.github.io,http://localhost:8081,http://localhost:19006"
    )
    # In-process token buckets per instance (ADR 0015): Redis-backed limits would cost a
    # command per request, and commands are the binding budget (ADR 0011).
    rl_user_feed_per_min: int = 60
    rl_user_events_per_min: int = 120
    rl_user_writes_per_min: int = 10
    rl_ip_per_min: int = 300
    rl_max_keys: int = 10_000

    # ---- observability ----
    metrics_token: SecretStr | None = None  # required outside local
    log_uid_salt: SecretStr = SecretStr(DEV_SALT)  # HMAC key for hashed user ids in logs

    @field_validator("auth_local_jwks_file", "auth_jwks_url", "auth_issuer", mode="before")
    @classmethod
    def _empty_is_unset(cls, v: object) -> object:
        return None if v in ("", None) else v

    @model_validator(mode="after")
    def _safe(self) -> "Settings":
        algs = set(self.algorithms)
        if not algs or not algs <= ALLOWED_ALGORITHMS:
            raise ValueError(f"AUTH_ALGORITHMS must be a subset of {sorted(ALLOWED_ALGORITHMS)}")
        if not 0 < self.auth_jwks_ttl_s <= MAX_JWKS_TTL_S:
            raise ValueError(f"AUTH_JWKS_TTL_S must be in (0, {MAX_JWKS_TTL_S}]")
        local_only = self.app_env in ("local", "test")
        if not local_only:
            if self.auth_dev_bypass:
                raise ValueError("AUTH_DEV_BYPASS is only allowed when APP_ENV is local or test")
            if self.auth_local_jwks_file is not None:
                raise ValueError(
                    "AUTH_LOCAL_JWKS_FILE is only allowed when APP_ENV is local or test"
                )
            if not (self.auth_jwks_url or "").startswith("https://"):
                raise ValueError("AUTH_JWKS_URL must be an https URL in prod")
            if not self.auth_issuer:
                raise ValueError("AUTH_ISSUER is required in prod")
            if self.metrics_token is None or not self.metrics_token.get_secret_value():
                raise ValueError("METRICS_TOKEN is required in prod")
            if self.log_uid_salt.get_secret_value() == DEV_SALT:
                raise ValueError("LOG_UID_SALT must be set to a secret in prod")
        return self

    @property
    def algorithms(self) -> list[str]:
        return [a.strip() for a in self.auth_algorithms.split(",") if a.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    @property
    def model_version(self) -> str:
        return f"{self.api_model_tag}+{self.api_ranking_version}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
