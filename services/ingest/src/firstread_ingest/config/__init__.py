"""Settings (environment) and the curated config files (feeds, junk titles)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from firstread_core.topics import TOPIC_IDS

CONFIG_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Ingestion configuration. Every field is documented in `.env.example`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix=""
    )

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: SecretStr = SecretStr("postgresql://firstread:firstread@localhost:5432/firstread")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    # Model release to load (ADR 0005/0006); pinned in models.py by manifest SHA256.
    ingest_model_tag: str = "v0.2.0"
    # Optional local folder holding that release (e.g. data/release/v0.2.0 before it is
    # uploaded). It is verified against the same pin as a Hub download.
    ingest_model_dir: Path | None = None
    # RSS: concurrent feeds overall (each host is fetched one at a time).
    ingest_rss_concurrency: int = 8
    # Dedupe / clustering (ADR 0009): fallback thresholds, plus validated per-pair-type
    # values in dedupe.VALIDATED_BY_PAIR (on unless INGEST_PER_PAIR_THRESHOLDS=false).
    ingest_tau_dup: float = 0.975
    ingest_tau_story: float = 0.925
    ingest_per_pair_thresholds: bool = True
    ingest_dedupe_window_hours: int = 48
    ingest_fingerprint_window_hours: int = 72
    # Titles with fewer words are dropped.
    ingest_min_title_words: int = 3
    # Retention (ADR 0008): articles 14 d; story coverage 48 h (trending uses 24 h).
    ingest_retention_article_days: int = 14
    ingest_retention_coverage_hours: int = 48
    ingest_retention_runs_days: int = 30
    ingest_size_warn_mb: int = 350
    # User data (ADR 0016). The size guard can only shorten these (store.guard).
    ingest_retention_impressions_days: float = 7
    ingest_retention_events_days: float = 30
    ingest_retention_slates_days: float = 30
    ingest_retention_inactive_anonymous_days: float = 30
    # Runner-side copy of the 72 h window (ADR 0012). GitHub Actions caches this file
    # between runs; it's only a cache (any mismatch -> full reload). Empty: no file,
    # the window is reloaded from Postgres every run (~14 MB of egress).
    ingest_window_file: Path | None = None
    # GDELT GKG files (ADR 0007).
    ingest_gkg_max_files: int = 8  # catch up at most 2 h of 15-minute files per run
    ingest_gkg_stale_minutes: int = 60  # newest file older than this -> "stale" in the summary
    ingest_gkg_max_embed: int = 2000  # newest GKG candidates embedded per run (CPU bound)
    ingest_gkg_max_per_run: int = 50  # GKG articles inserted per run (DB budget, ~3.2 KB each)
    ingest_gkg_max_per_domain: int = 3  # per outlet per run
    ingest_gkg_max_per_story: int = 3  # stored articles per story (feed shows <= 2)
    ingest_gkg_topic_min: int = 2  # slots reserved per topic before ranking fills the rest

    @field_validator("ingest_model_dir", "ingest_window_file", mode="before")
    @classmethod
    def _empty_is_unset(cls, v: object) -> object:
        return None if v in ("", None) else v


class Feed(BaseModel):
    id: str
    url: str
    source_name: str
    domain: str
    region: str
    geo: Literal["us", "intl"]
    topic_hint: str | None = None
    enabled: bool = True
    terms_checked: bool = False
    terms: Literal[
        "website_with_attribution", "personal_noncommercial", "forbidden", "no_explicit_permission"
    ] = "no_explicit_permission"
    rss_page_url: str | None = None
    terms_url: str | None = None
    # False when the publisher's permission covers text but not images.
    allow_images: bool = True

    @model_validator(mode="after")
    def _enabled_needs_explicit_permission(self) -> Feed:
        if self.enabled and self.terms != "website_with_attribution":
            raise ValueError(
                f"{self.id}: only feeds with explicit website permission may be enabled"
            )
        return self

    @field_validator("topic_hint")
    @classmethod
    def _known_topic(cls, v: str | None) -> str | None:
        if v is not None and v not in TOPIC_IDS:
            raise ValueError(f"unknown topic hint {v!r}")
        return v


class JunkTitles(BaseModel):
    whole: frozenset[str]
    media_prefixes: tuple[str, ...]
    sections: frozenset[str] = frozenset()
    outlet_words: frozenset[str] = frozenset()


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"{path}: expected a mapping with version: 1")
    return data


@lru_cache
def load_announcement_patterns(
    path: Path = CONFIG_DIR / "announcement_titles.yaml",
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(name, compiled regex) for corporate-announcement titles, in file order."""
    return tuple(
        (name, re.compile(pattern, re.IGNORECASE))
        for name, pattern in _yaml(path)["patterns"].items()
    )


def load_feeds(path: Path = CONFIG_DIR / "feeds.yaml", enabled_only: bool = True) -> list[Feed]:
    feeds = [Feed(**f) for f in _yaml(path)["feeds"]]
    ids = [f.id for f in feeds]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate feed ids")
    return [f for f in feeds if f.enabled or not enabled_only]


@lru_cache
def load_junk_titles(path: Path = CONFIG_DIR / "junk_titles.yaml") -> JunkTitles:
    data = _yaml(path)
    return JunkTitles(
        whole=frozenset(data["whole"]),
        media_prefixes=tuple(p.casefold() for p in data["media_prefixes"]),
        sections=frozenset(data.get("sections") or []),
        outlet_words=frozenset(data.get("outlet_words") or []),
    )


class OutletNetworks(BaseModel):
    """Syndication networks for trending breadth (ADR 0013).

    ``apply`` is set by the pre-registered materiality rule: when false, the list is
    kept for the record but trending counts every domain as its own outlet.
    """

    apply: bool
    networks: dict[str, list[str]]

    def mapping(self) -> dict[str, str]:
        """domain -> network id, only when applied."""
        if not self.apply:
            return {}
        return {d: name for name, domains in self.networks.items() for d in domains}

    @model_validator(mode="after")
    def _one_network_per_domain(self) -> OutletNetworks:
        seen: dict[str, str] = {}
        for name, domains in self.networks.items():
            for d in domains:
                if d in seen:
                    raise ValueError(f"{d} is in both {seen[d]} and {name}")
                seen[d] = name
        return self


@lru_cache
def load_outlet_networks(path: Path = CONFIG_DIR / "outlet_networks.yaml") -> OutletNetworks:
    if not path.is_file():
        return OutletNetworks(apply=False, networks={})
    data = _yaml(path)
    return OutletNetworks(apply=bool(data["apply"]), networks=data.get("networks") or {})
