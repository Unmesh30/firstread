"""The one versioned ranking config (``rank-<n>.yaml``, package data; ADR 0014).

A test pins each version's SHA-256: changing any constant means a new version, and
``model_version`` in every API response names both the model release and it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from firstread_core.ranking.features import NORMS, Norm

HERE = Path(__file__).resolve().parent
DEFAULT_VERSION = "rank-1"


@dataclass(frozen=True)
class SegmentWeights:
    """Separate blend weights for readers with at most ``max_history`` clicks."""

    max_history: int
    w_fresh: float
    w_trend: float


@dataclass(frozen=True)
class RankingConfig:
    version: str
    model_norm: Norm
    w_fresh: float
    w_trend: float
    half_life_hours: float
    trend_half_life_hours: float
    mmr_lambda: float
    story_cap: int
    cold_start_k: float
    cold_start_k_source: str  # "tuned" or "judgment" (ADR 0014 step 3)
    rerank_size: int
    slate_size: int
    page_size: int
    cold_segment: SegmentWeights | None = None

    def weights_for(self, n_history: int) -> tuple[float, float]:
        """(w_fresh, w_trend) for a reader with ``n_history`` clicks."""
        seg = self.cold_segment
        if seg is not None and n_history <= seg.max_history:
            return seg.w_fresh, seg.w_trend
        return self.w_fresh, self.w_trend

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RankingConfig:
        seg = d.get("cold_segment")
        cfg = cls(
            **{k: v for k, v in d.items() if k != "cold_segment"},
            cold_segment=SegmentWeights(**seg) if seg else None,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.model_norm not in NORMS:
            raise ValueError(f"unknown model_norm {self.model_norm!r}")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if min(self.w_fresh, self.w_trend, self.cold_start_k) < 0:
            raise ValueError("weights must be non-negative")
        if self.half_life_hours <= 0 or self.trend_half_life_hours <= 0:
            raise ValueError("half-lives must be positive")
        if not 0 < self.page_size <= self.slate_size <= self.rerank_size:
            raise ValueError("need 0 < page_size <= slate_size <= rerank_size")
        if self.story_cap < 1:
            raise ValueError("story_cap must be >= 1")
        if self.cold_start_k_source not in ("tuned", "judgment"):
            raise ValueError("cold_start_k_source must be 'tuned' or 'judgment'")


def config_path(version: str = DEFAULT_VERSION, directory: Path = HERE) -> Path:
    return directory / f"{version}.yaml"


@lru_cache
def load_config(version: str = DEFAULT_VERSION, directory: Path = HERE) -> RankingConfig:
    data = yaml.safe_load(config_path(version, directory).read_text())
    if not isinstance(data, dict) or data.get("version") != version:
        raise ValueError(f"{version}.yaml must be a mapping with version: {version}")
    return RankingConfig.from_dict(data)


def model_version(model_tag: str, cfg: RankingConfig) -> str:
    """What every response carries: the model release and the ranking config."""
    return f"{model_tag}+{cfg.version}"
