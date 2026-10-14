"""Training configuration (YAML -> validated pydantic models)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    heads: int = Field(default=8, ge=1)
    query_dim: int = Field(default=200, ge=1)  # additive-attention hidden size
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    attn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    init_scale: float = Field(default=10.0, ge=1.0, le=100.0)  # initial score temperature


class TrainConfig(_Strict):
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=256, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    negatives: int = Field(default=4, ge=1)  # K negatives per positive, same impression
    grad_clip: float = Field(default=1.0, gt=0)
    device: Literal["auto", "cpu", "mps"] = "auto"


class NRMSLiteConfig(_Strict):
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: Path) -> NRMSLiteConfig:
    return NRMSLiteConfig.model_validate(yaml.safe_load(path.read_text()) or {})
