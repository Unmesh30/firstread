"""NRMS-lite as an eval-harness ``Scorer``: ``fit`` trains, ``score`` ranks.

Trained weights are cached in ``<model_dir>/nrms_lite/<key[:16]>/`` with a
``key.json`` (config + embedding cache id + training-data fingerprint + code
version), so re-running an eval reuses a checkpoint only if nothing changed.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from firstread_ml import __version__, tracking
from firstread_ml.baselines import FitContext, score_by_user_vectors
from firstread_ml.config import NRMSLiteConfig, load_config
from firstread_ml.mind.data import Impressions
from firstread_ml.models.nrms_lite import NRMSLite
from firstread_ml.models.train import resolve_device, train_nrms_lite, user_vectors
from firstread_ml.paths import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "ml" / "configs" / "nrms_lite.yaml"


def _fingerprint(train: Impressions) -> str:
    h = hashlib.sha256()
    for arr in (train.imp_id, train.time, train.cand, train.label, train.hist):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def checkpoint_key(cfg: NRMSLiteConfig, ctx: FitContext) -> dict[str, Any]:
    return {
        "config": cfg.model_dump(),
        "embedding_cache_id": ctx.embedding_id,
        "train_fingerprint": _fingerprint(ctx.train),
        "firstread_ml": __version__,
    }


def save_checkpoint(model: NRMSLite, cfg: NRMSLiteConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path / "model.pt")
    (path / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))


def load_checkpoint(path: Path, device: torch.device) -> tuple[NRMSLite, NRMSLiteConfig]:
    cfg = NRMSLiteConfig.model_validate(yaml.safe_load((path / "config.yaml").read_text()))
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model = NRMSLite(cfg.model, state["population"])
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


class NRMSLiteScorer:
    name = "nrms_lite"
    needs_embeddings = True

    def __init__(
        self, config: NRMSLiteConfig | Path = DEFAULT_CONFIG, seed: int | None = None
    ) -> None:
        cfg = config if isinstance(config, NRMSLiteConfig) else load_config(config)
        self.cfg = cfg if seed is None else cfg.model_copy(update={"seed": seed})
        self.name = f"nrms_lite_s{self.cfg.seed}"
        self.device = resolve_device(self.cfg.train.device)
        self.history: list[dict[str, Any]] = []
        self.checkpoint: Path | None = None

    def fit(self, ctx: FitContext) -> None:
        if ctx.emb is None:
            raise ValueError("nrms_lite needs embeddings")
        self.table = torch.from_numpy(ctx.emb).to(self.device)
        key = checkpoint_key(self.cfg, ctx)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        ckpt = ctx.model_dir / "nrms_lite" / digest[:16] if ctx.model_dir else None
        if ckpt and self._matches(ckpt, key):
            self.model, _ = load_checkpoint(ckpt, self.device)
            self.history = json.loads((ckpt / "history.json").read_text())
            self.checkpoint = ckpt
            print(f"  nrms_lite: reusing checkpoint {ckpt}")
            return
        with tracking.run(f"nrms_lite-{ctx.tag}") as run:
            run.params(
                {"tag": ctx.tag, "seed": self.cfg.seed}
                | {f"model.{k}": v for k, v in self.cfg.model.model_dump().items()}
                | {f"train.{k}": v for k, v in self.cfg.train.model_dump().items()}
            )
            self.model, self.history = train_nrms_lite(
                self.cfg,
                ctx.train,
                ctx.emb,
                ctx.population_vector,
                log=lambda m, step: run.metrics(m, step),
            )
        if ckpt:
            save_checkpoint(self.model, self.cfg, ckpt)
            (ckpt / "history.json").write_text(json.dumps(self.history, indent=1) + "\n")
            (ckpt / "key.json").write_text(json.dumps(key, indent=1) + "\n")  # last
            self.checkpoint = ckpt

    @staticmethod
    def _matches(ckpt: Path, key: dict[str, Any]) -> bool:
        files = [ckpt / n for n in ("key.json", "model.pt", "history.json", "config.yaml")]
        if not all(f.is_file() for f in files):
            return False
        try:
            return bool(json.loads((ckpt / "key.json").read_text()) == key)
        except json.JSONDecodeError:
            return False

    def user_vectors(self, imps: Impressions, rows: slice) -> NDArray[np.float32]:
        return user_vectors(self.model, self.table, imps.histories(rows), self.device)

    def score(self, imps: Impressions) -> NDArray[np.float64]:
        return score_by_user_vectors(imps, self.table.cpu().numpy(), self.user_vectors)
