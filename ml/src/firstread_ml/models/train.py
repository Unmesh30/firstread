"""Train NRMS-lite on MIND impressions.

For every clicked candidate in a training impression we draw K negatives from
the *same* impression (with replacement if it has fewer than K) and minimise
softmax cross-entropy over the 1 + K candidates, as in NRMS.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from firstread_core.features import HISTORY_LEN, history_index_matrix
from firstread_ml.config import NRMSLiteConfig
from firstread_ml.mind.data import Impressions
from firstread_ml.models.nrms_lite import NRMSLite

EpochCallback = Callable[[int, NRMSLite], dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


@dataclass(frozen=True)
class Samples:
    """One training sample per clicked candidate."""

    imp: NDArray[np.int64]  # impression row of each sample
    pos: NDArray[np.int32]  # clicked article
    neg_offsets: NDArray[np.int64]  # per impression, into ``neg``
    neg: NDArray[np.int32]  # non-clicked articles, grouped by impression

    @property
    def n(self) -> int:
        return len(self.pos)


def build_samples(train: Impressions) -> Samples:
    cand_imp = train.cand_imp
    is_pos = train.label == 1
    neg_sizes = np.bincount(cand_imp[~is_pos], minlength=train.n)
    if (neg_sizes == 0).any():
        raise ValueError("every training impression needs at least one negative")
    offsets = np.zeros(train.n + 1, dtype=np.int64)
    np.cumsum(neg_sizes, out=offsets[1:])
    return Samples(
        imp=cand_imp[is_pos],
        pos=train.cand[is_pos],
        neg_offsets=offsets,
        neg=train.cand[~is_pos],
    )


def sample_negatives(
    s: Samples, rows: NDArray[np.int64], k: int, rng: np.random.Generator
) -> NDArray[np.int32]:
    """``[len(rows), k]`` negatives, each drawn from its sample's own impression."""
    imp = s.imp[rows]
    start = s.neg_offsets[imp]
    size = s.neg_offsets[imp + 1] - start
    pick = (rng.random((len(rows), k)) * size[:, None]).astype(np.int64)
    out: NDArray[np.int32] = s.neg[start[:, None] + pick]
    return out


def gather(
    table: torch.Tensor, idx: NDArray[np.integer], mask: NDArray[np.bool_], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up padded history indices on ``device``; padded slots become zeros."""
    i = torch.from_numpy(np.where(mask, idx, 0).astype(np.int64)).to(device)
    m = torch.from_numpy(mask).to(device)
    return table[i] * m.unsqueeze(-1), m


def train_nrms_lite(
    cfg: NRMSLiteConfig,
    train: Impressions,
    emb: NDArray[np.float32],
    population: NDArray[np.float32],
    on_epoch: EpochCallback | None = None,
    log: Callable[[dict[str, float], int], None] | None = None,
) -> tuple[NRMSLite, list[dict[str, Any]]]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    rng = np.random.default_rng(cfg.seed)
    samples = build_samples(train)
    hist_idx, hist_mask = history_index_matrix(train.histories(), HISTORY_LEN)
    hist_idx = hist_idx.astype(np.int32)  # MIND-large: 2.2M x 50 -> ~440 MB instead of ~880 MB
    table = torch.from_numpy(emb).to(device)
    model = NRMSLite(cfg.model, torch.from_numpy(population)).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    bs, k = cfg.train.batch_size, cfg.train.negatives
    total_steps = cfg.train.epochs * -(-samples.n // bs)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        if cfg.train.lr_schedule == "cosine"
        else None
    )
    print(f"training NRMS-lite on {device}: {samples.n:,} samples/epoch, batch {bs}, K={k}")
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0, total, steps = time.perf_counter(), 0.0, 0
        order = rng.permutation(samples.n)
        for start in range(0, samples.n, bs):
            rows = order[start : start + bs]
            imp = samples.imp[rows]
            hist, mask = gather(table, hist_idx[imp], hist_mask[imp], device)
            cand_ids = np.concatenate(
                [samples.pos[rows, None], sample_negatives(samples, rows, k, rng)], axis=1
            )
            cands = table[torch.from_numpy(cand_ids.astype(np.int64)).to(device)]
            logits = model.score(model(hist, mask), cands)
            target = torch.zeros(len(rows), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)  # the clicked item is column 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            total += float(loss.detach())
            steps += 1
        stats: dict[str, Any] = {
            "epoch": epoch,
            "loss": total / max(steps, 1),
            "alpha": float(model.alpha.detach()),
            "scale": float(model.scale.detach()),
            "lr": float(opt.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - t0,
        }
        if on_epoch is not None:
            stats.update(on_epoch(epoch, model))
        history.append(stats)
        if log is not None:
            log({k_: float(v) for k_, v in stats.items() if k_ != "epoch"}, epoch)
        print(
            "  "
            + " ".join(
                f"{k_}={v:.4f}" if isinstance(v, float) else f"{k_}={v}" for k_, v in stats.items()
            ),
            flush=True,
        )
    return model, history


@torch.no_grad()
def user_vectors(
    model: NRMSLite,
    table: torch.Tensor,
    histories: list[NDArray[np.int32]],
    device: torch.device,
    batch: int = 4096,
) -> NDArray[np.float32]:
    """Encode users (eval mode, no dropout) in batches."""
    model.eval()
    out = []
    for start in range(0, len(histories), batch):
        idx, mask = history_index_matrix(histories[start : start + batch], HISTORY_LEN)
        hist, m = gather(table, idx, mask, device)
        out.append(model(hist, m).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, table.shape[1]), np.float32)
