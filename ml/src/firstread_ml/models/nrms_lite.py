"""NRMS-lite: a user encoder over frozen article embeddings.

Input: a user's last ``HISTORY_LEN`` clicked-article embeddings (``hist[B, L, 384]``,
right-aligned, ``mask[B, L]`` True where a click exists). Output: a user vector in
the **same 384-d space** as article embeddings, so a score is just a dot product
and retrieval can run ANN directly on stored embeddings (ADR 0004: the item tower
is the frozen embedding).

    u = normalize(mean(hist)) + alpha * proj(additive_pool(MHSA(hist)))

``alpha`` starts at 0, so an untrained model *is* the mean-of-history baseline and
training can only learn on top of it.

Empty histories never go through attention: softmax over an all-masked row is
NaN, and ``torch.where`` does not stop NaN gradients from the unused branch. We
unmask one zero slot for those rows (so every row is finite) and then replace
their output with the population vector, which is a buffer baked into ONNX.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from firstread_core.embedding import EMBED_DIM
from firstread_ml.config import ModelConfig

MIN_SCALE, MAX_SCALE = 1.0, 100.0
_MASK_FILL = -1e9  # finite, so masked softmax never produces NaN


class AdditiveAttention(nn.Module):
    """Attention pooling with a learned query (NRMS): weights = softmax(q . tanh(Wx))."""

    def __init__(self, dim: int, query_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, query_dim)
        self.query = nn.Linear(query_dim, 1, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        logits = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # [B, L]
        weights = torch.softmax(logits.masked_fill(~mask, _MASK_FILL), dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class NRMSLite(nn.Module):
    population: Tensor

    def __init__(self, cfg: ModelConfig, population: Tensor) -> None:
        super().__init__()
        if EMBED_DIM % cfg.heads:
            raise ValueError(f"heads={cfg.heads} must divide {EMBED_DIM}")
        self.mhsa = nn.MultiheadAttention(
            EMBED_DIM, cfg.heads, dropout=cfg.attn_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool = AdditiveAttention(EMBED_DIM, cfg.query_dim)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.log_scale = nn.Parameter(torch.tensor(math.log(cfg.init_scale)))
        self.register_buffer("population", F.normalize(population.float(), dim=-1))

    @property
    def scale(self) -> Tensor:
        """Score temperature: log-parameterized and clamped, so always in [1, 100]."""
        return self.log_scale.clamp(math.log(MIN_SCALE), math.log(MAX_SCALE)).exp()

    def forward(self, hist: Tensor, mask: Tensor) -> Tensor:
        """``hist[B, L, D]``, ``mask[B, L]`` (bool) -> user vectors ``[B, D]``."""
        empty = ~mask.any(dim=1)  # [B]
        # Unmask the last slot (a zero vector: histories are right-aligned) for
        # empty rows only, so attention always has at least one key.
        safe = mask.clone()
        safe[:, -1] = safe[:, -1] | empty
        m = mask.unsqueeze(-1).to(hist.dtype)
        mean = (hist * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        base = F.normalize(mean, dim=-1, eps=1e-6)
        h, _ = self.mhsa(hist, hist, hist, key_padding_mask=~safe, need_weights=False)
        pooled = self.pool(self.dropout(h), safe)
        user = base + self.alpha * self.proj(pooled)
        return torch.where(empty.unsqueeze(-1), self.population.expand_as(user), user)

    def score(self, user: Tensor, cands: Tensor) -> Tensor:
        """``user[B, D]``, ``cands[B, C, D]`` -> logits ``[B, C]``."""
        return self.scale * torch.einsum("bd,bcd->bc", user, cands)
