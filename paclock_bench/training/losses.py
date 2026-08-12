"""Losses, one per dataset as pinned in PROTOCOLS.md appendix A.

The mapping is data-driven rather than a per-model choice: every model on a
given dataset trains against the same objective, which is what makes the C-group
architecture comparison mean anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) for CHB-MIT (~1% positives).

    Operates on a single logit per sample. The protocol pins focal loss with no
    resampling, so alpha/gamma are the only imbalance handling.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
        if logits.ndim == 2 and logits.shape[1] == 2:      # two-logit head -> positive logit
            logits = logits[:, 1] - logits[:, 0]
        t = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * t + (1 - p) * (1 - t)
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


class BinaryLogitLoss(nn.Module):
    """BCEWithLogits for TUAB / TUSZ, accepting either head shape."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.shape[1] == 2:
            logits = logits[:, 1] - logits[:, 0]
        else:
            logits = logits.squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, target.float())


def build_loss(cfg: dict) -> nn.Module:
    kind = cfg["loss"]
    if kind == "bce_with_logits":
        return BinaryLogitLoss()
    if kind == "focal":
        return FocalLoss(alpha=cfg.get("focal_alpha", 0.25),
                         gamma=cfg.get("focal_gamma", 2.0))
    if kind == "cross_entropy":
        # class_weight is null in every frozen protocol; honoured if ever set
        return nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    raise KeyError(f"unknown loss {kind!r}")
