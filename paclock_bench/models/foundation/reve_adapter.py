"""REVE (NeurIPS 2025) adapter -- group B row.

Weights: ``vendor/reve_weights`` (HF brain-bzh/reve-base snapshot, loaded
offline via trust_remote_code from the local dir). Positions:
``vendor/reve_positions/positions.json`` -- their released electrode bank;
bipolar channels take the MIDPOINT of the two electrodes, which is their own
``position_utils.load_positions`` convention ('FP1-F7' -> average), and the
TUH electrode lists in their task configs match this benchmark's channel
lists exactly. The head mirrors ``models/classifier.ReveClassifier`` with
``pooling="no"`` (their TUH setting): flatten all C x H x 512 tokens ->
RMSNorm -> Dropout -> Linear.

Recipe deviations from their two-stage LP->FT protocol, recorded per hard
rule 2's audit trail: single-stage full fine-tune at their FT stage's
lr 1e-4 / dropout 0.15; AdamW instead of stable_adamw; cosine instead of
plateau; no mixup. Same deviation class as the other FM adapters (all rows
share this training loop).
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import yaml

from ...paths import REPO, expand  # noqa: F401  (REPO for vendor paths)
from ...paths import vendored

WEIGHTS_DIR = os.path.join(os.path.dirname(vendored("reve")), "reve_weights")
POSITIONS_JSON = os.path.join(os.path.dirname(vendored("reve")),
                              "reve_positions", "positions.json")


class _RMSNorm(nn.Module):
    """RMSNorm as in their classifier head (backbone.RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


MONO19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
TUH16 = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6",
         "T6-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
         "C4-P4", "P4-O2"]


def positions_for(dataset: str, n_channels: int) -> torch.Tensor:
    """(C, 3) coordinates from the corpus's own channel list + their bank.

    npy-release corpora (adfd/apava/iiic) have no ``channels`` key in their
    dataset yaml; they fall back by channel count -- 19 is the canonical
    10-20 set, 16 the TUH double banana, both verified per corpus in
    PROTOCOLS appendix E.
    """
    bank_raw = json.load(open(POSITIONS_JSON))
    bank = {k.upper(): v for k, v in bank_raw.items()}
    path = os.path.join(REPO, "configs", "datasets", f"{dataset}.yaml")
    channels = None
    if os.path.exists(path):
        channels = yaml.safe_load(open(path)).get("channels")
    if channels is None:
        channels = {19: MONO19, 16: TUH16}.get(n_channels)
    if channels is None:
        raise ValueError(f"no channel list for {dataset!r} ({n_channels} ch)")
    out = []
    for ch in channels:
        name = str(ch).upper()
        if "-" in name:                     # bipolar: midpoint (their rule)
            a, b = name.split("-", 1)
            pa, pb = bank[a.strip()], bank[b.strip()]
            out.append([(x + y) / 2.0 for x, y in zip(pa, pb)])
        else:
            out.append(bank[name])
    return torch.tensor(out, dtype=torch.float32)


class ReveClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, positions: torch.Tensor,
                 n_channels: int, seq_len: int, n_classes: int,
                 dropout: float):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("positions", positions)
        patch, step = encoder.patch_size, encoder.patch_size - encoder.overlap_size
        n_patches = 1 + (seq_len - patch) // step
        out_shape = n_channels * n_patches * encoder.embed_dim
        self.linear_head = nn.Sequential(
            nn.Flatten(1),
            _RMSNorm(out_shape),
            nn.Dropout(dropout),
            nn.Linear(out_shape, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        pos = self.positions.unsqueeze(0).expand(B, -1, -1)
        feats = self.encoder(x, pos)              # (B, C, H, E)
        return self.linear_head(feats)


def build_reve(n_classes: int, n_channels: int, seq_len: int, dataset: str,
               *, pretrained: bool = True, dropout: float = 0.15) -> nn.Module:
    from transformers import AutoConfig, AutoModel
    if pretrained:
        enc = AutoModel.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)
        print(f"[reve] loaded pretrained encoder from {WEIGHTS_DIR}", flush=True)
    else:
        cfg = AutoConfig.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)
        enc = AutoModel.from_config(cfg, trust_remote_code=True)
    pos = positions_for(dataset, n_channels)
    if pos.shape[0] != n_channels:
        raise ValueError(f"{dataset}: {pos.shape[0]} positions for "
                         f"{n_channels} data channels")
    return ReveClassifier(enc, pos, n_channels, seq_len, n_classes, dropout)
