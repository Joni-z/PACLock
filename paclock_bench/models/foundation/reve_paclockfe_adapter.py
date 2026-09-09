"""CroFreMo tokenizer inside REVE's encoder (replacement transplant, 2026-09-10).

REVE's patch embedding (a Linear over 200-sample patches at stride 180) is replaced by
CroFreMo's frontend; everything else -- the learnable 4D positional embedding
(fourier4d + mlp4d + ln), the 22-block transformer, the final layer and the
ReveClassifier head (flatten -> RMSNorm -> dropout -> linear) -- is the vendored model,
untouched, and Reve.forward is reproduced here from the point after patch embedding.

Grid alignment: REVE forms h = 1 + (T - 200) // 180 overlapping patches; the frontend
uses non-overlapping 180-sample patches, giving P = T // 180 = h on every corpus we run
(TUEV 1000 -> 5, IIIC 2000 -> 11), so the time grid of the 4D positional embedding is
REVE's own. Every (electrode, token row) becomes one REVE "channel": row k of electrode
c takes electrode c's (x, y, z) coordinate, and a learned per-row offset is added after
the positional embedding, so position = electrode + row. Rows are mean-pooled after
the encoder, so the classifier head has exactly the native shape.

Deviation recorded: the frontend's 180-sample patches cover 90 % of each REVE patch
window (no overlap); the encoder sees the same number of tokens per electrode as
native x 8 rows.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .reve_adapter import WEIGHTS_DIR, _RMSNorm, positions_for
from ..paclock.frontend.triaxial import TriAxialFrontend

N_BANDS = 8


class PACLockReve(nn.Module):
    def __init__(self, encoder: nn.Module, positions: torch.Tensor, n_channels: int,
                 seq_len: int, n_classes: int, dropout: float, *,
                 tokenizer_mode: str = "pac_interaction", interaction_mode: str = "rotation"):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("positions", positions)
        E = encoder.embed_dim
        patch, step = encoder.patch_size, encoder.patch_size - encoder.overlap_size
        self.step = step
        self.n_patches = 1 + (seq_len - patch) // step
        if seq_len // step < self.n_patches:
            raise ValueError(f"frontend grid {seq_len // step} < REVE grid {self.n_patches}")
        self.frontend = TriAxialFrontend(
            n_bands=N_BANDS, hidden_dim=E, sample_rate=200,
            kernel_size=201, patch_len=step, pac_patch_len=step,
            tokenizer_mode=tokenizer_mode, pac_token_mode="measured",
            interaction_mode=interaction_mode,
        )
        with torch.no_grad():
            k = self.frontend(torch.zeros(1, n_channels, step))[0].shape[2]
        self.n_rows = k
        self.row_embed = nn.Parameter(torch.zeros(k, E))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        out_shape = n_channels * self.n_patches * E
        self.linear_head = nn.Sequential(
            nn.Flatten(1), _RMSNorm(out_shape), nn.Dropout(dropout), nn.Linear(out_shape, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        B, C, T = x.shape
        h = self.n_patches
        tok = self.frontend(x.float())[0]                       # (B,C,K,P,E)
        K = tok.shape[2]
        tok = tok[:, :, :, :h]                                  # align to REVE's patch count
        tok = tok + self.row_embed.view(1, 1, K, 1, -1)
        E = tok.shape[-1]
        tok = tok.reshape(B, C * K * h, E)
        # ---- Reve.forward, from after patch embedding ----
        pos = self.positions.repeat_interleave(K, dim=0).unsqueeze(0).expand(B, -1, -1)   # (B, C*K, 3)
        pos4 = type(enc.fourier4d).add_time_patch(pos, h)                                 # (B, C*K*h, 4)
        pos_embed = enc.ln(enc.fourier4d(pos4) + enc.mlp4d(pos4))
        z = enc.transformer(tok + pos_embed, False)
        z = z.reshape(B, C, K, h, E).mean(dim=2)                # rows pooled after the encoder
        z = enc.final_layer(z)                                  # (B, C, h, E), native shape
        return self.linear_head(z)


def build_reve_paclockfe(n_classes: int, n_channels: int, seq_len: int, dataset: str, *,
                         dropout: float = 0.15, tokenizer_mode: str = "pac_interaction",
                         interaction_mode: str = "rotation") -> nn.Module:
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)
    enc = AutoModel.from_config(cfg, trust_remote_code=True)     # from scratch, published architecture
    pos = positions_for(dataset, n_channels)
    if pos.shape[0] != n_channels:
        raise ValueError(f"{dataset}: {pos.shape[0]} positions for {n_channels} channels")
    return PACLockReve(enc, pos, n_channels, seq_len, n_classes, dropout,
                       tokenizer_mode=tokenizer_mode, interaction_mode=interaction_mode)
