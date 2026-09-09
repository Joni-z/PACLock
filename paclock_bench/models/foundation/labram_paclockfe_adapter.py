"""CroFreMo tokenizer inside LaBraM's encoder (replacement transplant, 2026-09-10).

LaBraM's own patch embedding (TemporalConv over 1 s / 200-sample patches) is replaced
by CroFreMo's frontend; the encoder (cls token, electrode positional embedding, time
embedding, blocks, norm, mean-pooling, head) is the vendored NeuralTransformer,
untouched, and its forward_features is reproduced here line for line from the point
after patch_embed. Every (electrode, token row) becomes one LaBraM "electrode":
row k of electrode c takes electrode c's positional-embedding index plus a learned
per-row offset, so position = electrode + row. Rows are pooled by LaBraM's own
mean-pooling over all tokens. Same grid as the CBraMod replacement transplant
(`band_mode="channels"`), which is the transplant reported in the paper.

d_model is LaBraM's native 200 and the patch is its native 200 samples, so the
frontend's tokens drop straight into the encoder's token dimension.

Deviation recorded: relative position bias is off (`use_rel_pos_bias=False`) in this
transplant for every corpus; the from-scratch native rows on positional-mode corpora
have it on (POSITIONAL_MODEL_ARGS). The bias table is sized by the native token grid
and does not extend to the row-expanded grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .labram_adapter import (
    CH_NAMES, MODEL_ARGS, _import_labram, get_input_chans,
)
from ..paclock.frontend.triaxial import TriAxialFrontend

D_MODEL = 200          # LaBraM embed_dim
PATCH = 200            # LaBraM patch_size (1 s at 200 Hz)
N_BANDS = 8


class PACLockLaBraM(nn.Module):
    def __init__(self, model: nn.Module, electrode_idx: list[int], *,
                 tokenizer_mode: str = "pac_interaction", interaction_mode: str = "rotation",
                 target_len: int | None = None):
        super().__init__()
        self.model = model
        self.target_len = target_len
        self.frontend = TriAxialFrontend(
            n_bands=N_BANDS, hidden_dim=D_MODEL, sample_rate=200,
            kernel_size=201, patch_len=PATCH, pac_patch_len=PATCH,
            tokenizer_mode=tokenizer_mode, pac_token_mode="measured",
            interaction_mode=interaction_mode,
        )
        with torch.no_grad():
            k = self.frontend(torch.zeros(1, len(electrode_idx), PATCH))[0].shape[2]
        self.n_rows = k
        self.row_embed = nn.Parameter(torch.zeros(k, D_MODEL))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        # [cls] + electrode index repeated for each of its rows (row-major within electrode)
        ic = [0] + [e for e in electrode_idx for _ in range(k)]
        self.register_buffer("input_chans", torch.IntTensor(ic), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_len is not None:
            from .channel_adapt import temporal_interpolation   # noqa: PLC0415
            x = temporal_interpolation(x, self.target_len)
        B, C, T = x.shape
        if T % PATCH:
            raise ValueError(f"time axis must be divisible by {PATCH}, got {T}")
        P = T // PATCH
        tok = self.frontend(x)[0]                                   # (B,C,K,P,D)
        K = tok.shape[2]
        tok = tok + self.row_embed.view(1, 1, K, 1, -1)
        tok = tok.reshape(B, C * K, P, D_MODEL)
        m = self.model
        # ---- NeuralTransformer.forward_features, from after patch_embed ----
        h = tok.flatten(1, 2)                                       # (B, C*K*P, D)
        cls = m.cls_token.expand(B, -1, -1)
        h = torch.cat((cls, h), dim=1)
        pe_used = m.pos_embed[:, self.input_chans]
        pe = pe_used[:, 1:, :].unsqueeze(2).expand(B, -1, P, -1).flatten(1, 2)
        pe = torch.cat((pe_used[:, 0:1, :].expand(B, -1, -1), pe), dim=1)
        h = h + pe
        te = m.time_embed[:, 0:P, :].unsqueeze(1).expand(B, C * K, -1, -1).flatten(1, 2)
        h = torch.cat((h[:, :1, :], h[:, 1:, :] + te), dim=1)
        h = m.pos_drop(h)
        for blk in m.blocks:
            h = blk(h, rel_pos_bias=None)
        h = m.norm(h)
        if m.fc_norm is not None:
            h = m.fc_norm(h[:, 1:, :].mean(1))
        else:
            h = h[:, 0]
        return m.head(h)


def build_labram_paclockfe(n_classes: int, n_channels: int, *,
                           montage_mode: str = "electrode",
                           target_len: int | None = None,
                           tokenizer_mode: str = "pac_interaction",
                           interaction_mode: str = "rotation",
                           model_name: str = "labram_base_patch200_200") -> nn.Module:
    create_model = _import_labram()
    if montage_mode == "electrode":
        if n_channels != len(CH_NAMES):
            raise ValueError(f"electrode mode expects {len(CH_NAMES)} channels, got {n_channels}")
        electrode_idx = get_input_chans(CH_NAMES)[1:]
    elif montage_mode == "positional":
        if target_len is None:
            raise ValueError("montage_mode='positional' needs target_len")
        electrode_idx = list(range(1, n_channels + 1))
    else:
        raise ValueError(f"unknown montage_mode {montage_mode!r}")
    model = create_model(model_name, num_classes=n_classes, **MODEL_ARGS)   # rel_pos_bias off
    return PACLockLaBraM(model, electrode_idx, tokenizer_mode=tokenizer_mode,
                         interaction_mode=interaction_mode, target_len=target_len)
