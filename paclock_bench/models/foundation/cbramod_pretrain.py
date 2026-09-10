"""Masked-patch pretraining models for the CBraMod re-pretraining experiment (2026-09-10).

The TFM-Tokenizer setting: a host foundation model is pretrained WITH the new tokenizer,
then finetuned -- rather than a released checkpoint being disturbed by new tokens (which
we showed fails, STATUS 14). Two models share one loop, one pool and one objective
(CBraMod's own: MSE reconstruction of the raw 200-sample patches at masked
(electrode, patch) cells, mask ratio 0.5):

  CBraModNativeMPM   the vendored CBraMod verbatim (its PatchEmbedding, mask_encoding,
                     positional conv, criss-cross encoder, proj_out)
  CBraModCroFreMoMPM the same encoder / positional conv / proj_out, with CBraMod's
                     PatchEmbedding replaced by CroFreMo's frontend (8 rows per electrode,
                     rows as channels -- the replacement transplant of STATUS 15). Masked
                     cells are zeroed in the raw signal BEFORE the frontend (so nothing of a
                     masked patch reaches any token through the filterbank's receptive
                     field) and their rows are replaced by a learned mask token.

Both save state dicts whose keys match the finetune-time modules, so the checkpoints load
into `build_cbramod(pretrained_path=...)` and `build_cbramod_paclockfe(pretrained_path=...)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cbramod_adapter import BACKBONE_ARGS, PATCH, _import_cbramod
from ..paclock.frontend.triaxial import TriAxialFrontend

D_MODEL = BACKBONE_ARGS["d_model"]
N_BANDS = 8


def generate_mask(bz: int, ch: int, pn: int, mask_ratio: float, device) -> torch.Tensor:
    """CBraMod's utils.generate_mask, verbatim: Bernoulli(mask_ratio) over (B, C, P)."""
    return torch.zeros((bz, ch, pn), dtype=torch.long, device=device).bernoulli_(mask_ratio)


class CBraModNativeMPM(nn.Module):
    def __init__(self):
        super().__init__()
        CBraMod = _import_cbramod()
        self.backbone = CBraMod(**BACKBONE_ARGS)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, mask=mask)                     # (B,C,P,200)

    def export_state_dict(self) -> dict:
        return self.backbone.state_dict()                      # loads into CBraMod(**BACKBONE_ARGS)


class CBraModCroFreMoMPM(nn.Module):
    def __init__(self, tokenizer_mode: str = "pac_interaction", interaction_mode: str = "rotation"):
        super().__init__()
        CBraMod = _import_cbramod()
        real = CBraMod(**BACKBONE_ARGS)
        self.positional_encoding = real.patch_embedding.positional_encoding
        self.encoder = real.encoder
        self.proj_out = real.proj_out
        self.frontend = TriAxialFrontend(
            n_bands=N_BANDS, hidden_dim=D_MODEL, sample_rate=200,
            kernel_size=201, patch_len=PATCH, pac_patch_len=PATCH,
            tokenizer_mode=tokenizer_mode, pac_token_mode="measured",
            interaction_mode=interaction_mode,
        )
        self.mask_token = nn.Parameter(torch.zeros(D_MODEL))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, P, ps = x.shape
        m = (mask == 1)
        xm = x.masked_fill(m.unsqueeze(-1), 0.0)               # masked patches carry nothing
        tokens = self.frontend(xm.reshape(B, C, P * ps))[0]    # (B,C,R,P,D)
        R = tokens.shape[2]
        mr = m.unsqueeze(2).unsqueeze(-1)                       # (B,C,1,P,1)
        tokens = torch.where(mr, self.mask_token.view(1, 1, 1, 1, -1).to(tokens.dtype), tokens)
        grid = tokens.reshape(B, C * R, P, D_MODEL)
        pe = self.positional_encoding(grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.encoder(grid + pe)                           # (B,C*R,P,D)
        out = out.reshape(B, C, R, P, D_MODEL).mean(dim=2)      # rows pooled -> one prediction per cell
        return self.proj_out(out)                               # (B,C,P,200)

    def export_state_dict(self) -> dict:
        # keys match PACLockCBraModBackbone (frontend / positional_encoding / encoder / proj_out)
        return {k: v for k, v in self.state_dict().items() if not k.startswith("mask_token")}


def build_mpm(kind: str, **kw) -> nn.Module:
    if kind == "native":
        return CBraModNativeMPM()
    if kind == "crofremo":
        return CBraModCroFreMoMPM(**kw)
    raise ValueError(f"unknown kind {kind!r}")
