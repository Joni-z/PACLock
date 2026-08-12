"""Readout over the tri-axial token grid.

The backbone emits (B, C, n_bands, P, D) -- electrode x band x time-patch. How
that grid is collapsed to one vector per window is a modelling choice, and it
was never varied: every result to date used a uniform mean over all
C*n_bands*P tokens (1280 on TUEV, 2560 on FACED) followed by one linear layer.

That is a strange readout for this model in particular. The whole claim is that
*particular band pairs at particular electrodes* carry the signal; averaging
every band, electrode and time patch together with equal weight discards exactly
the structure the PAC tokenizer builds. The upstream architecture sweep
(AGENT.md 13.40-A) varied width, depth, band count, frequency mixer and both
positional encodings -- and found nothing that beat the base -- but it never
touched this.

The baselines that currently beat us do carry richer readouts: CBraMod's
`all_patch_reps` concatenates every patch representation before classifying,
which is why its head grows with channels x patches (9.85M on Sleep-EDF,
56.25M on FACED).

Modes:
  mean   : the original. Uniform mean over all tokens.
  band   : mean over electrodes and time within each band, then concatenate the
           n_bands vectors. Keeps band identity -- the axis PAC is defined on --
           at a cost of (n_bands*D) x n_classes head parameters.
  attn   : one learned query attends over all tokens. Lets the model choose
           which electrode/band/patch to read, at ~2 D^2 parameters, and does
           not assume in advance which axis matters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """``mode='mean'`` reproduces the original head exactly."""

    def __init__(self, d_model: int, num_classes: int, mode: str = "mean",
                 n_bands: int | None = None):
        super().__init__()
        if mode not in ("mean", "band", "attn"):
            raise ValueError(f"head mode must be mean/band/attn, got {mode!r}")
        self.mode = mode
        self.norm = nn.LayerNorm(d_model)
        if mode == "band":
            if not n_bands:
                raise ValueError("head mode 'band' needs n_bands")
            self.n_bands = n_bands
            self.proj = nn.Linear(d_model * n_bands, num_classes)
        elif mode == "attn":
            self.query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.query, std=0.02)
            self.to_kv = nn.Linear(d_model, d_model * 2)
            self.proj = nn.Linear(d_model, num_classes)
        else:
            self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, grid: tuple[int, int, int] | None = None):
        """``x`` is (B, N, D) with N = C*n_bands*P; ``grid`` is (C, n_bands, P)."""
        if self.mode == "mean":
            return self.proj(self.norm(x.mean(dim=1)))

        if self.mode == "band":
            if grid is None:
                raise ValueError("head mode 'band' needs the token grid shape")
            C, nb, P = grid
            B, N, D = x.shape
            # (B, C, nb, P, D) -> mean over electrodes and patches, per band
            pooled = x.reshape(B, C, nb, P, D).mean(dim=(1, 3))     # (B, nb, D)
            return self.proj(self.norm(pooled).reshape(B, nb * D))

        # attn: a single learned query reads the grid
        h = self.norm(x)
        k, v = self.to_kv(h).chunk(2, dim=-1)
        q = self.query.expand(h.shape[0], -1, -1)
        pooled = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(pooled.squeeze(1))
