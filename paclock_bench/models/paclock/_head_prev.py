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
        if mode not in ("mean", "band", "attn", "spatial"):
            raise ValueError(
                f"head mode must be mean/band/attn/spatial, got {mode!r}")
        self.mode = mode
        self.norm = nn.LayerNorm(d_model)
        if mode == "band":
            if not n_bands:
                raise ValueError("head mode 'band' needs n_bands")
            self.n_bands = n_bands
            self.proj = nn.Linear(d_model * n_bands, num_classes)
        elif mode == "spatial":
            # Pool the band and time axes but KEEP electrode identity, then let
            # one linear layer see every electrode separately.
            #
            # mean/band/attn all collapse the electrode axis, so the classifier
            # cannot know *where* on the scalp anything happened. That is fine
            # for the TUH corpora (a 16-channel bipolar montage, and the label
            # is a property of the recording), but it throws away the entire
            # signal on paradigms whose label IS a spatial pattern -- motor
            # imagery is mu/beta desynchronisation over the contralateral
            # sensorimotor strip, and emotion has well-known frontal asymmetry.
            # CBraMod, which beats us by 0.2-0.4 on exactly those corpora,
            # flattens its whole (channel, patch, feature) grid into the
            # classifier (vendor/cbramod/models/model_for_faced.py,
            # ); this is the same idea with the band and time
            # axes pooled first so the layer stays small.
            self.pool_proj = None          # built lazily: needs C from the grid
            self.num_classes = num_classes
            self.d_model = d_model
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

        if self.mode == "spatial":
            if grid is None:
                raise ValueError("head mode spatial needs the token grid shape")
            C, nb, P = grid
            B, N, D = x.shape
            # (B, C, nb, P, D) -> mean over bands and patches, per electrode
            pooled = x.reshape(B, C, nb, P, D).mean(dim=(2, 3))     # (B, C, D)
            pooled = self.norm(pooled).reshape(B, C * D)
            if self.pool_proj is None:
                self.pool_proj = nn.Linear(C * D, self.num_classes).to(pooled.device)
            return self.pool_proj(pooled)

        # attn: a single learned query reads the grid
        h = self.norm(x)
        k, v = self.to_kv(h).chunk(2, dim=-1)
        q = self.query.expand(h.shape[0], -1, -1)
        pooled = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(pooled.squeeze(1))
