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
  spatial: mean over bands and patches, then one linear layer that sees every
           electrode separately. The only mode that does NOT collapse the
           electrode axis, so it is the only one that can represent a spatial
           contrast -- which is what motor imagery (mu/beta desynchronisation
           over the contralateral sensorimotor strip) and emotion (frontal
           asymmetry) actually are. Costs (C*D) x n_classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """``mode='mean'`` reproduces the original head exactly."""

    def __init__(self, d_model: int, num_classes: int, mode: str = "mean",
                 n_bands: int | None = None, n_channels: int | None = None,
                 n_patches: int | None = None):
        super().__init__()
        if mode not in ("mean", "band", "attn", "spatial", "meanspatial",
                        "gated_meanspatial", "flatten"):
            raise ValueError(
                "head mode must be mean/band/attn/spatial/meanspatial/"
                f"gated_meanspatial/flatten, got {mode!r}")
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
            # Built here, NOT lazily in forward(): the optimizer is handed
            # model.parameters() before the first forward runs, so a layer
            # created inside forward() never enters the parameter groups and
            # keeps its initial random weights for the whole run -- the head
            # would look like it was tested when it was never trained.
            if not n_channels:
                raise ValueError("head mode 'spatial' needs n_channels")
            self.n_channels = n_channels
            self.proj = nn.Linear(n_channels * d_model, num_classes)
        elif mode in ("meanspatial", "gated_meanspatial"):
            # BOTH readouts, concatenated: the global mean vector (what every
            # tier-1 result was built on) and the per-electrode vectors (what
            # motor imagery needs). Nothing is removed -- the classifier sees
            # [mean ; electrode_1 ; ... ; electrode_C] and the gradient decides
            # how much spatial identity to use. The measured motivation: the
            # spatial head gains +0.08~0.11 on the MI corpora but COSTS 0.056
            # on TUEV, and that failure came from REPLACING the mean path, the
            # same mistake the constitutive tokenizer made at the token level.
            if not n_channels:
                raise ValueError("head mode 'meanspatial' needs n_channels")
            self.n_channels = n_channels
            self.proj = nn.Linear((n_channels + 1) * d_model, num_classes)
            if mode == "gated_meanspatial":
                # Zero-init gate on the per-electrode block. The plain
                # meanspatial head CONTAINS the mean head mathematically, yet
                # measured single seeds show its randomly initialised spatial
                # block derails the seizure corpora (TUSZ -0.096, CHB-MIT
                # collapse) -- the same no-worst-case failure the product
                # tokenizer had. gamma = 0 makes the init exactly a mean head;
                # spatial identity must be earned by the gradient, the one
                # principle that has worked every time it was applied here
                # (fused beta, interaction alpha).
                self.spatial_scale = nn.Parameter(torch.zeros(1))
        elif mode == "flatten":
            # H4 (2026-08-20): every head so far pools the TIME axis uniformly,
            # but an MI trial is cue-locked -- the discriminative content is
            # the power TRAJECTORY over the trial, and averaging collapses the
            # trajectory to a point. CBraMod, which wins these paradigm tasks,
            # flattens its whole (channel, patch, feature) grid into the
            # classifier instead. This is that readout on our grid: pool the
            # band axis only, keep (C, P, D), flatten into one linear layer.
            if not (n_channels and n_patches):
                raise ValueError("head mode 'flatten' needs n_channels and n_patches")
            self.n_channels = n_channels
            self.n_patches = n_patches
            self.proj = nn.Linear(n_channels * n_patches * d_model, num_classes)
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
            if C != self.n_channels:
                raise ValueError(
                    f"head mode 'spatial' was built for {self.n_channels} "
                    f"electrodes but the grid has {C}")
            return self.proj(self.norm(pooled).reshape(B, C * D))

        if self.mode in ("meanspatial", "gated_meanspatial"):
            if grid is None:
                raise ValueError("head mode meanspatial needs the token grid shape")
            C, nb, P = grid
            B, N, D = x.shape
            if C != self.n_channels:
                raise ValueError(
                    f"head mode 'meanspatial' was built for {self.n_channels} "
                    f"electrodes but the grid has {C}")
            # pool THEN norm, matching the mean and spatial heads own order --
            # LayerNorm and mean do not commute, and the containment property
            # (zeroing the spatial columns must recover the mean head exactly)
            # only holds if the mean branch is literally the mean head.
            g_mean = self.norm(x.mean(dim=1))                       # (B, D)
            per_el = self.norm(x.reshape(B, C, nb, P, D).mean(dim=(2, 3)))
            per_el = per_el.reshape(B, C * D)
            if self.mode == "gated_meanspatial":
                per_el = per_el * self.spatial_scale
            return self.proj(torch.cat([g_mean, per_el], dim=1))

        if self.mode == "flatten":
            if grid is None:
                raise ValueError("head mode flatten needs the token grid shape")
            C, nb, P = grid
            B, N, D = x.shape
            if C != self.n_channels or P != self.n_patches:
                raise ValueError(
                    f"head mode 'flatten' was built for (C={self.n_channels}, "
                    f"P={self.n_patches}) but the grid is (C={C}, P={P})")
            pooled = self.norm(x.reshape(B, C, nb, P, D).mean(dim=2))  # (B,C,P,D)
            return self.proj(pooled.reshape(B, C * P * D))

        # attn: a single learned query reads the grid
        h = self.norm(x)
        k, v = self.to_kv(h).chunk(2, dim=-1)
        q = self.query.expand(h.shape[0], -1, -1)
        pooled = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(pooled.squeeze(1))
