"""Ablation: CBraMod's architecture, PACLock's tokenizer, everything else pinned.

    build_cbramod_paclockfe(n_classes, n_channels, seq_len, sequence=False)

Tests the same claim TFM-Tokenizer makes about itself -- that the tokenizer
carries the result, not the model size -- against OUR tokenizer instead of
theirs, using the same single-variable-swap logic the rest of this benchmark
is built on (docs/FINDINGS.md).

What is swapped and what is not
--------------------------------
CBraMod's own PatchEmbedding does two things to a raw (B, C, n_patches, 200)
window: turn each 200-sample patch into a 200-dim vector (a small stack of
Conv2d + GroupNorm + GELU, summed with an rfft-magnitude projection), and add a
depthwise-Conv2d positional encoding over the (channel, patch) grid.

Only the first is the tokenizer. This module replaces exactly that with
PACLock's TriAxialFrontend, and reuses CBraMod's own positional encoding,
encoder (its criss-cross TransformerEncoder, vendor code, untouched) and
classifier head verbatim -- via CBraModClassifier/CBraModSequence from
cbramod_adapter.py, same wrappers the official-recipe build uses. Everything
downstream of the first 200 -> 200 projection is bit-for-bit the vendored
architecture.

d_model is pinned to CBraMod's native 200 and patch_len/pac_patch_len to its
native 200-sample patch, so the token COUNT and GRID SHAPE match exactly what
CBraMod's own tokenizer produces -- the ablation asks "is this representation
better at the model's own resolution", not "is a different resolution better",
which is a question this benchmark already answered separately
(docs/FINDINGS.md's PAC-window result) and would confound this one.

Band pooling
------------
PACLock's frontend outputs one token per (channel, band, patch); CBraMod
expects one token per (channel, patch). The n_bands axis is mean-pooled away.
This is not information loss in the way it would be for a raw amplitude
spectrogram: each band's token already IS a cross-band coupling summary --
_interaction_tokens aligns every band's phase against every lower band before
this function ever sees it -- so pooling averages several already-relational
summaries into one, rather than discarding the relational content itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cbramod_adapter import (
    BACKBONE_ARGS, PATCH, _import_cbramod, CBraModClassifier, CBraModSequence,
)
from ..paclock.frontend.triaxial import TriAxialFrontend

D_MODEL = BACKBONE_ARGS["d_model"]        # 200, CBraMod's native width
N_BANDS = 8


class PACLockCBraModBackbone(nn.Module):
    """Drop-in replacement for CBraMod's ``patch_embedding + encoder + proj_out``.

    Same call signature as the vendored ``CBraMod`` -- ``forward(x, mask=None)``
    taking (B, C, n_patches, 200) and returning (B, C, n_patches, 200) -- so
    CBraModClassifier / CBraModSequence wrap it exactly as they wrap the real
    CBraMod, with no changes to either.
    """

    def __init__(self, tokenizer_mode: str = "pac_interaction", interaction_mode: str = "product",
                 band_mode: str = "mean"):
        super().__init__()
        if band_mode not in ("mean", "channels"):
            raise ValueError(f"band_mode must be mean|channels, got {band_mode!r}")
        self.band_mode = band_mode
        CBraMod = _import_cbramod()
        real = CBraMod(**BACKBONE_ARGS)
        # keep CBraMod's own positional encoding, encoder and output head;
        # drop only its patch_embedding (the part being replaced)
        self.positional_encoding = real.patch_embedding.positional_encoding
        self.encoder = real.encoder
        self.proj_out = real.proj_out

        self.frontend = TriAxialFrontend(
            n_bands=N_BANDS, hidden_dim=D_MODEL, sample_rate=200,
            kernel_size=201, patch_len=PATCH, pac_patch_len=PATCH,
            tokenizer_mode=tokenizer_mode, pac_token_mode="measured",
            interaction_mode=interaction_mode,
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, n_patches, patch_size = x.shape
        raw = x.reshape(B, C, n_patches * patch_size)          # frontend wants (B,C,T)
        tokens = self.frontend(raw)[0]                          # (B,C,R,P,D); R = token rows per band grid
        if self.band_mode == "channels":
            # Keep the frequency axis: every (channel, row) becomes its own
            # CBraMod "channel", so the criss-cross encoder attends across
            # bands as well as electrodes. Rows are pooled only AFTER the
            # encoder, so the classifier head sees the same (B,C,P,D) it always
            # did -- head capacity is unchanged between the two band modes.
            R = tokens.shape[2]
            t = tokens.reshape(B, C * R, n_patches, tokens.shape[-1])
            pe = self.positional_encoding(t.permute(0, 3, 1, 2))
            t = t + pe.permute(0, 2, 3, 1)
            out = self.proj_out(self.encoder(t))                # (B,C*R,P,D)
            return out.reshape(B, C, R, n_patches, out.shape[-1]).mean(dim=2)
        tokens = tokens.mean(dim=2)                             # pool bands -> (B,C,P,D)

        pe = self.positional_encoding(tokens.permute(0, 3, 1, 2))
        tokens = tokens + pe.permute(0, 2, 3, 1)

        return self.proj_out(self.encoder(tokens))


def build_cbramod_paclockfe(n_classes: int, n_channels: int, seq_len: int,
                            *, dropout: float = 0.1, sequence: bool = False,
                            tokenizer_mode: str = "pac_interaction",
                            interaction_mode: str = "product",
                            band_mode: str = "mean",
                            adapter: str = "replace",
                            readout: str = "native"):
    """Same call convention as build_cbramod (build.py passes seq_len=T).

    tokenizer_mode="raw" is the control this ablation needs to mean anything.
    Swapping CBraMod's tokenizer for PACLock's PAC frontend and winning shows
    that OUR FRONTEND is better than theirs -- it does not show that the PAC
    interaction is what did it, because the frontend also brings a learned sinc
    filterbank and a per-band token axis. Running the same swap with the
    frontend's raw tokenizer separates the two, and the two are exactly
    parameter-matched (both 40,200 in the tokenizer: raw is one Conv1d(1,200,200)
    with bias; pac is Conv1d(1,100,200) without bias plus Conv1d(1,100,200) with
    bias plus a 100-entry scale), so neither arm can win on capacity."""
    if adapter == "additive":
        backbone = PACLockCBraModAugmented(tokenizer_mode=tokenizer_mode,
                                           interaction_mode=interaction_mode,
                                           readout=readout)
    else:
        backbone = PACLockCBraModBackbone(tokenizer_mode=tokenizer_mode,
                                          interaction_mode=interaction_mode,
                                          band_mode=band_mode)
    n_patches = seq_len // PATCH
    if sequence:
        return CBraModSequence(backbone, n_channels, n_patches, n_classes)
    return CBraModClassifier(backbone, n_channels, n_patches, n_classes, dropout)


class PACLockCBraModAugmented(nn.Module):
    """Additive transplant (2026-09-07): CBraMod + K extra CroFreMo rows per electrode.

    CBraMod's own PatchEmbedding is reproduced from its submodules (proj_in, spectral_proj,
    positional_encoding) so that ONE positional-encoding conv runs over the joint
    (electrode x [native, extra_1..K]) grid, exactly as CBraMod runs it over its own grid.
    Encoder and proj_out are the vendored modules, untouched. Same forward contract as
    CBraMod: (B, C, n_patches, 200) -> (B, C, n_patches, 200) when readout='native'.
    """

    def __init__(self, tokenizer_mode: str = "pac_interaction", interaction_mode: str = "rotation",
                 readout: str = "native"):
        super().__init__()
        if readout not in ("native", "mean"):
            raise ValueError(f"readout must be native|mean, got {readout!r}")
        CBraMod = _import_cbramod()
        real = CBraMod(**BACKBONE_ARGS)
        self.patch_embedding = real.patch_embedding      # kept whole: proj_in, spectral_proj, PE conv
        self.encoder = real.encoder
        self.proj_out = real.proj_out
        self.readout = readout
        self.frontend = TriAxialFrontend(
            n_bands=N_BANDS, hidden_dim=D_MODEL, sample_rate=200,
            kernel_size=201, patch_len=PATCH, pac_patch_len=PATCH,
            tokenizer_mode=tokenizer_mode, pac_token_mode="measured",
            interaction_mode=interaction_mode,
        )
        # extra rows enter CBraMod's scale via a LayerNorm (native tokens are GroupNorm-ed
        # conv features + a spectral projection; ours are a linear patch projection)
        self.extra_norm = nn.LayerNorm(D_MODEL)

    def _native(self, x: torch.Tensor) -> torch.Tensor:
        """CBraMod PatchEmbedding.forward without the positional encoding (applied jointly later)."""
        pe = self.patch_embedding
        bz, ch, pn, ps = x.shape
        mx = x.contiguous().view(bz, 1, ch * pn, ps)
        emb = pe.proj_in(mx).permute(0, 2, 1, 3).contiguous().view(bz, ch, pn, pe.d_model)
        spec = torch.fft.rfft(mx.contiguous().view(bz * ch * pn, ps), dim=-1, norm="forward")
        spec = pe.spectral_proj(torch.abs(spec).contiguous().view(bz, ch, pn, 101))
        return emb + spec

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, P, ps = x.shape
        native = self._native(x)                                          # (B,C,P,D)
        extra = self.frontend(x.reshape(B, C, P * ps))[0]                 # (B,C,K,P,D)
        extra = self.extra_norm(extra)
        K = extra.shape[2]
        grid = torch.cat([native.unsqueeze(2), extra], dim=2)             # (B,C,1+K,P,D), channel-major
        grid = grid.reshape(B, C * (1 + K), P, native.shape[-1])
        pe = self.patch_embedding.positional_encoding(grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        grid = grid + pe
        out = self.encoder(grid)                                          # (B,C*(1+K),P,D)
        out = out.reshape(B, C, 1 + K, P, out.shape[-1])
        out = out[:, :, 0] if self.readout == "native" else out.mean(dim=2)
        return self.proj_out(out)
