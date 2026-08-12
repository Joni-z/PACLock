"""Ablation: CBraMod's architecture, PACLock's tokenizer, everything else pinned.

    build_cbramod_paclockfe(n_classes, n_channels, seq_len, sequence=False)

Tests the same claim TFM-Tokenizer makes about itself -- that the tokenizer
carries the result, not the model size -- against OUR tokenizer instead of
theirs, using the same single-variable-swap logic the rest of this benchmark
is built on (docs/ARCH_SEARCH.md).

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
(docs/ARCH_SEARCH.md's PAC-window result) and would confound this one.

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

    def __init__(self):
        super().__init__()
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
            tokenizer_mode="pac_interaction", pac_token_mode="measured",
            interaction_mode="product",
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, n_patches, patch_size = x.shape
        raw = x.reshape(B, C, n_patches * patch_size)          # frontend wants (B,C,T)
        tokens = self.frontend(raw)[0]                          # (B,C,nb,P,D)
        tokens = tokens.mean(dim=2)                             # pool bands -> (B,C,P,D)

        pe = self.positional_encoding(tokens.permute(0, 3, 1, 2))
        tokens = tokens + pe.permute(0, 2, 3, 1)

        return self.proj_out(self.encoder(tokens))


def build_cbramod_paclockfe(n_classes: int, n_channels: int, seq_len: int,
                            *, dropout: float = 0.1, sequence: bool = False):
    """Same call convention as build_cbramod (build.py passes seq_len=T)."""
    backbone = PACLockCBraModBackbone()
    n_patches = seq_len // PATCH
    if sequence:
        return CBraModSequence(backbone, n_channels, n_patches, n_classes)
    return CBraModClassifier(backbone, n_channels, n_patches, n_classes, dropout)
