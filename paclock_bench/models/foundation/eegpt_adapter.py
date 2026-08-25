"""EEGPT for group B: official code, official checkpoint, official recipe.

Architecture and weights come from the vendored repo (``vendor/eegpt``). This
module builds the model exactly as ``downstream/finetune_EEGPT_SleepEDF.py``
does -- that script, not the paper, is the operational definition of "EEGPT
fine-tuned on a downstream corpus", and the other ``downstream/*EEGPT*`` scripts
differ from it only in the channel list and the sequence length.

Encoder, verbatim from that script::

    EEGTransformer(img_size=[len(use_channels_names), 256*seconds],
                   patch_size=32*2, embed_num=4, embed_dim=512, depth=8,
                   num_heads=8, mlp_ratio=4.0, drop_rate=0.0,
                   attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02,
                   qkv_bias=True, norm_layer=partial(LayerNorm, eps=1e-6))

Checkpoint handling, verbatim::

    pretrain_ckpt = torch.load(load_path)
    for k, v in pretrain_ckpt['state_dict'].items():
        if k.startswith("target_encoder."):
            target_encoder_stat[k[15:]] = v
    self.target_encoder.load_state_dict(target_encoder_stat)

Two things about this model are unlike the other group-B baselines and are
easy to get wrong:

1. **The head is not a linear layer.** It is a 4-layer ``TransformerDecoder``
   over a CLS token, fed by a 2048->64 constrained projection plus a 1-D
   sin-cos positional embedding. 2048 is ``embed_dim * embed_num`` (512 * 4),
   so it is fixed across datasets. Replacing this with a linear probe would
   silently change what "EEGPT" means in the table.

2. **The encoder is put in ``eval()`` on every forward but is not frozen.**
   Gradients still flow into it; only its dropout/norm behaviour is pinned.
   ``forward`` calls ``self.target_encoder.eval()`` each time, which is what
   upstream does, so it is reproduced rather than hoisted into ``train()``.

``use_channels_names`` defines the *virtual* montage the encoder sees, not the
data's own channels: a ``Conv1dWithConstraint`` projects the dataset's channels
onto it. Upstream picks the list per corpus, and Sleep-EDF is the clearest
illustration -- 2 real channels are projected up to 13 named ones. The lists
below are upstream's, reused for the corpora of the same kind.

EEGPT was pretrained at **256 Hz** while our arrays are at 200 Hz. That is
handled the same way upstream handles its own rate mismatches: by
``temporal_interpolation`` to ``256 * seconds``, which is exactly what its
SleepEDF script does.
"""

from __future__ import annotations

import os
import sys
from functools import partial

from ...paths import vendored
import torch
import torch.nn as nn

VENDOR = vendored("eegpt")
DOWNSTREAM = os.path.join(VENDOR, "downstream")
CHECKPOINT = os.path.join(VENDOR, "checkpoint", "eegpt_mcae_58chs_4s_large4E.ckpt")

EMBED_DIM = 512
EMBED_NUM = 4
HEAD_IN = EMBED_DIM * EMBED_NUM      # 2048, the linear_probe1 input
EEGPT_RATE = 256                     # the rate EEGPT was pretrained at

# Upstream's per-corpus channel lists.
#   SLEEP  -- finetune_EEGPT_SleepEDF.py
#   TEN20  -- linear_probe_EEGPT_BCIC2A.py and linear_probe_EEGPT_KaggleERN.py
#   FULL   -- linear_probe_EEGPT_PhysioP300.py (the 10-10 set)
SLEEP = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ',
         'POZ', 'OZ']
TEN20 = ['FP1', 'FP2',
         'F7', 'F3', 'FZ', 'F4', 'F8',
         'T7', 'C3', 'CZ', 'C4', 'T8',
         'P7', 'P3', 'PZ', 'P4', 'P8',
         'O1', 'O2']
FULL = ['FP1', 'FPZ', 'FP2',
        'AF3', 'AF4',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
        'O1', 'OZ', 'O2']

# dataset -> (channel list, window seconds). The seconds are the corpus's own
# window length, matching how upstream sizes img_size (256*30 for a 30 s sleep
# epoch, 256*4 for a 4 s motor-imagery trial).
DATASETS = {
    "sleepedf":     (SLEEP, 30),
    "isruc":        (SLEEP, 30),
    "bci_iv_2a":    (TEN20, 4),      # upstream's own BCIC2A list
    "physionet_mi": (FULL, 4),       # upstream's PhysioP300 list, same corpus family
    "faced":        (TEN20, 10),
    "tuab":         (TEN20, 10),
    "tuev":         (TEN20, 5),
    "tusz":         (TEN20, 10),
    "chbmit":       (TEN20, 10),
    # 12-corpus slate additions. The TUH corpora reuse TEN20 exactly as
    # tuab/tuev/tusz do (upstream tolerates a bipolar montage on this
    # positional list); adfd/mumtaz/eegmat ARE the 19-electrode 10-20
    # set, so TEN20 is their montage rather than an approximation.
    "tuep":         (TEN20, 10),
    "tuar":         (TEN20, 5),
    "adfd":         (TEN20, 10),
    "mumtaz":       (TEN20, 5),
    "eegmat":       (TEN20, 5),
    "iiic":         (TEN20, 10),
    "caueeg":       (TEN20, 10),
    "siena":        (TEN20, 10),
}


def _import_eegpt():
    for p in (DOWNSTREAM, VENDOR):
        if p not in sys.path:
            sys.path.insert(0, p)
    from Modules.models.EEGPT_mcae import EEGTransformer            # noqa: PLC0415
    from Modules.Network.utils import (                             # noqa: PLC0415
        Conv1dWithConstraint, LinearWithConstraint,
    )
    from Modules.Transformers.pos_embed import (                    # noqa: PLC0415
        create_1d_absolute_sin_cos_embedding,
    )
    return (EEGTransformer, Conv1dWithConstraint, LinearWithConstraint,
            create_1d_absolute_sin_cos_embedding)


class EEGPTClassifier(nn.Module):
    """EEGPT encoder + upstream's transformer-decoder head.

    Mirrors ``LitEEGPTCausal`` in finetune_EEGPT_SleepEDF.py; only the
    PyTorch-Lightning scaffolding is dropped, since our training loop supplies
    the optimiser, schedule and metrics.
    """

    def __init__(self, encoder, chans_id, n_data_channels: int, n_chans: int,
                 n_classes: int, target_len: int,
                 Conv1dWithConstraint, LinearWithConstraint, pos_embed_fn,
                 dropout: float = 0.50):
        super().__init__()
        self.target_encoder = encoder
        self.register_buffer("chans_id", chans_id, persistent=False)
        self.target_len = target_len
        self._pos_embed_fn = pos_embed_fn

        self.chan_conv = Conv1dWithConstraint(n_data_channels, n_chans, 1, max_norm=1)
        self.linear_probe1 = LinearWithConstraint(HEAD_IN, 64, max_norm=1)
        self.drop = nn.Dropout(p=dropout)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                64, 4, 64 * 4, activation=torch.nn.functional.gelu,
                batch_first=False),
            num_layers=4,
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, 64) * 0.001,
                                      requires_grad=True)
        self.linear_probe2 = LinearWithConstraint(64, n_classes, max_norm=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .channel_adapt import temporal_interpolation       # noqa: PLC0415

        x = temporal_interpolation(x, self.target_len)
        x = self.chan_conv(x)
        # upstream calls .eval() inside forward on every pass: the encoder's
        # dropout/norm stay in inference mode while gradients still flow
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        pos = self._pos_embed_fn(h.shape[1], dim=64)
        h = h + pos.repeat((h.shape[0], 1, 1)).to(h)

        h = torch.cat([self.cls_token.repeat((h.shape[0], 1, 1)).to(h.device), h],
                      dim=1)
        h = h.transpose(0, 1)
        h = self.decoder(h, h)[0, :, :]
        return self.linear_probe2(h)

    def backbone_parameters(self):
        return self.target_encoder.parameters()

    def head_parameters(self):
        mods = (self.chan_conv, self.linear_probe1, self.decoder, self.linear_probe2)
        for m in mods:
            yield from m.parameters()
        yield self.cls_token


def build_eegpt(n_classes: int, n_channels: int, dataset: str, *,
                pretrained: bool = True) -> nn.Module:
    """Build EEGPT for ``dataset``; ``pretrained=False`` is the group-C row."""
    (EEGTransformer, Conv1dWithConstraint, LinearWithConstraint,
     pos_embed_fn) = _import_eegpt()

    if dataset not in DATASETS:
        raise KeyError(f"no EEGPT channel list for {dataset!r}; "
                       f"known: {sorted(DATASETS)}")
    ch_names, seconds = DATASETS[dataset]
    target_len = EEGPT_RATE * seconds

    encoder = EEGTransformer(
        img_size=[len(ch_names), target_len],
        patch_size=32 * 2,
        embed_num=EMBED_NUM,
        embed_dim=EMBED_DIM,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    chans_id = encoder.prepare_chan_ids(ch_names)
    if pretrained:
        load_pretrained(encoder)

    return EEGPTClassifier(
        encoder, chans_id, n_channels, len(ch_names), n_classes, target_len,
        Conv1dWithConstraint, LinearWithConstraint, pos_embed_fn)


def load_pretrained(encoder: nn.Module, path: str = CHECKPOINT) -> None:
    """Load the target encoder out of the released Lightning checkpoint.

    Strict, and with a positive check that blocks were actually populated: the
    prefix filter below would happily produce an empty state dict if the key
    layout ever changed, leaving a randomly initialised encoder reported as
    pretrained.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- download eegpt_mcae_58chs_4s_large4E.ckpt "
            f"from figshare (see docs) and place it there")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    prefix = "target_encoder."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_state:
        raise RuntimeError(
            f"no keys with prefix {prefix!r} in the checkpoint; "
            f"found e.g. {sorted(state)[:5]}")

    encoder.load_state_dict(enc_state)
    n_blocks = sum(1 for k in enc_state if k.startswith("blocks."))
    print(f"  EEGPT: loaded {len(enc_state)} tensors ({n_blocks} block tensors)",
          flush=True)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
