"""BIOT for group B: official code, official checkpoints, official hyperparameters.

Nothing here reimplements BIOT. The architecture comes from the vendored repo
(``vendor/biot/model/biot.py``); this module only builds it with the arguments
BIOT's own ``run_binary_supervised.py`` / ``run_multiclass_supervised.py`` pass,
and loads the released weights.

Constructor arguments, from those scripts:

    BIOTClassifier(n_classes=args.n_classes,
                   n_channels=args.in_channels,
                   n_fft=args.token_size,      # 200
                   hop_length=args.hop_length) # 100

and the weights are loaded into the *encoder* only, guarded on the sample rate:

    if args.pretrain_model_path and (args.sampling_rate == 200):
        model.biot.load_state_dict(torch.load(args.pretrain_model_path))

The classifier head is left randomly initialised -- that is upstream behaviour,
and is why fine-tuning is needed at all.

Checkpoint provenance, from BIOT's README:

  EEG-PREST-16-channels        5M MGH resting samples, 16 montages
                               -> the xlsx's "BIOT★ (单语料预训练)" row
  EEG-SHHS+PREST-18-channels   + 5M SHHS sleep, 18 montages
  EEG-six-datasets-18-channels + TUAB/TUEV/CHB-MIT/IIIC training sets
                               -> the xlsx's "BIOT (4 语料)" row; BIOT reports
                                  TUAB AUROC 0.8815, which is exactly the value
                                  the xlsx lists for that row

The 18-channel checkpoints expect the 16 bipolar montages plus C3-A2 and C4-A1.
Datasets that only carry the 16 cannot use them, and ``load_pretrained`` refuses
rather than silently loading a mismatched embedding table.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

VENDOR = "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/vendor/biot"

# checkpoint -> number of channels it was pretrained with
CHECKPOINTS = {
    "prest16": ("EEG-PREST-16-channels.ckpt", 16),
    "shhs_prest18": ("EEG-SHHS+PREST-18-channels.ckpt", 18),
    "six_datasets18": ("EEG-six-datasets-18-channels.ckpt", 18),
}

# BIOT's own defaults, from its argparse and model file
TOKEN_SIZE = 200        # n_fft
HOP_LENGTH = 100
EMB_SIZE = 256
HEADS = 8
DEPTH = 4


def _import_biot():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from model.biot import BIOTClassifier      # noqa: PLC0415
    return BIOTClassifier


def build_biot(n_classes: int, n_channels: int, *,
               checkpoint: str | None = None,
               token_size: int = TOKEN_SIZE,
               hop_length: int = HOP_LENGTH,
               target_len: int | None = None) -> nn.Module:
    """Build BIOT and, when ``checkpoint`` is given, load the released weights.

    ``checkpoint=None`` gives the randomly initialised model -- that is the
    "Vanilla BIOT (无预训练)" / group-C scratch row, not a group-B row.

    When the dataset's channel count differs from the checkpoint's, the model is
    wrapped in EEGPT's 1x1 channel projection (see ``channel_adapt``) rather than
    refused. That is how the BIOT baselines outside the TUH/CHB-MIT family are
    produced in the published literature; ``seq_len`` (the window length in
    samples at 200 Hz) is then required, because the wrapper also does upstream's
    ``temporal_interpolation`` to that length.
    """
    BIOTClassifier = _import_biot()
    ckpt_channels = CHECKPOINTS[checkpoint][1] if checkpoint is not None else n_channels
    # the encoder must be built with the checkpoint's channel count, since that
    # is what the pretrained channel-token embedding was trained with; the
    # projection below is what reconciles the dataset with it
    model = BIOTClassifier(
        n_classes=n_classes,
        n_channels=ckpt_channels,
        n_fft=token_size,
        hop_length=hop_length,
        emb_size=EMB_SIZE,
        heads=HEADS,
        depth=DEPTH,
    )
    if checkpoint is not None:
        load_pretrained(model, checkpoint, ckpt_channels)

    if n_channels != ckpt_channels:
        if target_len is None:
            raise ValueError(
                f"the data has {n_channels} channels but the model was built for "
                f"{ckpt_channels}; the cross-corpus path needs target_len (window "
                f"length in samples at 200 Hz) to size temporal_interpolation")
        from .channel_adapt import ChannelProjectedBIOT   # noqa: PLC0415
        return ChannelProjectedBIOT(model, n_channels, ckpt_channels, target_len)
    return model


def load_pretrained(model: nn.Module, checkpoint: str, n_channels: int) -> None:
    """Load a released checkpoint into the encoder, strictly.

    Strict on purpose: a partial load would leave part of the encoder randomly
    initialised while still reporting a "pretrained" number, which is precisely
    the kind of silent failure hard rule 1 exists to catch.
    """
    if checkpoint not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {checkpoint!r}; "
                       f"choose from {sorted(CHECKPOINTS)}")
    fname, ckpt_channels = CHECKPOINTS[checkpoint]
    if n_channels != ckpt_channels:
        raise ValueError(
            f"{checkpoint} was pretrained with {ckpt_channels} channels but the "
            f"data has {n_channels}. The {ckpt_channels}-channel checkpoints "
            f"expect the 16 bipolar montages plus C3-A2 and C4-A1; a dataset "
            f"carrying only 16 cannot use them."
        )
    path = os.path.join(VENDOR, "pretrained-models", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    model.biot.load_state_dict(state, strict=True)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
