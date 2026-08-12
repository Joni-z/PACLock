"""Group A: light supervised baselines.

Thin adapter over the **official BIOT implementations**, vendored verbatim under
``biot_official/`` (``ycq091044/BIOT``, ``model/``). Nothing here reimplements an
architecture.

This replaced a from-scratch reimplementation. The reimplementation ran and
trained, but its parameter counts did not match the values the xlsx lists per
row (e.g. FFCL 0.70M against the expected 2.4M), which means it was not the
same architecture -- and group A exists purely to check that our pipeline
reproduces published numbers. A calibration set built on the wrong architecture
calibrates nothing, and every downstream group B/C conclusion would inherit the
error. The fix is to run the authors' code, not a lookalike.

Two structural facts the reimplementation had wrong, worth stating because they
change what the models see:

* ContraWR / CNN-Transformer / FFCL consume an **STFT spectrogram**, not the raw
  waveform, and their ResBlocks are therefore 2D.
* FFCL's LSTM runs over an interleaved reshape of the signal
  (``shorten()``), not a naive stride-subsample.

Hyperparameters are taken from the authors' own training scripts
(``run_binary_supervised.py`` / ``run_multiclass_supervised.py``), not from the
model files' constructor defaults -- the two disagree, and the scripts are what
produced the published numbers:

* ``token_size = 200``, ``hop_length = 100`` -> ``steps = hop_length // 5 = 20``
* ContraWR and FFCL take ``fft = token_size``; CNN-Transformer takes
  ``fft = sampling_rate``. The scripts really do differ here.
* **ST-Transformer uses ``depth=4``**, while ``st_transformer.py`` defaults to
  3. This alone moves it from 2.64M to 3.43M against the 3.5M the xlsx lists.
* SPaRCNet uses ``block_layers=4, growth_rate=16, bn_size=16``, which are the
  file defaults.

``token_size`` is 200 because the corpora are 200 Hz. Sleep-EDF is 100 Hz, so
its config must override ``fft``/``token_size`` rather than inherit 200.
"""

from __future__ import annotations

import torch.nn as nn

from .biot_official.cnn_transformer import CNNTransformer as _CNNTransformer
from .biot_official.contrawr import ContraWR as _ContraWR
from .biot_official.ffcl import FFCL as _FFCL
from .biot_official.sparcnet import SPaRCNet as _SPaRCNet
from .biot_official.st_transformer import STTransformer as _STTransformer


# BIOT script defaults. token_size assumes a 200 Hz corpus; a 100 Hz dataset
# (Sleep-EDF) must override token_size in its experiment config.
TOKEN_SIZE = 200
HOP_LENGTH = 100


def _steps(kw: dict) -> int:
    return kw.pop("steps", kw.pop("hop_length", HOP_LENGTH) // 5)


def _sparcnet(in_channels, seq_len, num_classes, sample_rate, **kw):
    kw.setdefault("block_layers", 4)
    kw.setdefault("growth_rate", 16)
    kw.setdefault("bn_size", 16)
    kw.setdefault("drop_rate", 0.5)
    kw.setdefault("conv_bias", True)
    kw.setdefault("batch_norm", True)
    return _SPaRCNet(
        in_channels=in_channels,
        sample_length=seq_len,
        n_classes=num_classes,
        **kw,
    )


def _contrawr(in_channels, seq_len, num_classes, sample_rate, **kw):
    # script: fft=token_size, steps=hop_length//5
    return _ContraWR(
        in_channels=in_channels,
        n_classes=num_classes,
        fft=kw.pop("token_size", TOKEN_SIZE),
        steps=_steps(kw),
        **kw,
    )


def _cnn_transformer(in_channels, seq_len, num_classes, sample_rate, **kw):
    # script: fft=sampling_rate here, unlike ContraWR/FFCL which use token_size.
    # token_size is still accepted so one config block can set the STFT window
    # for all three STFT models; on the 200 Hz corpora the two coincide anyway.
    fft = kw.pop("fft", None)
    token_size = kw.pop("token_size", None)
    return _CNNTransformer(
        in_channels=in_channels,
        n_classes=num_classes,
        fft=fft if fft is not None else (token_size if token_size is not None
                                         else sample_rate),
        steps=_steps(kw),
        dropout=kw.pop("dropout", 0.2),
        nhead=kw.pop("nhead", 4),
        emb_size=kw.pop("emb_size", 256),
        **kw,
    )


def _ffcl(in_channels, seq_len, num_classes, sample_rate, **kw):
    # script: fft=token_size, steps=hop_length//5, shrink_steps=20
    return _FFCL(
        in_channels=in_channels,
        n_classes=num_classes,
        fft=kw.pop("token_size", TOKEN_SIZE),
        steps=_steps(kw),
        sample_length=seq_len,
        shrink_steps=kw.pop("shrink_steps", 20),
        **kw,
    )


def _st_transformer(in_channels, seq_len, num_classes, sample_rate, **kw):
    # depth=4 comes from the training script; the model file defaults to 3.
    return _STTransformer(
        emb_size=kw.pop("emb_size", 256),
        depth=kw.pop("depth", 4),
        n_classes=num_classes,
        channel_legnth=seq_len,          # sic: the official spelling
        n_channels=in_channels,
        **kw,
    )


REGISTRY = {
    "sparcnet": _sparcnet,
    "contrawr": _contrawr,
    "cnn_transformer": _cnn_transformer,
    "ffcl": _ffcl,
    "st_transformer": _st_transformer,
}

# Parameter counts the xlsx lists for each group-A row. Checked by
# tests/test_baseline_params.py in the TUEV configuration (16 ch, 1000 samples,
# 200 Hz), which is the configuration those counts were measured in -- FFCL is
# length-dependent and pins it (2.416M measured against 2.40M listed).
EXPECTED_PARAMS_M = {
    "sparcnet": 0.79,
    "contrawr": 1.6,
    "cnn_transformer": 3.2,
    "ffcl": 2.4,
    "st_transformer": 3.5,
}

# Rows where the xlsx count cannot be reproduced from the authors' own code at
# the authors' own settings. Recorded rather than tuned away: matching a
# parameter count by inventing hyperparameters would silently make the
# calibration baseline a different model from the published one, which is the
# exact failure this whole check exists to catch.
KNOWN_PARAM_DISCREPANCIES = {
    "sparcnet": (
        "Measures 0.99M at TUEV (1.14M at TUAB) using BIOT's own script "
        "settings (block_layers=4, growth_rate=16, bn_size=16 -- verified "
        "against run_binary_supervised.py and run_multiclass_supervised.py). "
        "The xlsx lists 0.79M, which no tested window length or documented "
        "hyperparameter combination reproduces. Treated as an inconsistency in "
        "the published figure; the architecture here is the official one. "
        "The binding calibration check is the metric comparison, not this count."
    ),
}
