"""Tuned supervised specialists: EEGNet-v4 and EEG-Conformer via braindecode.

Group A additions per the 2026-08-24 plan (docs/DIRECTION.md): the critique
literature's finding is that UNTUNED deep baselines are strawmen
(arXiv:2605.26910), so these two run with a small, declared search budget --
the grid lives in configs/experiments/<ds>_{eegnet,eegconformer}_t*.yaml and
the winner is chosen on validation, exactly like every other model here.

Why braindecode rather than a reimplementation: the same rule that put the
official BIOT code behind group A (see light_supervised.py's docstring).
braindecode's EEGConformer is the authors' own upstreamed port (Song et al.,
TNSRE 2023), and EEGNetv4 is the community-standard PyTorch reference of
Lawhern et al. braindecode 0.8 appends a LogSoftmax head by default; it is
disabled here because the training loop applies CrossEntropy/BCE on logits.
"""

from __future__ import annotations

import torch.nn as nn


def _strip_log_softmax(model: nn.Module) -> nn.Module:
    """Remove a trailing LogSoftmax wherever braindecode 0.8 put one."""
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.LogSoftmax):
            setattr(model, name, nn.Identity())
        else:
            _strip_log_softmax(mod)
    return model


def _eegnet(in_channels, seq_len, num_classes, sample_rate, **kw):
    from braindecode.models import EEGNetv4
    net = EEGNetv4(
        n_chans=in_channels,
        n_outputs=num_classes,
        n_times=seq_len,
        drop_prob=kw.get("dropout", 0.25),
    )
    return _strip_log_softmax(net)


def _eegconformer(in_channels, seq_len, num_classes, sample_rate, **kw):
    from braindecode.models import EEGConformer
    net = EEGConformer(
        n_chans=in_channels,
        n_outputs=num_classes,
        n_times=seq_len,
        drop_prob=kw.get("dropout", 0.5),
        final_fc_length="auto",
    )
    return _strip_log_softmax(net)


REGISTRY = {
    "eegnet": _eegnet,
    "eegconformer": _eegconformer,
}
