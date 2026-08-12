"""Channel/length adaptation for running BIOT and LaBraM outside their own corpora.

BIOT and LaBraM only ship dataset makers for the TUH corpora (plus a couple of
others), so neither repo answers the question "how do I fine-tune this on
Sleep-EDF, which has two bipolar channels?" on its own.

The answer used here is not invented: it is the one in EEGPT's repo
(``vendor/eegpt/downstream/``), whose ``finetune_BIOT_SleepEDF.py``,
``finetune_LaBraM_SleepEDF.py``, ``linear_probe_BIOT_BCIC2A.py`` and
``linear_probe_LaBraM_BCIC2A.py`` are the published source of the BIOT/LaBraM
baseline numbers in the EEGPT paper (NeurIPS 2024). Copying that adaptation is
what makes our non-TUH BIOT/LaBraM rows comparable to the ones in the
literature; inventing a "better" montage alignment would not be.

Two facts about that adaptation are worth stating plainly, because both are
cruder than one might assume and both are load-bearing:

1. **BIOT** does not match montages at all. A 1x1 convolution projects the
   dataset's channels onto the checkpoint's channel count, and the pretrained
   channel-token embedding is then indexed positionally::

       self.chan_conv = Conv1dWithConstraint(2, in_channels, 1, max_norm=1)

   So a 2-channel Sleep-EDF recording is linearly mixed up to 16 "channels"
   that have no electrode identity. The projection is trained.

2. **LaBraM** does not look up ``standard_1020`` at all::

       z = self.feature(x, input_chans=[i for i in range(C+1)])

   The positional embedding is indexed by *position*, not by electrode. This is
   why the "a bipolar montage cannot be inverted to unipolar potentials"
   objection does not block anything: upstream never attempted that alignment.
   Our TUH rows keep the electrode-identity mapping in ``labram_adapter.py``,
   because there LaBraM's own dataset maker defines one; everywhere else we
   follow this positional convention.

``temporal_interpolation`` is ported verbatim, including the ``use_avg=True``
default that subtracts the channel mean -- a common average reference applied
before the model sees the signal. It changes the numbers, so it is kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dWithConstraint(nn.Conv1d):
    """Max-norm-constrained 1x1 convolution, ported from EEGPT's Modules/Network/utils.py.

    Originally from EEGNet (Lawhern et al., 2018); EEGPT reuses it as the
    channel projection in every cross-corpus BIOT/LaBraM script. The weight is
    renormalised in ``forward`` rather than via an optimiser constraint, so the
    renorm must stay inside forward to match upstream.
    """

    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


def temporal_interpolation(x: torch.Tensor, desired_sequence_length: int,
                           mode: str = "nearest", use_avg: bool = True) -> torch.Tensor:
    """Resample the time axis, ported verbatim from EEGPT's downstream/utils.py.

    ``use_avg=True`` is upstream's default and subtracts the mean across
    channels -- a common average reference. It is not incidental: it is applied
    on every cross-corpus BIOT/LaBraM forward pass in that repo, so removing it
    would make our numbers incomparable to the published ones.

    ``mode='nearest'`` is also upstream's; it is a cruder resampler than the
    polyphase filter used in our own preprocessing, but this function only runs
    on the group-B cross-corpus path, where matching upstream is the point.
    """
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if x.dim() == 2:
        return F.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    if x.dim() == 3:
        return F.interpolate(x, desired_sequence_length, mode=mode)
    raise ValueError(
        "temporal_interpolation only supports (C, T) or (B, C, T), got "
        f"{tuple(x.shape)}")


class ChannelProjectedBIOT(nn.Module):
    """BIOT preceded by EEGPT's 1x1 channel projection.

    Used when the dataset's channel count differs from the checkpoint's, which
    is every corpus outside the TUH/CHB-MIT 16-bipolar family. Mirrors
    ``finetune_BIOT_SleepEDF.py``::

        self.chan_conv = Conv1dWithConstraint(2, in_channels, 1, max_norm=1)
        ...
        x = temporal_interpolation(x, 200*15)
        x = self.chan_conv(x)
        pred = self.feature(x)

    ``target_len`` is ``200 * window_seconds``; the caller supplies it because
    only the config knows the window length.
    """

    def __init__(self, biot: nn.Module, in_channels: int, ckpt_channels: int,
                 target_len: int):
        super().__init__()
        self.chan_conv = Conv1dWithConstraint(in_channels, ckpt_channels, 1, max_norm=1)
        self.feature = biot
        self.target_len = target_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = temporal_interpolation(x, self.target_len)
        x = self.chan_conv(x)
        return self.feature(x)

    def backbone_parameters(self):
        return self.feature.parameters()

    def head_parameters(self):
        return self.chan_conv.parameters()
