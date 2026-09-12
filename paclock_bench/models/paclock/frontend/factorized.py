"""One token per band with disjoint coupling and waveform coordinates.

The coupling lane retains the original rot2 width and arithmetic. The local
lane is concatenated, not added or mixed with it. Only the encoder mixes the
two. This preserves both projected features at the tokenizer output; it is
not a claim of invertibility of the complete signal or of the encoder.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .analytic import hilbert, phase_amplitude
from .triaxial import TriAxialFrontend, _patch_project, patch_pac_vector


class FactorizedFrontend(TriAxialFrontend):
    def __init__(self, n_bands, hidden_dim, sample_rate, coupling_dim=128,
                 content_source="band", content_scale=1.0, filter_range=None,
                 **kwargs):
        if coupling_dim <= 0 or coupling_dim % 2 or coupling_dim >= hidden_dim:
            raise ValueError("factorized needs an even coupling_dim < hidden_dim")
        if content_source not in ("band", "broadband", "zero"):
            raise ValueError("content_source must be band/broadband/zero")
        if not math.isfinite(content_scale) or content_scale <= 0:
            raise ValueError("content_scale must be finite and positive")
        if kwargs.get("raw_stem", "linear") != "linear":
            raise ValueError("factorized uses a linear local lane, not raw_stem")
        if kwargs.get("interaction_mode", "rotation") != "rotation":
            raise ValueError("factorized currently requires rotation")
        kwargs.pop("tokenizer_mode", None)
        super().__init__(n_bands, coupling_dim, sample_rate,
                         tokenizer_mode="pac_interaction", **kwargs)
        self.tokenizer_mode = "factorized"
        self.coupling_dim = coupling_dim
        self.content_source = content_source
        self.content_scale = float(content_scale)
        # Constructed even for zero control: identical RNG consumption, shared
        # coupling weights and encoder initialization across the ablation pair.
        self.local_projection = nn.Conv1d(
            1, hidden_dim - coupling_dim, self.patch_len, stride=self.patch_len)
        if filter_range is not None:
            low, high = map(float, filter_range)
            if not 0.1 <= low < high < sample_rate / 2:
                raise ValueError("filter_range must satisfy .1 <= low < high < Nyquist")
            # Store parameters after subtracting SincBandpass's positive
            # offsets. Legacy initialization is intentionally unchanged.
            self.sinc.min_low_hz = 0.1
            self.sinc.min_band_hz = 0.05
            edges = torch.logspace(math.log10(low), math.log10(high), n_bands + 1)
            if (edges.diff() <= self.sinc.min_band_hz).any():
                raise ValueError("filter_range produces bands narrower than .05 Hz")
            with torch.no_grad():
                self.sinc.low_hz_.copy_((edges[:-1] - .1).view(-1, 1))
                self.sinc.band_hz_.copy_((edges.diff() - .05).view(-1, 1))

    def _forward_fp32(self, x, return_amp_target=False):
        B, C, T = x.shape
        if T % self.patch_len:
            raise ValueError("factorized input length must be divisible by patch_len")
        P = T // self.patch_len
        filtered = self.sinc(x.reshape(B * C, 1, T)).reshape(B, C, self.n_bands, T)
        phase, amplitude = phase_amplitude(hilbert(filtered))
        vectors = []
        for win in self.pac_patch_lens:
            if T % win or win % self.patch_len:
                raise ValueError("PAC windows must tile the input and token patches")
            pv = patch_pac_vector(phase, amplitude, T // win, self.normalize)
            if win != self.patch_len:
                pv = pv.repeat_interleave(win // self.patch_len, dim=2)
            vectors.append(pv)
        relation = self._interaction_tokens(phase, amplitude, vectors)
        if self.content_source == "broadband":
            local = _patch_project(self.local_projection, x.reshape(B * C, T))
            local = local.reshape(B, C, 1, P, -1).expand(-1, -1, self.n_bands, -1, -1)
        else:
            local = _patch_project(
                self.local_projection, filtered.reshape(B * C * self.n_bands, T))
            local = local.reshape(B, C, self.n_bands, P, -1)
            if self.content_source == "zero":
                local = local * 0.0
        tokens = torch.cat((relation, self.content_scale * local), dim=-1)
        result = (tokens, vectors[0].abs(), self.token_band_hz())
        if return_amp_target:
            amp = amplitude.reshape(B, C, self.n_bands, P, self.patch_len)
            result += (torch.log(amp.mean(dim=-1) + 1e-6),)
        if self.return_pac_vector:
            result += (vectors[0],)
        return result
