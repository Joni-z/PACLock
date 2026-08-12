"""OURS: differentiable analytic signal via an FFT-based Hilbert transform.

Mathematically identical to the NVIDIA-cluster original
(``_reference/frontend/analytic.py``); ``tests/test_paclock_equivalence.py``
asserts that numerically. Two AMD/ROCm-motivated changes, neither of which
touches the architecture:

1. **Explicit float32 around the FFT.** The original builds the step function
   with ``dtype=x.dtype``, which makes the transform precision depend on
   whatever autocast happens to hand it. Measured on this ROCm build, the
   original does survive a bf16 autocast region (torch promotes the FFT
   internally) -- so this is hardening, not a bug fix. It is still worth
   pinning: the promotion is an implementation detail we do not control, an
   FFT over a 2000-sample window has no business running at bf16 precision,
   and a silent precision change here would move every PAC coefficient
   downstream. Making it explicit costs nothing and removes the dependency.

2. **Cached step function.** ``h`` depends only on (length, device, dtype), so
   rebuilding it every forward allocated a tensor per call in the hottest
   loop. It is now memoised. The values are unchanged.

Given a real band-limited signal we return its analytic signal ``z`` (complex),
from which the MI operator reads two things, *never an angle* (AGENT.md sec. 4):

  * unit phase vector :  ``z / |z|``   (clamped away from |z| = 0)
  * amplitude         :  ``|z|``

The Hilbert transform is the textbook frequency-domain construction: FFT, zero
the negative frequencies and double the positive ones (a step function in
frequency), inverse FFT. It is fully differentiable -- no ``atan2``, no ``arg``.
"""

from __future__ import annotations

import torch

# (length, device, dtype) -> step function. Bounded: one entry per distinct
# window length actually used.
_STEP_CACHE: dict[tuple, torch.Tensor] = {}


def _step(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Frequency-domain step: 1 at DC/Nyquist, 2 on positive freqs, 0 negative."""
    key = (T, str(device), dtype)
    h = _STEP_CACHE.get(key)
    if h is None:
        h = torch.zeros(T, dtype=dtype, device=device)
        if T % 2 == 0:
            h[0] = h[T // 2] = 1.0
            h[1: T // 2] = 2.0
        else:
            h[0] = 1.0
            h[1: (T + 1) // 2] = 2.0
        _STEP_CACHE[key] = h
    return h


def hilbert(x: torch.Tensor) -> torch.Tensor:
    """Analytic signal of real ``x`` along the last axis. Returns complex tensor.

    Always computed in float32 -- see module docstring, point 1.
    """
    xf = x.float()
    T = xf.shape[-1]
    Xf = torch.fft.fft(xf, dim=-1)
    return torch.fft.ifft(Xf * _step(T, xf.device, xf.dtype), dim=-1)


def phase_amplitude(z: torch.Tensor, eps: float = 1e-6):
    """Split analytic ``z`` into (unit phase vector, amplitude).

    Stays complex throughout; the only singular point (|z| -> 0) is clamped.
    """
    amp = z.abs()
    phase_unit = z / amp.clamp_min(eps)
    return phase_unit, amp
