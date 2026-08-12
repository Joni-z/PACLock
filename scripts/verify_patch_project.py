"""Prove the GEMM tokeniser is the same map as the Conv1d it replaced, then show
where the rest of the frontend's time actually goes.

Run on a compute node (slurm/verify_patch.slurm).

Section 1-3 are the correctness case for the change already made. Section 4 is
the part that decides what to do next: the patch_len ladder from the finished
runs puts the tokenisers at only ~30% of a step, so replacing them can be worth
at most ~1.4x on its own. The other ~70% is the sinc filterbank, the Hilbert
FFT and the PAC statistic, and guessing which of those dominates is exactly the
mistake that produced three wrong speed diagnoses here already.

Equality below means "to fp32 rounding", not bit-identical: a convolution and a
GEMM reduce over ``patch`` in a different order. That distinction is not
pedantic in this repo -- an SDPA swap verified at 4e-7 moved a finished ISRUC
run by 0.029 kappa, so numbers from before and after this change are not
comparable and the affected cells have to be re-run.
"""

from __future__ import annotations

import time

import torch

from paclock_bench.models.paclock.frontend.analytic import hilbert, phase_amplitude
from paclock_bench.models.paclock.frontend.triaxial import (
    TriAxialFrontend, _patch_project, patch_pac_vector,
)
from paclock_bench.models.paclock.frontend import _triaxial_conv_ref as ref

dev = "cuda"
torch.manual_seed(0)

B, C, nb, T, d = 32, 16, 8, 1000, 128
PATCH = 200
N = B * C * nb


def timeit(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


print("=" * 74)
print("1. operator: _patch_project(conv, x)  vs  conv(x)")
print("=" * 74)
for name, out_ch, bias in [
    ("tokenizer      (raw)", d, True),
    ("phase_tokenizer     ", d // 2, False),
    ("amplitude_tokenizer ", d // 2, True),
]:
    conv = torch.nn.Conv1d(1, out_ch, PATCH, stride=PATCH, bias=bias).to(dev)
    x = torch.randn(N, T, device=dev)
    want = conv(x.unsqueeze(1)).transpose(1, 2)
    got = _patch_project(conv, x)
    err = (want - got).abs().max().item()
    t_conv = timeit(lambda: conv(x.unsqueeze(1)))
    t_mm = timeit(lambda: _patch_project(conv, x))
    print("  %s  max|d| %.3e  rel %.3e   conv %7.2f ms  gemm %6.2f ms  %6.1fx"
          % (name, err, err / want.abs().max().item(), t_conv, t_mm, t_conv / t_mm))

print()
print("=" * 74)
print("2/3. whole frontend + gradients: new vs pre-change reference")
print("=" * 74)
for mode in ["raw", "pac_interaction"]:
    kw = dict(n_bands=nb, hidden_dim=d, sample_rate=200, kernel_size=201,
              patch_len=PATCH, tokenizer_mode=mode)
    torch.manual_seed(1)
    old = ref.TriAxialFrontend(**kw).to(dev).eval()
    torch.manual_seed(1)
    new = TriAxialFrontend(**kw).to(dev).eval()
    new.load_state_dict(old.state_dict())

    x = torch.randn(B, C, T, device=dev)
    with torch.no_grad():
        o_tok, o_cp = old(x)[:2]
        n_tok, n_cp = new(x)[:2]
    e = (o_tok - n_tok).abs().max().item()
    print("  mode=%-16s tokens %s  max|d| %.3e  rel %.3e  coupling %.3e"
          % (mode, tuple(n_tok.shape), e, e / o_tok.abs().max().item(),
             (o_cp - n_cp).abs().max().item()))

    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    old.zero_grad(); new.zero_grad()
    old(x1)[0].square().sum().backward()
    new(x2)[0].square().sum().backward()
    e_in = (x1.grad - x2.grad).abs().max().item()
    worst, wname = 0.0, ""
    for (na, pa), (_, pb) in zip(old.named_parameters(), new.named_parameters()):
        if pa.grad is not None and pb.grad is not None:
            e2 = (pa.grad - pb.grad).abs().max().item()
            if e2 > worst:
                worst, wname = e2, na
    print("  %-22s d(grad wrt input) %.3e  rel %.3e | worst param grad %.3e (%s)"
          % ("", e_in, e_in / x1.grad.abs().max().item(), worst, wname))

    t_old = timeit(lambda: old(x), n=10, warmup=3)
    t_new = timeit(lambda: new(x), n=10, warmup=3)
    print("  %-22s frontend forward   old %8.2f ms   new %8.2f ms   %5.2fx"
          % ("", t_old, t_new, t_old / t_new))
    print()

print("=" * 74)
print("4. where the remaining frontend time goes (new code, pac_interaction)")
print("=" * 74)
kw = dict(n_bands=nb, hidden_dim=d, sample_rate=200, kernel_size=201,
          patch_len=PATCH, tokenizer_mode="pac_interaction")
fe = TriAxialFrontend(**kw).to(dev).eval()
x = torch.randn(B, C, T, device=dev)

with torch.no_grad():
    filtered = fe.sinc(x.reshape(B * C, 1, T)).reshape(B, C, nb, T)
    z = hilbert(filtered)
    pu, amp = phase_amplitude(z)
    flat = (N, T)

stages = {}
with torch.no_grad():
    stages["sinc filterbank  Conv1d(1,%d,k=201) x%d" % (nb, B * C)] = timeit(
        lambda: fe.sinc(x.reshape(B * C, 1, T)), n=10, warmup=3)
    stages["hilbert  fft+ifft complex64 x%d" % N] = timeit(
        lambda: hilbert(filtered), n=10, warmup=3)
    stages["phase_amplitude  abs + divide"] = timeit(
        lambda: phase_amplitude(z), n=10, warmup=3)
    stages["patch_pac_vector (PAC statistic)"] = timeit(
        lambda: patch_pac_vector(pu, amp, T // PATCH, True), n=10, warmup=3)
    stages["tokenizers x3 (NEW gemm path)"] = timeit(
        lambda: (_patch_project(fe.phase_tokenizer, pu.real.reshape(flat)),
                 _patch_project(fe.phase_tokenizer, pu.imag.reshape(flat)),
                 _patch_project(fe.amplitude_tokenizer,
                                torch.log1p(amp).reshape(flat))), n=10, warmup=3)
    total_fe = timeit(lambda: fe(x), n=10, warmup=3)

acc = sum(stages.values())
for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
    print("  %-46s %8.2f ms  %5.1f%%" % (k, v, v / total_fe * 100))
print("  %-46s %8.2f ms  (stages sum to %.2f ms)" % ("FULL FRONTEND", total_fe, acc))
