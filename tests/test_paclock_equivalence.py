"""Prove the AMD refactor did not change the PACLock architecture.

    python -m tests.test_paclock_equivalence

The NVIDIA-cluster original is kept untouched under
``paclock_bench/models/paclock/_reference/``. This test instantiates both it and
the refactored version, copies one state_dict into the other, and asserts the
outputs match to float32 round-off on identical inputs.

This is the only evidence that matters for the "architecture must stay exactly
the same" requirement. If a future edit changes behaviour, this test fails --
not a downstream benchmark number three weeks later.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.is_complex() or b.is_complex():
        return float((a - b).abs().max())
    return float((a - b).abs().max())


DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# 1. hilbert: refactored vs original, same values
# --------------------------------------------------------------------------- #
from paclock_bench.models.paclock._reference.frontend import analytic as ref_analytic
from paclock_bench.models.paclock.frontend import analytic as new_analytic

for T in (2000, 1000, 999):                       # even and odd lengths
    x = torch.randn(4, 3, T, device=DEV)
    zr = ref_analytic.hilbert(x)
    zn = new_analytic.hilbert(x)
    d = max_abs_diff(zr, zn)
    check(f"hilbert T={T} identical", d == 0.0, f"max|diff|={d:.3e}")

# phase/amplitude split
x = torch.randn(2, 4, 2000, device=DEV)
pr, ar = ref_analytic.phase_amplitude(ref_analytic.hilbert(x))
pn, an = new_analytic.phase_amplitude(new_analytic.hilbert(x))
check("phase_unit identical", max_abs_diff(pr, pn) == 0.0)
check("amplitude identical", max_abs_diff(ar, an) == 0.0)

# the AMD-motivated behaviour: FFT must survive an autocast/bf16 region
if DEV == "cuda":
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = new_analytic.hilbert(torch.randn(2, 4, 2000, device=DEV))
        check("hilbert works under bf16 autocast", True, f"dtype={z.dtype}")
    except Exception as e:                                        # noqa: BLE001
        check("hilbert works under bf16 autocast", False, f"{type(e).__name__}: {e}")

    # Record, rather than assert, whether the original also survives autocast.
    # It does on this ROCm build (torch promotes the FFT internally), so the
    # float32 pin is hardening rather than a bug fix -- see analytic.py. Kept
    # as a tripwire: if a future torch/ROCm stops promoting, this line changes
    # to original_ok=False and the pin is what keeps the model running.
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref_analytic.hilbert(torch.randn(2, 4, 2000, device=DEV))
        ref_ok = True
    except Exception:                                             # noqa: BLE001
        ref_ok = False
    check("autocast behaviour of original recorded", True,
          f"original_survives_autocast={ref_ok}")

# --------------------------------------------------------------------------- #
# 2. TriAxialFrontend: full forward, both tokenizer modes
# --------------------------------------------------------------------------- #
from paclock_bench.models.paclock._reference.frontend.triaxial import (
    TriAxialFrontend as RefFrontend,
)
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend as NewFrontend

for tok_mode, pac_mode, inter in [
    ("raw", "measured", "product"),
    ("pac_interaction", "measured", "product"),      # the main architecture
    ("pac_interaction", "uniform", "product"),
    ("pac_interaction", "magnitude", "product"),
    ("pac_interaction", "measured", "concat"),
]:
    kw = dict(n_bands=8, hidden_dim=128, sample_rate=200, kernel_size=201,
              patch_len=200, tokenizer_mode=tok_mode, pac_token_mode=pac_mode,
              interaction_mode=inter)
    torch.manual_seed(1)
    ref = RefFrontend(**kw).to(DEV).eval()
    torch.manual_seed(1)
    new = NewFrontend(**kw).to(DEV).eval()
    new.load_state_dict(ref.state_dict())            # identical weights

    x = torch.randn(2, 4, 2000, device=DEV)
    with torch.no_grad():
        tr, cr, br = ref(x)
        tn, cn, bn = new(x)
    dt = max_abs_diff(tr, tn)
    dc = max_abs_diff(cr, cn)
    db = max_abs_diff(br, bn)
    tag = f"{tok_mode}/{pac_mode}/{inter}"
    check(f"frontend {tag}: token shape", tr.shape == tn.shape, str(tuple(tn.shape)))
    check(f"frontend {tag}: tokens match", dt < 1e-5, f"max|diff|={dt:.3e}")
    check(f"frontend {tag}: coupling match", dc < 1e-6, f"max|diff|={dc:.3e}")
    check(f"frontend {tag}: band_hz match", db == 0.0, f"max|diff|={db:.3e}")

# --------------------------------------------------------------------------- #
# 3. root-band rule: band 0 keeps its own phase feature (the in-place rewrite)
# --------------------------------------------------------------------------- #
kw = dict(n_bands=8, hidden_dim=128, sample_rate=200, kernel_size=201,
          patch_len=200, tokenizer_mode="pac_interaction",
          pac_token_mode="measured", interaction_mode="product")
torch.manual_seed(1)
new = NewFrontend(**kw).to(DEV).eval()
B, C, P, nb, K = 2, 3, 4, 8, 64
phase_feat = torch.randn(B, C, P, nb, K, device=DEV, dtype=torch.complex64)
amp_feat = torch.randn(B, C, P, nb, K, device=DEV)
pac_vec = torch.randn(B, C, P, nb, nb, device=DEV, dtype=torch.complex64)
with torch.no_grad():
    out = new._pac_interaction(phase_feat, amp_feat, pac_vec)
    expect_root = amp_feat[:, :, :, 0, :].to(torch.complex64) * phase_feat[:, :, :, 0, :]
check("root band uses its own phase feature",
      max_abs_diff(out[:, :, :, 0, :], expect_root) < 1e-6,
      f"max|diff|={max_abs_diff(out[:, :, :, 0, :], expect_root):.3e}")

# --------------------------------------------------------------------------- #
# 4. full model forward + backward
# --------------------------------------------------------------------------- #
from paclock_bench.models.paclock._reference.build import PACLock as RefPACLock
from paclock_bench.models.paclock.build import PACLock as NewPACLock

cfg = dict(
    arch="triaxial", mixer="attention", freq_mixer="attention", spatial_pe="index",
    tokenizer_mode="pac_interaction", pac_token_mode="measured",
    interaction_mode="product",
    n_channels=16, seq_len=2000, sample_rate=200, sampling_rate=200,
    num_classes=2, n_bands=8, d_model=128, depth=6, dropout=0.0,
    kernel_size=201, patch_len=200, n_heads=4, augmentations=[], dataset="tuab",
)
torch.manual_seed(2)
refm = RefPACLock(cfg).to(DEV).eval()
torch.manual_seed(2)
newm = NewPACLock(cfg).to(DEV).eval()
newm.load_state_dict(refm.state_dict())

n_ref = sum(p.numel() for p in refm.parameters())
n_new = sum(p.numel() for p in newm.parameters())
check("param count identical", n_ref == n_new, f"{n_ref} vs {n_new} ({n_new/1e6:.2f}M)")

x = torch.randn(3, 16, 2000, device=DEV)
with torch.no_grad():
    lr = refm(x)
    ln = newm(x)
d = max_abs_diff(lr, ln)
check("full model logits match", d < 1e-4, f"max|diff|={d:.3e}")

# gradients must match too -- a forward-only match can hide a backward change
refm.train(); newm.train()
torch.manual_seed(3)
xg = torch.randn(3, 16, 2000, device=DEV)
yg = torch.randint(0, 2, (3,), device=DEV)
lossf = torch.nn.CrossEntropyLoss()

refm.zero_grad(); lossf(refm(xg), yg).backward()
newm.zero_grad(); lossf(newm(xg), yg).backward()
gd, worst = 0.0, ""
for (nr, pr_), (nn_, pn_) in zip(refm.named_parameters(), newm.named_parameters()):
    if pr_.grad is None and pn_.grad is None:
        continue
    if pr_.grad is None or pn_.grad is None:
        gd, worst = float("inf"), f"{nr}: one grad is None"
        break
    d_ = max_abs_diff(pr_.grad, pn_.grad)
    scale = max(float(pr_.grad.abs().max()), 1e-12)
    if d_ / scale > gd:
        gd, worst = d_ / scale, f"{nr} rel={d_/scale:.2e}"
check("gradients match", gd < 1e-3, f"worst rel diff: {worst}")

# --------------------------------------------------------------------------- #
print("=" * 78)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
print("=" * 78)
nfail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - nfail}/{len(results)} passed  [device={DEV}]")
sys.exit(1 if nfail else 0)
