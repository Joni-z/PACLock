"""Correctness gates for interaction_mode="rotation".

  1. product/concat are bit-identical to the implementation from before the
     change (_triaxial_prev.py). Every frozen config names one of those two, so
     drift here would silently invalidate the finished TUEV/TUSZ/CHB-MIT cells.
  2. THE DEFINING PROPERTY: |h_jk| == |a_jk| for every complex feature. This is
     the whole point -- band power reaches the encoder unscaled -- and it is what
     `product` violates, by a measured factor whose CV is 0.75.
  3. the phase is still entirely the coupling's: rotating aligned_phase by any
     per-band angle changes h. A "rotation" that discarded the phase would pass
     test 2 trivially and be a plain amplitude tokenizer, i.e. the method thrown
     away, so this is the test that separates the two.
  4. gauge invariance, the method's central claim: under a phase-reference shift
     p_i -> exp(i delta_i) p_i, which also sends Z_ij -> exp(i delta_i) Z_ij, the
     tokens of bands 1.. must not move. Checked on `rotation` AND on `product`
     (it should hold for both; if it fails for product the claim in the docstring
     was never true). Band 0 legitimately moves: it keeps its own phase by
     construction, so it carries the gauge.
  5. no NaN/Inf on a dead electrode, where |aligned_phase| can approach zero.
  6. the noise being removed, on real data: CV of |aligned_phase| across patches
     (what `product` multiplies amplitude by) vs CV under `rotation` (exactly 0
     by construction). Reported per corpus so the effect size is on record.

    sbatch slurm/run.slurm scripts.verify_rotation
"""
import numpy as np
import torch

from paclock_bench.models.paclock.frontend.triaxial import (
    TriAxialFrontend, patch_pac_vector,
)
from paclock_bench.models.paclock.frontend._triaxial_prev import (
    TriAxialFrontend as PrevFrontend,
)
from paclock_bench.models.paclock.frontend.analytic import hilbert, phase_amplitude
from paclock_bench.paths import expand

KW = dict(n_bands=8, hidden_dim=128, sample_rate=200, patch_len=200,
          tokenizer_mode="pac_interaction", pac_token_mode="measured")
NB, K = 8, 64
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


def make(mode, **over):
    torch.manual_seed(0)
    return TriAxialFrontend(**KW, interaction_mode=mode, **over).eval()


def prev(mode):
    torch.manual_seed(0)
    return PrevFrontend(**KW, interaction_mode=mode).eval()


x = torch.randn(2, 4, 1000)

print("=== 1. product / concat unchanged vs pre-change implementation", flush=True)
for mode in ("product", "concat"):
    new, old = make(mode), prev(mode)
    old.load_state_dict(new.state_dict(), strict=True)
    with torch.no_grad():
        d = (new(x)[0] - old(x)[0]).abs().max().item()
    check(f"{mode} bit-identical", d == 0.0, f"max|diff| = {d:.3e}")

# shared ingredients for the interaction-level tests
rot = make("rotation")
with torch.no_grad():
    B, C, T = x.shape
    filt = rot.sinc(x.reshape(B * C, 1, T)).reshape(B, C, NB, T)
    ph, am = phase_amplitude(hilbert(filt))
    pv = patch_pac_vector(ph, am, T // 200, True)
    P = T // 200
    torch.manual_seed(1)
    phase_feat = torch.randn(B, C, P, NB, K, dtype=torch.cfloat)
    amp_feat = torch.randn(B, C, P, NB, K)

print("\n=== 2. |h| == |a|: band power passes through unscaled", flush=True)
with torch.no_grad():
    h_rot = rot._pac_interaction(phase_feat, amp_feat, pv)
    h_prod = make("product")._pac_interaction(phase_feat, amp_feat, pv)
err = (h_rot.abs() - amp_feat.abs()).abs().max().item()
check("rotation preserves amplitude modulus", err < 1e-5, f"max|diff| = {err:.3e}")
gain = (h_prod.abs() / amp_feat.abs().clamp_min(1e-12))
check("product does NOT (this is the defect)", gain.std().item() > 1e-3,
      f"gain mean {gain.mean():.3f}  std {gain.std():.3f}")

print("\n=== 3. the phase is still the coupling's", flush=True)
with torch.no_grad():
    twist = torch.exp(1j * torch.rand(1, 1, 1, NB, 1) * 6.283).to(phase_feat.dtype)
    h_twisted = rot._pac_interaction(phase_feat * twist, amp_feat, pv)
moved = (h_twisted - h_rot).abs().max().item()
check("phase content retained (not amplitude-only)", moved > 1e-3,
      f"max|change| = {moved:.3e}")

print("\n=== 4. gauge invariance: p_i -> e^{i d_i} p_i, Z_ij -> e^{i d_i} Z_ij",
      flush=True)
with torch.no_grad():
    delta = torch.rand(NB) * 6.283
    rot_i = torch.exp(1j * delta)
    pf_g = phase_feat * rot_i.view(1, 1, 1, NB, 1).to(phase_feat.dtype)
    pv_g = pv * rot_i.view(1, 1, 1, NB, 1).to(pv.dtype)     # Z[..., i, j], i axis -2
    for mode in ("rotation", "product"):
        m = make(mode)
        a = m._pac_interaction(phase_feat, amp_feat, pv)
        b = m._pac_interaction(pf_g, amp_feat, pv_g)
        d = (a[..., 1:, :] - b[..., 1:, :]).abs().max().item()
        scale = a[..., 1:, :].abs().max().item()
        check(f"{mode}: bands 1.. invariant", d / max(scale, 1e-12) < 1e-4,
              f"rel = {d / max(scale, 1e-12):.3e}")

print("\n=== 5. dead electrode", flush=True)
x_flat = x.clone()
x_flat[:, 0, :] = 0.0
with torch.no_grad():
    t = rot(x_flat)[0]
check("no NaN/Inf", torch.isfinite(t).all().item())

print("\n=== 6. the noise removed, on real data", flush=True)
for ds in ("tuev", "bci_iv_2a"):
    root = expand(f"$PACLOCK_PROC/processed/{ds}")
    X = np.load(f"{root}/train_signals.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(X.shape[0], min(64, X.shape[0]), replace=False))
    xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32))
    with torch.no_grad():
        B, C, T = xb.shape
        f = rot.sinc(xb.reshape(B * C, 1, T)).reshape(B, C, NB, T)
        p_, a_ = phase_amplitude(hilbert(f))
        Pp = T // 200
        pvr = patch_pac_vector(p_, a_, Pp, True)
        pfr = torch.randn(B, C, Pp, NB, K, dtype=torch.cfloat)
        afr = torch.ones(B, C, Pp, NB, K)
        car = rot._pac_interaction(pfr, afr, pvr).abs()      # == 1 by construction
        prd = make("product")._pac_interaction(pfr, afr, pvr).abs()
    print(f"  {ds:>10}:  |carrier| under product  mean {prd.mean():.3f}  "
          f"CV {(prd.std() / prd.mean()):.3f}   |  under rotation  mean "
          f"{car.mean():.3f}  CV {(car.std() / car.mean()):.3e}")

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
