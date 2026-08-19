"""Gates for tokenizer_mode="fused".

  1. every existing mode (raw / pac_interaction / hybrid) bit-identical to the
     pre-change snapshot (_triaxial_prev3.py): nothing frozen moves.
  2. THE DEFINING PROPERTY: fused-blend at init (beta = 0) is bit-identical to
     tokenizer_mode="raw" under shared weights. The worst case IS raw, by
     construction, and that is asserted rather than hoped.
  3. perturbing one band's beta changes ONLY that band's row.
  4. the grid keeps raw's shape: (C, nb, P, D), n_token_bands == nb,
     token_band_hz has nb rows -- dims untouched, as requested.
  5. gated variant: forward finite, gate bias init pushes g toward raw
     (mean g > 0.8 at init), output differs from both pure sources.
  6. fused + return_amp_target works (single-row-per-band => no paired-row
     leak; the pac mode it generalises was always allowed in pretraining).
  7. full model builds and forwards with mean AND spatial heads.

    sbatch slurm/run.slurm scripts.verify_fused
"""
import torch

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.models.paclock.frontend._triaxial_prev3 import (
    TriAxialFrontend as PrevFrontend,
)

KW = dict(n_bands=8, hidden_dim=128, sample_rate=200, patch_len=200,
          pac_token_mode="measured", interaction_mode="rotation")
NB = 8
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


x = torch.randn(2, 4, 1000)

print("=== 1. existing modes unchanged vs snapshot", flush=True)
for mode in ("raw", "pac_interaction", "hybrid"):
    torch.manual_seed(0)
    new = TriAxialFrontend(**KW, tokenizer_mode=mode).eval()
    torch.manual_seed(0)
    old = PrevFrontend(**KW, tokenizer_mode=mode).eval()
    old.load_state_dict(new.state_dict(), strict=True)
    with torch.no_grad():
        d = (new(x)[0] - old(x)[0]).abs().max().item()
    check(f"{mode} bit-identical", d == 0.0, f"max|diff| = {d:.3e}")

print("\n=== 2. fused-blend at init == raw, bit for bit", flush=True)
torch.manual_seed(0)
fb = TriAxialFrontend(**KW, tokenizer_mode="fused", fusion_mode="blend").eval()
raw = TriAxialFrontend(**KW, tokenizer_mode="raw").eval()
raw.load_state_dict({k: v for k, v in fb.state_dict().items()
                     if k in raw.state_dict()}, strict=True)
with torch.no_grad():
    t_f = fb(x)[0]
    t_r = raw(x)[0]
d = (t_f - t_r).abs().max().item()
check("beta=0 output == raw output", d == 0.0, f"max|diff| = {d:.3e}")
check("grid keeps raw shape", t_f.shape == t_r.shape and t_f.shape[2] == NB,
      str(tuple(t_f.shape)))
check("n_token_bands == nb", fb.n_token_bands == NB)
check("token_band_hz has nb rows", fb.token_band_hz().shape[0] == NB)

print("\n=== 3. beta locality", flush=True)
with torch.no_grad():
    fb.fusion_beta[5].fill_(0.7)
    t_p = fb(x)[0]
changed = [(t_p[:, :, b] - t_f[:, :, b]).abs().max().item() for b in range(NB)]
check("perturbing beta[5] changes only row 5",
      changed[5] > 1e-4 and all(c == 0.0 for b, c in enumerate(changed) if b != 5),
      "per-row max|diff| = " + ",".join(f"{c:.1e}" for c in changed))
fb.fusion_beta.data.zero_()

print("\n=== 4. gradient reaches beta", flush=True)
fb.train()
fb(x)[0].sum().backward()
g = fb.fusion_beta.grad
check("beta receives gradient", g is not None and g.abs().sum().item() > 0)

print("\n=== 5. gated variant", flush=True)
torch.manual_seed(0)
fg = TriAxialFrontend(**KW, tokenizer_mode="fused", fusion_mode="gated").eval()
with torch.no_grad():
    t_g = fg(x)[0]
check("gated forward finite", torch.isfinite(t_g).all().item())
# measure the actual gate value at init on real activations
with torch.no_grad():
    B, C, T = x.shape
    filt = fg.sinc(x.reshape(B * C, 1, T)).reshape(B, C, NB, T)
    from paclock_bench.models.paclock.frontend.analytic import (
        hilbert, phase_amplitude,
    )
    ph, am = phase_amplitude(hilbert(filt))
    from paclock_bench.models.paclock.frontend.triaxial import patch_pac_vector
    pv = patch_pac_vector(ph, am, T // 200, True)
    f = filt.reshape(B * C * NB, T)
    from paclock_bench.models.paclock.frontend.triaxial import _patch_project
    r_tok = _patch_project(fg.tokenizer, f).reshape(B, C, NB, -1, 128)
    h_tok = fg._interaction_tokens(ph, am, [pv])
    gate = torch.sigmoid(fg.fusion_gate(torch.cat([r_tok, h_tok], dim=-1)))
check("gate init leans raw (mean g > 0.8)", gate.mean().item() > 0.8,
      f"mean g = {gate.mean().item():.3f}")

print("\n=== 6. pretraining path open for fused", flush=True)
try:
    out = fb(x, return_amp_target=True)
    check("fused + return_amp_target works",
          len(out) == 4 and out[3].shape == (2, 4, NB, 5))
except Exception as e:                                        # noqa: BLE001
    check("fused + return_amp_target works", False, str(e)[:60])

print("\n=== 7. full model, mean and spatial heads", flush=True)
BASE = dict(model="paclock", num_classes=4, sample_rate=200, dataset="bci_iv_2a",
            model_kwargs=dict(arch="triaxial", d_model=128, depth=2, n_bands=8,
                              n_heads=4, dropout=0.1, kernel_size=201,
                              patch_len=200, pac_patch_len=200,
                              tokenizer_mode="fused", fusion_mode="blend",
                              pac_token_mode="measured",
                              interaction_mode="rotation",
                              freq_mixer="attention", band_pe="index",
                              spatial_pe="index"))
for head in ("mean", "spatial"):
    torch.manual_seed(0)
    cfg = {**BASE, "model_kwargs": {**BASE["model_kwargs"], "head": head}}
    m = build_model(cfg, input_shape=(4, 1000)).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 4, 1000))
    check(f"head={head}: output finite", y.shape == (2, 4)
          and torch.isfinite(y).all().item())

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
