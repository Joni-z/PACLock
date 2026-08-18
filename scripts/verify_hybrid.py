"""Gates for tokenizer_mode="hybrid".

The hybrid claim is compositional -- "raw rows are exactly the raw tokenizer,
interaction rows are exactly the PAC tokenizer, side by side" -- so that is
what gets asserted, row by row, under shared weights:

  1. raw / pac_interaction modes are bit-identical to the pre-change
     implementation (the snapshot in frontend/_triaxial_prev2.py): no frozen
     result moves.
  2. hybrid rows 0..nb-1  == a raw-mode frontend's tokens, bit for bit, when
     the raw tokenizer weights are copied over.
  3. hybrid rows nb..2nb-1 == a pac-mode frontend's tokens, bit for bit, when
     the phase/amplitude tokenizer weights are copied over.
  4. token_band_hz has 2*nb rows and n_token_bands reports 2*nb; the full
     TriAxialPACLock builds, sizes BandPE(index) at 2*nb, forwards finitely,
     and works with head mean AND spatial.
  5. the guards fire: hybrid + freq_mixer=coupling raises; hybrid +
     return_amp_target raises (pretraining leak); hybrid + aux_recon raises.
  6. parameter accounting: hybrid == raw + pac tokenizer params on the
     frontend, and the full model's delta over raw mode is those tokenizer
     params plus n_bands extra BandPE(index) embeddings -- stated, since this
     is a candidate deliverable, not a parameter-matched ablation arm.

    sbatch slurm/run.slurm scripts.verify_hybrid
"""
import torch

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.models.paclock.frontend._triaxial_prev2 import (
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

print("=== 1. raw / pac_interaction unchanged vs pre-hybrid snapshot", flush=True)
for mode in ("raw", "pac_interaction"):
    torch.manual_seed(0)
    new = TriAxialFrontend(**KW, tokenizer_mode=mode).eval()
    torch.manual_seed(0)
    old = PrevFrontend(**KW, tokenizer_mode=mode).eval()
    old.load_state_dict(new.state_dict(), strict=True)
    with torch.no_grad():
        d = (new(x)[0] - old(x)[0]).abs().max().item()
    check(f"{mode} bit-identical", d == 0.0, f"max|diff| = {d:.3e}")

print("\n=== 2+3. hybrid decomposes exactly", flush=True)
torch.manual_seed(0)
hy = TriAxialFrontend(**KW, tokenizer_mode="hybrid").eval()
raw = TriAxialFrontend(**KW, tokenizer_mode="raw").eval()
pac = TriAxialFrontend(**KW, tokenizer_mode="pac_interaction").eval()
# copy the SHARED weights out of the hybrid model into the single-mode ones,
# so all three compute with identical parameters
hy_sd = hy.state_dict()
raw.load_state_dict({k: v for k, v in hy_sd.items() if k in raw.state_dict()},
                    strict=True)
pac.load_state_dict({k: v for k, v in hy_sd.items() if k in pac.state_dict()},
                    strict=True)
with torch.no_grad():
    t_h = hy(x)[0]
    t_r = raw(x)[0]
    t_p = pac(x)[0]
check("grid is (B,C,2nb,P,D)", t_h.shape[2] == 2 * NB, str(tuple(t_h.shape)))
d_raw = (t_h[:, :, :NB] - t_r).abs().max().item()
check("rows 0..nb-1 == raw tokens", d_raw == 0.0, f"max|diff| = {d_raw:.3e}")
d_pac = (t_h[:, :, NB:] - t_p).abs().max().item()
check("rows nb..2nb-1 == pac tokens", d_pac == 0.0, f"max|diff| = {d_pac:.3e}")
check("n_token_bands = 2nb", hy.n_token_bands == 2 * NB)
check("token_band_hz has 2nb rows", hy.token_band_hz().shape[0] == 2 * NB)
check("single modes report nb", raw.n_token_bands == NB and pac.n_token_bands == NB)

print("\n=== 4. full model builds and forwards", flush=True)
BASE = dict(model="paclock", num_classes=4, sample_rate=200, dataset="bci_iv_2a",
            model_kwargs=dict(arch="triaxial", d_model=128, depth=2, n_bands=8,
                              n_heads=4, dropout=0.1, kernel_size=201,
                              patch_len=200, pac_patch_len=200,
                              tokenizer_mode="hybrid", pac_token_mode="measured",
                              interaction_mode="rotation", freq_mixer="attention",
                              band_pe="index", spatial_pe="index"))
for head in ("mean", "spatial"):
    torch.manual_seed(0)
    cfg = {**BASE, "model_kwargs": {**BASE["model_kwargs"], "head": head}}
    m = build_model(cfg, input_shape=(4, 1000)).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 4, 1000))
    check(f"head={head}: output finite, shape {tuple(y.shape)}",
          y.shape == (2, 4) and torch.isfinite(y).all().item())
check("BandPE(index) sized 2nb", m.band_pe.emb.num_embeddings == 2 * NB)

print("\n=== 5. guards", flush=True)
try:
    build_model({**BASE, "model_kwargs": {**BASE["model_kwargs"],
                                          "freq_mixer": "coupling"}},
                input_shape=(4, 1000))
    check("hybrid + coupling mixer raises", False)
except ValueError as e:
    check("hybrid + coupling mixer raises", True, str(e)[:60])
try:
    hy(x, return_amp_target=True)
    check("hybrid + return_amp_target raises", False)
except NotImplementedError as e:
    check("hybrid + return_amp_target raises", True, str(e)[:60])
try:
    build_model({**BASE, "model_kwargs": {**BASE["model_kwargs"],
                                          "aux_recon_weight": 0.5}},
                input_shape=(4, 1000))
    check("hybrid + aux_recon raises", False)
except ValueError as e:
    check("hybrid + aux_recon raises", True, str(e)[:60])

print("\n=== 6. parameter accounting", flush=True)
def n(m):
    return sum(p.numel() for p in m.parameters())
tok_raw = sum(p.numel() for k, p in raw.named_parameters() if "tokenizer" in k)
tok_pac = sum(p.numel() for k, p in pac.named_parameters()
              if "tokenizer" in k or "amplitude_scale" in k)
check("frontend: hybrid = raw + pac tokenizer params",
      n(hy) == n(raw) + tok_pac and n(hy) == n(pac) + tok_raw,
      f"hybrid {n(hy):,} = raw {n(raw):,} + {tok_pac:,} = pac {n(pac):,} + {tok_raw:,}")

print("\n=== 7. per-band interaction gate", flush=True)
torch.manual_seed(0)
gated = TriAxialFrontend(**KW, tokenizer_mode="hybrid", hybrid_gate="band").eval()
gated.load_state_dict({**hy.state_dict(),
                       "interaction_gate": torch.ones(NB)}, strict=True)
with torch.no_grad():
    t_g = gated(x)[0]
d = (t_g - t_h).abs().max().item()
check("gate=1 bit-identical to plain hybrid", d == 0.0, f"max|diff| = {d:.3e}")
with torch.no_grad():
    gated.interaction_gate[3] = 0.0
    t_g0 = gated(x)[0]
check("zeroing gate band 3 kills only interaction row 3",
      t_g0[:, :, NB + 3].abs().max().item() == 0.0
      and torch.equal(t_g0[:, :, :NB], t_h[:, :, :NB])
      and torch.equal(t_g0[:, :, NB:NB + 3], t_h[:, :, NB:NB + 3]))
gated.interaction_gate.data.fill_(1.0)
gated.train()
gated(x)[0].sum().backward()
g = gated.interaction_gate.grad
check("gate receives gradient", g is not None and g.abs().sum().item() > 0)
try:
    TriAxialFrontend(**KW, tokenizer_mode="raw", hybrid_gate="band")
    check("gate without hybrid raises", False)
except ValueError as e:
    check("gate without hybrid raises", True, str(e)[:50])

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
