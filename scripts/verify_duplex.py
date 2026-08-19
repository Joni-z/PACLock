"""Gates for tokenizer_mode="duplex" and head mode "meanspatial".

  1. raw / pac / hybrid / fused bit-identical to the pre-change snapshot
     (_triaxial_prev4.py): nothing frozen moves.
  2. duplex at init (beta = 0, alpha = 1) is bit-identical to
     hybrid + hybrid_gate under shared weights -- its regression anchor.
  3. duplex worst case: with beta = 0 and alpha driven to 0, the fused rows
     equal raw tokens exactly and the interaction rows are exactly zero.
  4. head "meanspatial" decomposes: with the spatial columns of its projection
     zeroed and the mean columns copied from a mean head, its output equals
     the mean head's output exactly -- the mean path is contained, not
     approximated. Both branches receive gradient.
  5. existing head modes bit-identical to the pre-change head; full duplex +
     meanspatial model builds and forwards.

    sbatch slurm/run.slurm scripts.verify_duplex
"""
import torch

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.models.paclock.frontend._triaxial_prev4 import (
    TriAxialFrontend as PrevFrontend,
)
from paclock_bench.models.paclock.head import ClassificationHead

KW = dict(n_bands=8, hidden_dim=128, sample_rate=200, patch_len=200,
          pac_token_mode="measured", interaction_mode="rotation")
NB, D, C, P, NCLS = 8, 128, 4, 5, 4
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


x = torch.randn(2, C, 1000)

print("=== 1. existing tokenizer modes unchanged", flush=True)
for mode in ("raw", "pac_interaction", "hybrid", "fused"):
    torch.manual_seed(0)
    new = TriAxialFrontend(**KW, tokenizer_mode=mode).eval()
    torch.manual_seed(0)
    old = PrevFrontend(**KW, tokenizer_mode=mode).eval()
    old.load_state_dict(new.state_dict(), strict=True)
    with torch.no_grad():
        d = (new(x)[0] - old(x)[0]).abs().max().item()
    check(f"{mode} bit-identical", d == 0.0, f"max|diff| = {d:.3e}")

print("\n=== 2. duplex at init == hybrid+gate", flush=True)
torch.manual_seed(0)
dx = TriAxialFrontend(**KW, tokenizer_mode="duplex").eval()
hy = TriAxialFrontend(**KW, tokenizer_mode="hybrid", hybrid_gate="band").eval()
hy_sd = {k: v for k, v in dx.state_dict().items() if k in hy.state_dict()}
hy.load_state_dict(hy_sd, strict=True)
with torch.no_grad():
    t_d = dx(x)[0]
    t_h = hy(x)[0]
d = (t_d - t_h).abs().max().item()
check("duplex(beta=0, alpha=1) == hybrid+gate", d == 0.0, f"max|diff| = {d:.3e}")
check("grid is 2nb", t_d.shape[2] == 2 * NB and dx.n_token_bands == 2 * NB)

print("\n=== 3. duplex worst case is raw + zero rows", flush=True)
raw = TriAxialFrontend(**KW, tokenizer_mode="raw").eval()
raw.load_state_dict({k: v for k, v in dx.state_dict().items()
                     if k in raw.state_dict()}, strict=True)
with torch.no_grad():
    dx.interaction_gate.zero_()
    t_w = dx(x)[0]
    t_r = raw(x)[0]
check("fused rows == raw tokens", torch.equal(t_w[:, :, :NB], t_r))
check("interaction rows all zero", t_w[:, :, NB:].abs().max().item() == 0.0)
dx.interaction_gate.data.fill_(1.0)
dx.train()
dx(x)[0].sum().backward()
check("beta and alpha both receive gradient",
      dx.fusion_beta.grad is not None and dx.fusion_beta.grad.abs().sum() > 0
      and dx.interaction_gate.grad is not None
      and dx.interaction_gate.grad.abs().sum() > 0)

print("\n=== 4. meanspatial decomposes onto the mean head", flush=True)
torch.manual_seed(0)
ms = ClassificationHead(D, NCLS, mode="meanspatial", n_bands=NB, n_channels=C).eval()
torch.manual_seed(0)
mh = ClassificationHead(D, NCLS, mode="mean", n_bands=NB, n_channels=C).eval()
# same LayerNorm, mean-columns copied, spatial columns zeroed
ms.norm.load_state_dict(mh.norm.state_dict())
with torch.no_grad():
    ms.proj.weight.zero_()
    ms.proj.weight[:, :D] = mh.proj.weight
    ms.proj.bias.copy_(mh.proj.bias)
xt = torch.randn(3, C * NB * P, D)
with torch.no_grad():
    d = (ms(xt, (C, NB, P)) - mh(xt, (C, NB, P))).abs().max().item()
check("spatial-cols-zeroed meanspatial == mean head", d < 1e-6,
      f"max|diff| = {d:.3e}")
ms.train()
torch.manual_seed(1)
ms2 = ClassificationHead(D, NCLS, mode="meanspatial", n_bands=NB, n_channels=C)
loss = ms2(xt, (C, NB, P)).sum()
loss.backward()
g = ms2.proj.weight.grad
check("both branches get gradient",
      g[:, :D].abs().sum().item() > 0 and g[:, D:].abs().sum().item() > 0)

print("\n=== 5. full duplex + meanspatial model", flush=True)
BASE = dict(model="paclock", num_classes=NCLS, sample_rate=200, dataset="bci_iv_2a",
            model_kwargs=dict(arch="triaxial", d_model=128, depth=2, n_bands=8,
                              n_heads=4, dropout=0.1, kernel_size=201,
                              patch_len=200, pac_patch_len=200,
                              tokenizer_mode="duplex", pac_token_mode="measured",
                              interaction_mode="rotation",
                              freq_mixer="attention", band_pe="index",
                              spatial_pe="index", head="meanspatial"))
torch.manual_seed(0)
m = build_model(BASE, input_shape=(C, 1000)).eval()
with torch.no_grad():
    y = m(torch.randn(2, C, 1000))
check("forward finite", y.shape == (2, NCLS) and torch.isfinite(y).all().item())
check("BandPE sized 2nb", m.band_pe.emb.num_embeddings == 2 * NB)

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
