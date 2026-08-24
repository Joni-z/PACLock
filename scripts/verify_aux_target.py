"""Gates for the pretraining-objective fix (aux_target modes).

  1. default aux_target="amp" is bit-identical to the pre-change snapshot
     (_build_prev2.py) -- raw and duplex, both mask modes.
  2. band_norm: the standardized target really is ~N(0,1) per band, and the
     loss trains.
  3. band_norm_pac: coupling head exists and gets gradient; the loss is
     deterministic; with EVERY band masked the coupling term contributes
     exactly zero (no visible phase band => no supervised entry => the
     leak-exclusion mask works).

    sbatch slurm/run.slurm scripts.verify_aux_target
"""
import torch
import torch.nn.functional as F

from paclock_bench.models.paclock.build import TriAxialPACLock
from paclock_bench.models.paclock._build_prev2 import TriAxialPACLock as PrevModel

C, T, NB, D = 4, 1000, 8, 128
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


def cfg_for(mode, **extra):
    c = dict(arch="triaxial", d_model=D, depth=2, n_bands=NB, n_heads=4,
             dropout=0.1, kernel_size=201, patch_len=200, pac_patch_len=200,
             sample_rate=200, seq_len=T, n_channels=C, num_classes=2,
             dataset="tusz", tokenizer_mode=mode, pac_token_mode="measured",
             interaction_mode="rotation", freq_mixer="attention",
             band_pe="index", spatial_pe="index", head="mean",
             aux_recon_weight=1.0)
    c.update(extra)
    return c


x = torch.randn(2, C, T)

print("=== 1. default 'amp' bit-identical to snapshot", flush=True)
for mode in ("raw", "duplex"):
    for mm in ("crossfreq", "random"):
        torch.manual_seed(0)
        new = TriAxialPACLock(cfg_for(mode, aux_mask_mode=mm)).eval()
        torch.manual_seed(0)
        old = PrevModel(cfg_for(mode, aux_mask_mode=mm)).eval()
        old.load_state_dict(new.state_dict(), strict=True)
        torch.manual_seed(7)
        ln = new.crossfreq_aux_loss(x)
        torch.manual_seed(7)
        lo = old.crossfreq_aux_loss(x)
        d = (ln - lo).abs().item()
        check(f"{mode}/{mm} identical", d == 0.0, f"|diff| = {d:.3e}")

print("\n=== 2. band_norm: target is standardized per band", flush=True)
torch.manual_seed(0)
m = TriAxialPACLock(cfg_for("duplex", aux_target="band_norm")).eval()
with torch.no_grad():
    _, _, _, at = m.frontend(x, return_amp_target=True)
    mu = at.mean(dim=(0, 1, 3), keepdim=True)
    sd = at.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-6)
    z = (at - mu) / sd
    pm = z.mean(dim=(0, 1, 3)).abs().max().item()
    ps = (z.std(dim=(0, 1, 3)) - 1).abs().max().item()
check("per-band mean ~ 0", pm < 1e-5, f"max|mean| = {pm:.2e}")
check("per-band std ~ 1", ps < 1e-3, f"max|std-1| = {ps:.2e}")
m.train()
loss = m.crossfreq_aux_loss(x)
check("loss finite", torch.isfinite(loss).item(), f"{loss.item():.4f}")
loss.backward()
check("recon head gets grad", m.recon[0].weight.grad.abs().sum().item() > 0)

print("\n=== 3. band_norm_pac", flush=True)
torch.manual_seed(0)
m = TriAxialPACLock(cfg_for("duplex", aux_target="band_norm_pac"))
check("coupling head exists", hasattr(m, "recon_pac"))
m.train()
loss = m.crossfreq_aux_loss(x)
check("loss finite", torch.isfinite(loss).item(), f"{loss.item():.4f}")
loss.backward()
check("recon_pac gets grad", m.recon_pac[0].weight.grad.abs().sum().item() > 0)
check("fusion_beta gets grad", m.frontend.fusion_beta.grad.abs().sum().item() > 0)
m.eval()
torch.manual_seed(3)
l1 = m.crossfreq_aux_loss(x)
torch.manual_seed(3)
l2 = m.crossfreq_aux_loss(x)
check("deterministic target", (l1 - l2).abs().item() == 0.0)

# every band masked => no visible phase band => coupling term must be 0
m.aux_mask_mode = "random"
m.aux_mask_ratio = 1.1                       # rand < 1.1 always True
torch.manual_seed(5)
l_pac = m.crossfreq_aux_loss(x)
m.aux_pac_weight = 0.0
torch.manual_seed(5)
l_nopac = m.crossfreq_aux_loss(x)
d = (l_pac - l_nopac).abs().item()
check("all-masked -> coupling term exactly 0", d == 0.0, f"|diff| = {d:.3e}")

print("\n=== 4. hybrid/pac_interaction still healthy under band_norm_pac", flush=True)
for mode in ("pac_interaction", "hybrid"):
    torch.manual_seed(0)
    kw = dict(aux_target="band_norm_pac")
    if mode == "hybrid":
        kw["hybrid_gate"] = "band"
    mm = TriAxialPACLock(cfg_for(mode, **kw))
    mm.train()
    l = mm.crossfreq_aux_loss(x)
    check(f"{mode} loss finite", torch.isfinite(l).item(), f"{l.item():.4f}")

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
