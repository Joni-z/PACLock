"""Gates for paired-row masked-amplitude pretraining on hybrid/duplex grids.

  1. raw / pac_interaction: crossfreq_aux_loss bit-identical to the pre-change
     implementation (_build_prev.py) under shared weights, both mask modes --
     nothing frozen moves.
  2. duplex: the old guards are gone; the loss is finite and every new
     parameter (mask_token, recon, fusion_beta, interaction_gate) receives
     gradient through it.
  3. THE LEAK TEST: a forward hook on the encoder captures what actually goes
     in. For every masked physical band, BOTH its fused row j and its
     interaction row nb+j must carry mask_token (+PEs) at masked positions;
     the visible low-half rows must not.
  4. duplex at init == hybrid+gate under shared weights: the aux losses agree
     exactly (same grids, same targets -- regression anchor for the pair).

    (CPU-friendly; run inside any allocation)
    python -m scripts.verify_duplex_pretrain
"""
import torch

from paclock_bench.models.paclock.build import TriAxialPACLock
from paclock_bench.models.paclock._build_prev import TriAxialPACLock as PrevModel

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

print("=== 1. unpaired modes bit-identical to snapshot", flush=True)
for mode in ("raw", "pac_interaction"):
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
        check(f"{mode}/{mm} aux loss identical", d == 0.0, f"|diff| = {d:.3e}")

print("\n=== 2. duplex aux loss exists and trains", flush=True)
torch.manual_seed(0)
dx = TriAxialPACLock(cfg_for("duplex"))
dx.train()
loss = dx.crossfreq_aux_loss(x)
check("loss finite", torch.isfinite(loss).item(), f"loss = {loss.item():.4f}")
loss.backward()
g = lambda t: t.grad is not None and t.grad.abs().sum().item() > 0
check("mask_token gets grad", g(dx.mask_token))
check("recon head gets grad", g(dx.recon[0].weight))
check("fusion_beta gets grad", g(dx.frontend.fusion_beta))
check("interaction_gate gets grad", g(dx.frontend.interaction_gate))
torch.manual_seed(0)
hy = TriAxialPACLock(cfg_for("hybrid", hybrid_gate="band"))
hy.train()
lh = hy.crossfreq_aux_loss(x)
check("hybrid loss finite too", torch.isfinite(lh).item(), f"loss = {lh.item():.4f}")

print("\n=== 3. leak test: both rows of a masked band are hidden", flush=True)
torch.manual_seed(0)
dx = TriAxialPACLock(cfg_for("duplex")).eval()
captured = {}
def grab(module, args):
    captured["tok"] = args[0].detach().clone()
h = dx.encoder.register_forward_pre_hook(grab)
with torch.no_grad():
    dx.crossfreq_aux_loss(x)
h.remove()
tok = captured["tok"]                                      # (B,C,2NB,P,Dm)
B, _, nbg, P, Dm = tok.shape
check("grid entering encoder is 2nb", nbg == 2 * NB)
with torch.no_grad():
    _, _, band_hz, _ = dx.frontend(x, return_amp_target=True)
    pe = dx.band_pe(band_hz).view(1, 1, nbg, 1, Dm)
    sp = dx.spatial_pe(C, x.device).view(1, C, 1, 1, Dm)
    content = tok - pe - sp                                # undo the PEs
    mt = dx.mask_token.view(1, 1, 1, 1, Dm)
    hi = slice(NB // 2, NB)                                # masked physical bands
    d_fused = (content[:, :, hi] - mt).abs().max().item()
    d_inter = (content[:, :, NB + NB // 2: 2 * NB] - mt).abs().max().item()
    lo_is_masked = (content[:, :, :NB // 2] - mt).abs().max().item()
check("masked bands' FUSED rows are mask_token", d_fused < 1e-5,
      f"max|diff| = {d_fused:.3e}")
check("masked bands' INTERACTION rows are mask_token", d_inter < 1e-5,
      f"max|diff| = {d_inter:.3e}")
check("visible rows are NOT mask_token", lo_is_masked > 1e-3)

print("\n=== 4. duplex at init == hybrid+gate (aux loss)", flush=True)
torch.manual_seed(0)
dx = TriAxialPACLock(cfg_for("duplex")).eval()
hy = TriAxialPACLock(cfg_for("hybrid", hybrid_gate="band")).eval()
hy_sd = hy.state_dict()
hy.load_state_dict({k: v for k, v in dx.state_dict().items() if k in hy_sd},
                   strict=True)
torch.manual_seed(11)
ld = dx.crossfreq_aux_loss(x)
torch.manual_seed(11)
lh = hy.crossfreq_aux_loss(x)
d = (ld - lh).abs().item()
check("losses identical at init", d == 0.0, f"|diff| = {d:.3e}")

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
