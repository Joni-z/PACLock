import sys, os, yaml, torch
sys.path.insert(0, os.getcwd())
from paclock_bench.models.paclock.build import build_model as build_paclock
cfg = yaml.safe_load(open("configs/pretrain/axisfree_rawpatch.yaml"))
mk = dict(cfg["model_kwargs"]); mk["spatial_pe"] = "index"
pac = {**mk, "num_classes": 2, "n_channels": 16, "seq_len": 1, "sample_rate": 200, "dataset": "pretrain"}
m = build_paclock(pac).cuda(); n = sum(p.numel() for p in m.parameters()) / 1e6
x = torch.randn(8, 16, 2000, device="cuda") * 0.5
loss = m.crossfreq_aux_loss(x); loss.backward()
print("raw_patch loss %.4f finite=%s params %.2fM" % (loss.item(), bool(torch.isfinite(loss)), n), flush=True)
# leakage check: the prediction at a masked cell must not depend on that cell's own raw samples
torch.manual_seed(0); m.eval()
B, C, T = 2, 16, 2000; L = m.frontend.patch_len; P = T // L
x = torch.randn(B, C, T, device="cuda") * 0.5
def pred_with_mask(x, cell):
    xm = x.reshape(B, C, P, L).masked_fill(cell.unsqueeze(-1), 0.0).reshape(B, C, T)
    tokens, coupling, band_hz = m.frontend(xm)[:3]; nb = tokens.shape[2]; D = tokens.shape[-1]
    mm = cell.unsqueeze(2).expand(B, C, nb, P)
    tok = torch.where(mm.unsqueeze(-1), m.mask_token.view(1,1,1,1,D), tokens) + m.band_pe(band_hz).view(1,1,nb,1,D) + m.spatial_pe(C, x.device).view(1,C,1,1,D)
    h = m.encoder(tok, coupling * (~cell).to(coupling.dtype).view(B,C,P,1,1), None)
    return m.recon_raw(h.mean(dim=2))
cell = torch.zeros(B, C, P, dtype=torch.bool, device="cuda"); cell[:, :, ::2] = True
with torch.no_grad():
    p1 = pred_with_mask(x, cell)
    x2 = x.clone().reshape(B, C, P, L); x2[cell] = torch.randn_like(x2[cell]) * 0.5; x2 = x2.reshape(B, C, T)   # change ONLY masked cells' samples
    p2 = pred_with_mask(x2, cell)
print("change of predictions at masked cells when only masked cells' raw samples change: %.3e (must be 0)" % (p1[cell] - p2[cell]).abs().max().item(), flush=True)
print("patch_len", L, "cells per window", P, "SMOKE_OK")
