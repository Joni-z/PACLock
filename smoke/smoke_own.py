import sys, os, yaml, torch
sys.path.insert(0, os.getcwd())
from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
dev = "cuda"
# 1) build + forward/backward of the control config
cfg = yaml.safe_load(open("configs/cf2/tuev_cf2_v1d192own.yaml")); cfg["device"] = dev
m = build_model(cfg, (16, 1000)).to(dev); x = torch.randn(4, 16, 1000, device=dev) * 0.5
y = m(x); y.float().sum().backward(); print("own config: out", tuple(y.shape), "params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6), "finite", bool(torch.isfinite(y).all()), flush=True)
# 2) intervention test on the frontend: perturb slower bands' phase, check band j's interaction token
def fe(mode):
    torch.manual_seed(0)
    return TriAxialFrontend(n_bands=8, hidden_dim=192, sample_rate=200, kernel_size=201, patch_len=200, pac_patch_len=200,
                            tokenizer_mode="duplex", pac_token_mode=mode, interaction_mode="rotation").to(dev).eval()
torch.manual_seed(1); B, C, nb, T, P = 2, 3, 8, 1000, 5
phase = torch.randn(B, C, nb, T, device=dev, dtype=torch.cfloat); phase = phase / phase.abs()
amp = torch.rand(B, C, nb, T, device=dev) + 0.1
from paclock_bench.models.paclock.frontend.triaxial import patch_pac_vector
def tokens(f, ph):
    pv = patch_pac_vector(ph, amp, P, True)
    with torch.no_grad(): return f._interaction_tokens(ph, amp, pv)   # (B,C,nb,P,D)
ph2 = phase.clone(); ph2[:, :, :4] = ph2[:, :, :4] * torch.exp(1j * torch.rand(B, C, 4, T, device=dev) * 6.28)  # scramble bands 0-3 phases in time
for mode in ["measured", "own"]:
    f = fe(mode); a = tokens(f, phase); b = tokens(f, ph2)
    d_high = (a[:, :, 4:] - b[:, :, 4:]).abs().max().item()   # bands 4-7 depend on bands 0-3 only through alignment
    d_low = (a[:, :, :4] - b[:, :, :4]).abs().max().item()
    print(f"{mode:9s}: max change in bands 4-7 when bands 0-3 phases are perturbed = {d_high:.3e}; bands 0-3 themselves = {d_low:.3e}", flush=True)
print("SMOKE_OK")
