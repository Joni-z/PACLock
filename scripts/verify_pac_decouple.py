"""pac_patch_len must change nothing when it equals patch_len, and must produce
the documented broadcast when it does not.

The PAC estimation window is the model's central quantity, so a change here has
to be shown inert on the default path rather than assumed inert.
"""
import sys

import torch
import yaml

sys.path.insert(0, ".")
from paclock_bench.models.build import build_model  # noqa: E402

cfg = yaml.safe_load(open("configs/experiments/tuev_paclock_full.yaml"))
shape = (16, 1000)
x = torch.randn(2, *shape)

torch.manual_seed(0); a = build_model(dict(cfg), shape).eval()
c2 = dict(cfg); c2["model_kwargs"] = {**cfg["model_kwargs"], "pac_patch_len": 200}
torch.manual_seed(0); b = build_model(c2, shape).eval()
with torch.no_grad():
    ya, yb = a(x), b(x)
print(f"  default vs pac_patch_len=patch_len(200): max|delta| = "
      f"{(ya - yb).abs().max():.3e}  identical={torch.equal(ya, yb)}")

# token stride and PAC window moved independently
for pl, ppl in ((100, 100), (100, 200), (200, 200), (200, 400)):
    c = dict(cfg)
    c["model_kwargs"] = {**cfg["model_kwargs"], "patch_len": pl, "pac_patch_len": ppl}
    torch.manual_seed(0); m = build_model(c, shape).eval()
    with torch.no_grad():
        y = m(x)
    P, Ppac = 1000 // pl, 1000 // ppl
    print(f"  patch_len={pl:3d} pac_patch_len={ppl:3d} -> {P:2d} tokens, "
          f"{Ppac:2d} PAC windows, out {tuple(y.shape)}")

# the guard fires when the windows do not tile the tokens
try:
    c = dict(cfg)
    c["model_kwargs"] = {**cfg["model_kwargs"], "patch_len": 200, "pac_patch_len": 300}
    build_model(c, shape)(x)
    print("  FAIL: non-dividing pac_patch_len was accepted")
except ValueError as e:
    print(f"  guard ok: {str(e)[:70]}...")
