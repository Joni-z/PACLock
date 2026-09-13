"""Check joint-content coordinates and trainability inside a GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, os.getcwd())
import torch
from paclock_bench.models.paclock.frontend.factorized import FactorizedFrontend
from paclock_bench.models.paclock.build import TriAxialPACLock
from paclock_bench.training.train import set_seed

assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
common = dict(n_bands=8, hidden_dim=192, sample_rate=200, patch_len=50,
              pac_patch_len=50, interaction_mode="rotation", pac_token_mode="measured")
frontends = {}
for source in ("band", "broadband", "band_broadband"):
    set_seed(0)
    frontends[source] = FactorizedFrontend(content_source=source, **common).cuda()
band, broad, joint = (frontends[k] for k in ("band", "broadband", "band_broadband"))
for other in (broad, joint):
    assert band.state_dict().keys() == other.state_dict().keys()
    for key, value in band.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
x = torch.randn(2, 3, 1000, device="cuda")
for inp in (x, torch.zeros_like(x), torch.full_like(x, 1e-7)):
    with torch.no_grad():
        b, r, j = (fe(inp)[0] for fe in (band, broad, joint))
        assert j.shape == (2, 3, 8, 20, 192) and torch.isfinite(j).all()
        torch.testing.assert_close(j[..., :128], b[..., :128], rtol=0, atol=0)
        torch.testing.assert_close(j[..., 128:160], b[..., 128:160], rtol=0, atol=0)
        torch.testing.assert_close(j[..., 160:], r[..., 160:], rtol=0, atol=0)
    joint.zero_grad(set_to_none=True)
    joint(inp)[0].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in joint.parameters() if p.grad is not None)
try:
    FactorizedFrontend(content_source="band_broadband", **dict(common, hidden_dim=191))
except ValueError as exc:
    assert "even content width" in str(exc)
else:
    raise AssertionError("odd content width was accepted")
cfg = dict(d_model=192, depth=2, n_bands=8, n_heads=4, sample_rate=200,
           n_channels=3, num_classes=3, seq_len=1000, patch_len=50, pac_patch_len=50,
           tokenizer_mode="factorized", content_source="band_broadband", coupling_dim=128,
           interaction_mode="rotation", freq_mixer="attention", spatial_pe="index", band_pe="index")
set_seed(0)
m = TriAxialPACLock(cfg).cuda()
torch.nn.functional.cross_entropy(m(x), torch.tensor([0, 1], device="cuda")).backward()
grad = m.frontend.local_projection.weight.grad
assert grad is not None and torch.isfinite(grad).all()
assert grad[:32].abs().sum() > 0 and grad[32:].abs().sum() > 0
for p in (m.frontend.phase_tokenizer.weight, m.frontend.amplitude_tokenizer.weight):
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
m2 = TriAxialPACLock(cfg).cuda()
m2.load_state_dict(m.state_dict(), strict=True)
m.eval(); m2.eval()
with torch.no_grad():
    torch.testing.assert_close(m(x), m2(x), rtol=0, atol=0)
paths = [__file__, "paclock_bench/models/paclock/frontend/factorized.py"]
receipt = dict(ok=True, job=os.environ["SLURM_JOB_ID"], host=os.uname().nodename,
               git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               coordinate_widths=[128, 32, 32], identical_initial_state=True,
               exact_source_coordinates=True, all_three_lanes_receive_gradients=True,
               finite_random_flat_nearflat=True, strict_checkpoint_roundtrip=True,
               sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
dest = Path("results") / ("factorized-joint-contract-" + os.environ["SLURM_JOB_ID"] + ".json")
dest.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
