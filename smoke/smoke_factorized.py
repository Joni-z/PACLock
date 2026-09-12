"""GPU contract tests and real training-batch timing for the factorized pilot."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.getcwd())
import torch
import yaml
from smoke.filter_response import check_low_band_response
from paclock_bench.models.paclock.frontend.factorized import FactorizedFrontend
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend, _patch_project
from paclock_bench.models.paclock.build import TriAxialPACLock
from paclock_bench.training.train import set_seed
from paclock_bench.models.build import build_model
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.training.losses import build_loss

assert torch.cuda.is_available()
set_seed(0)
common = dict(n_bands=8, sample_rate=200, patch_len=50, pac_patch_len=50,
              interaction_mode="rotation", pac_token_mode="measured")
set_seed(0)
old = TriAxialFrontend(hidden_dim=128, tokenizer_mode="pac_interaction", **common).cuda()
set_seed(0)
new = FactorizedFrontend(hidden_dim=192, **common).cuda()
x = torch.randn(2, 3, 1000, device="cuda")
with torch.no_grad():
    a, b = old(x)[0], new(x)[0]
    torch.testing.assert_close(a, b[..., :128], rtol=0, atol=0)
    filtered = new.sinc(x.reshape(6, 1, 1000))
    local = _patch_project(new.local_projection, filtered.reshape(48, 1000))
    torch.testing.assert_close(local.reshape(2, 3, 8, 20, 64), b[..., 128:], rtol=0, atol=0)
    assert b.shape == (2, 3, 8, 20, 192)
    assert sum(p.numel() for p in new.parameters()) > sum(p.numel() for p in old.parameters())
print("PASS exact legacy coupling coordinates, exact waveform coordinates, unchanged row count", flush=True)

for source in ("band", "broadband", "zero"):
    set_seed(0)
    fe = FactorizedFrontend(hidden_dim=192, content_source=source, **common).cuda()
    for inp in (x, torch.zeros_like(x), torch.full_like(x, 1e-7)):
        fe.zero_grad(set_to_none=True)
        out = fe(inp)[0]
        assert torch.isfinite(out).all()
        if source == "zero":
            assert out[..., 128:].count_nonzero().item() == 0
        out.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in fe.parameters() if p.grad is not None)
    del fe
print("PASS finite forward/backward on random, flat and near-flat inputs for all content controls", flush=True)

wide = FactorizedFrontend(hidden_dim=192, filter_range=[.5, 75.0], **common).cuda()
bh = wide.band_hz()
torch.testing.assert_close(bh[0, 0] - bh[0, 1]/2, torch.tensor(.5, device="cuda"))
torch.testing.assert_close(bh[-1, 0] + bh[-1, 1]/2, torch.tensor(75., device="cuda"))
assert torch.isfinite(wide(x)[0]).all()
# The previous cutoff-only assertion missed the f3 DC-dominated filter.
check_low_band_response(wide)
# Both halves must learn on the first optimizer step; zero-init gates would
# block one half's gradient and do not satisfy this test.
model_cfg = dict(d_model=192, n_bands=8, sample_rate=200, patch_len=50,
                 n_channels=3, num_classes=3, seq_len=1000, depth=2,
                 freq_mixer="attention", tokenizer_mode="factorized", coupling_dim=128,
                 interaction_mode="rotation", band_pe="index", spatial_pe="index")
m = TriAxialPACLock(model_cfg).cuda()
torch.nn.functional.cross_entropy(m(x), torch.tensor([0, 1], device="cuda")).backward()
for p in (m.frontend.local_projection.weight, m.frontend.phase_tokenizer.weight,
          m.frontend.amplitude_tokenizer.weight):
    assert p.grad is not None and p.grad.abs().sum() > 0
state = m.state_dict()
m2 = TriAxialPACLock(model_cfg).cuda()
m2.load_state_dict(state, strict=True)
m.eval(); m2.eval()
with torch.no_grad():
    torch.testing.assert_close(m(x), m2(x), rtol=0, atol=0)
del m, m2, old, new, wide, a, b, state
gc.collect(); torch.cuda.empty_cache()
print("PASS effective frequency edges, both lanes learn, strict checkpoint round trip", flush=True)

report = []
for ds in ("tuev", "chbmit", "tusz", "tuar", "sleepedf", "isruc"):
    cfg = yaml.safe_load(Path(f"configs/factorized/{ds}_f1.yaml").read_text())
    cfg["num_workers"] = 0
    train, val, test, info = build_dataloaders(cfg)
    xx, yy = next(iter(train))
    del train, val, test
    xx, yy = xx.cuda(), yy.cuda().long()
    # Full configured batch, actual montage, loss and sequence handling.
    for arm in ("f0", "f1", "f2", "f3"):
        cfg = yaml.safe_load(Path(f"configs/factorized/{ds}_{arm}.yaml").read_text())
        set_seed(0)
        model = build_model(cfg, info["input_shape"]).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = build_loss(cfg)
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for step in range(4):
            torch.cuda.synchronize(); started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = criterion(logits.float(), yy)
            assert torch.isfinite(loss), (ds, arm, step)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            torch.cuda.synchronize()
            if step >= 2: elapsed.append(time.perf_counter() - started)
        record = dict(dataset=ds, arm=arm, batch=list(xx.shape),
                      params_M=sum(p.numel() for p in model.parameters())/1e6,
                      seconds_per_step=sum(elapsed)/len(elapsed),
                      peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                      loss=float(loss.detach()))
        report.append(record)
        print(json.dumps(record), flush=True)
        del model, opt, criterion, logits, loss
        gc.collect(); torch.cuda.empty_cache()
    del xx, yy

paths = ["paclock_bench/models/paclock/frontend/factorized.py", "paclock_bench/models/paclock/build.py",
         "paclock_bench/models/paclock/frontend/triaxial.py",
         "paclock_bench/models/paclock/frontend/sinc.py",
         "paclock_bench/models/paclock/frontend/analytic.py",
         "paclock_bench/models/paclock/triaxial.py",
         "paclock_bench/models/paclock/head.py",
         "paclock_bench/models/build.py", "paclock_bench/training/train.py",
         "scripts/launch_factorized.py", "slurm/configs_packed.slurm",
         "smoke/smoke_factorized.py", "smoke/filter_response.py"] + sorted(str(p) for p in Path("configs/factorized").glob("*.yaml"))
hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
result = dict(ok=True, job_id=os.environ.get("SLURM_JOB_ID"),
              step_id=os.environ.get("SLURM_STEP_ID"), sha256=hashes, timing=report)
dest = Path("results/factorized_smoke.json")
dest.parent.mkdir(exist_ok=True)
tmp = dest.with_suffix(".tmp")
tmp.write_text(json.dumps(result, indent=2)); tmp.replace(dest)
print("FACTORIZED_SMOKE_OK", flush=True)
