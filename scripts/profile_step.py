"""Time the training step in pieces, because two rounds of inferring the
bottleneck from aggregate throughput got it wrong.

Round 1 blamed kernel-launch overhead (SDPA, bf16): no effect.
Round 2 blamed random reads over Lustre (preload to RAM): no effect.
Throughput sat at ~31 samples/s through all eight arms. So measure directly.
"""
import sys
import time

import torch
import yaml

sys.path.insert(0, ".")
from paclock_bench.data.datasets import build_dataloaders   # noqa: E402
from paclock_bench.models.build import build_model          # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
dev = "cuda"

train_loader, _, _, info = build_dataloaders(cfg)
model = build_model(cfg, info["input_shape"]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()
bs = cfg["batch_size"]
print(f"batch={bs} input={info['input_shape']}", flush=True)


def sync():
    torch.cuda.synchronize()


# (a) dataloader alone
it = iter(train_loader)
batches = []
sync(); t = time.time()
for _ in range(N):
    batches.append(next(it))
sync(); t_load = time.time() - t

# warm-up so MIOpen autotuning is not billed to the measurement
X, y = batches[0]
X, y = X.to(dev), y.to(dev).long()
for _ in range(3):
    loss = crit(model(X).float(), y); loss.backward(); opt.zero_grad()
sync()

# (b) host-to-device
gpu = []
t = time.time()
for X, y in batches:
    gpu.append((X.to(dev, non_blocking=True), y.to(dev, non_blocking=True).long()))
sync(); t_h2d = time.time() - t

# (c) forward only
t = time.time()
with torch.no_grad():
    for X, y in gpu:
        model(X)
sync(); t_fwd = time.time() - t

# (d) forward + backward + step
t = time.time()
for X, y in gpu:
    loss = crit(model(X).float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
sync(); t_full = time.time() - t

n = N * bs
print(f"\n  {N} batches, {n} samples")
for name, dt in (("dataload", t_load), ("host->device", t_h2d),
                 ("forward", t_fwd), ("fwd+bwd+step", t_full)):
    print(f"  {name:14s} {dt:7.2f}s   {n/dt:9.1f} samples/s")
print(f"\n  measured end-to-end ceiling = {n/(t_load+t_full):.1f} samples/s")
