"""Sustained dataloader rate over a whole epoch, with no model in the loop.

The step profile read only 30 batches, which come out of the prefetch queue and
therefore measure the queue, not the loader. This drains a full epoch.
"""
import sys
import time

import yaml

sys.path.insert(0, ".")
from paclock_bench.data.datasets import build_dataloaders   # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
if len(sys.argv) > 2:
    cfg["num_workers"] = int(sys.argv[2])
train_loader, _, _, info = build_dataloaders(cfg)

bs = cfg["batch_size"]
nw = cfg.get("num_workers", 8)
n = 0
t0 = time.time()
last = t0
slowest = 0.0
for X, y in train_loader:
    n += X.shape[0]
    now = time.time()
    slowest = max(slowest, now - last)
    last = now
dt = time.time() - t0
print(f"  batch={bs} workers={nw}: {n} samples in {dt:.1f}s = {n/dt:.1f} samples/s"
      f"   (slowest single batch {slowest*1000:.0f} ms)")
