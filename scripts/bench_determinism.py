"""A/B the two flags set_seed() turns on, on the real TUEV config.

train.py calls set_seed() before anything else, and set_seed does:

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

On ROCm those map onto MIOpen. Nothing else separates the finished runs (34
samples/s) from an identical bench that never calls set_seed (613 samples/s,
flat across a whole epoch, constant memory) -- same config file, same loaders,
same model, same node type, same batch.

The same flag also explains why D_repeat's three identical configs produced
bit-identical loss curves. Ordinary GPU training is not bit-reproducible,
because reduction order in atomics is not fixed; forced determinism is exactly
what makes it so. The reproducibility finding and the speed problem are the
same fact seen twice.

Reported per 200-step chunk so a warm-up artefact cannot be mistaken for a rate.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import yaml

from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.train import build_loss

MODE = os.environ.get("DET_MODE", "off")
N_STEPS = 810
CHUNK = 200

# seed identically in both arms; the ONLY difference is the two backend flags
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
if MODE == "on":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif MODE == "off":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
elif MODE == "bench":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

cfg = yaml.safe_load(open("configs/_cand/tuev_base.yaml"))
device = "cuda"
train_loader, _, _, info = build_dataloaders(cfg)
model = build_model(cfg, info["input_shape"]).to(device)
criterion = build_loss(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))
bs = cfg["batch_size"]

print("DET_MODE=%s   deterministic=%s benchmark=%s"
      % (MODE, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark),
      flush=True)

model.train()
t0 = mark = None
rates = []
for step, (X, y) in enumerate(train_loader):
    if step == 10:
        torch.cuda.synchronize(); t0 = mark = time.time()
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()
    loss = criterion(model(X).float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step > 10 and (step - 10) % CHUNK == 0:
        torch.cuda.synchronize()
        now = time.time()
        r = CHUNK * bs / (now - mark)
        rates.append(r)
        print("  step %5d   %7.2f ms/step   %8.1f samples/s"
              % (step, (now - mark) / CHUNK * 1000, r), flush=True)
        mark = now
    if step + 1 >= N_STEPS:
        break

torch.cuda.synchronize()
per_step = (time.time() - t0) / (step - 10)
print("DET_MODE=%s  ->  %.1f ms/step  %.1f samples/s   full TUEV epoch = %.0f s"
      % (MODE, per_step * 1000, bs / per_step, per_step * 2139), flush=True)
