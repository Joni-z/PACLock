"""Account for a whole TUEV epoch: training steps vs validation.

The numbers that force this:

  finished cand_base   20 epochs in 38772 s  =  1939 s/epoch
  C_tuev_ctrl (live)    7 epochs in 14183 s  =  2026 s/epoch
  in-loop profile      231 samples/s -> 68445/231 = 296 s of training steps

So ~1700 s of every epoch, 85% of the run, happens somewhere other than the
training steps -- and the only thing left in the loop is the once-per-epoch
validate(). This times the two halves against each other with the real config,
the real loaders and the real model, instead of reasoning about them.
"""

from __future__ import annotations

import time

import torch
import yaml

from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.train import evaluate, build_loss
from paclock_bench.training.metrics import primary_metric

CFG_PATH = "configs/_cand/tuev_base.yaml"
N_STEPS = 2139               # a FULL TUEV epoch
CHUNK = 200

cfg = yaml.safe_load(open(CFG_PATH))
device = "cuda"
train_loader, val_loader, test_loader, info = build_dataloaders(cfg)
model = build_model(cfg, info["input_shape"]).to(device)
criterion = build_loss(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))

n_train = info["n_samples"]["train"]
n_val = len(val_loader.dataset)
n_test = len(test_loader.dataset)
bs = cfg["batch_size"]
steps_per_epoch = -(-n_train // bs)
print("train %d  val %d  test %d  batch %d  -> %d steps/epoch"
      % (n_train, n_val, n_test, bs, steps_per_epoch))
print("val batch_size = %s   val num_workers = %s"
      % (val_loader.batch_size, val_loader.num_workers))
print()

# ---- training steps
model.train()
t0 = mark = None
# A per-chunk rate is the whole point: an average over 200 warm steps cannot
# tell "615 samples/s, steady" from "615 decaying to 34", and those two imply
# completely different fixes. The finished runs average 34, the first 200 steps
# measure 615, so one of them has to be a function of step number.
print("  step   chunk_s   ms/step   samples/s   alloc_GB   resvd_GB")
for step, (X, y) in enumerate(train_loader):
    if step == 10:                       # skip warm-up/loader spin-up
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
        print("  %5d  %8.2f  %8.2f  %10.1f  %9.2f  %9.2f"
              % (step, now - mark, (now - mark) / CHUNK * 1000,
                 CHUNK * bs / (now - mark),
                 torch.cuda.memory_allocated() / 1024 ** 3,
                 torch.cuda.memory_reserved() / 1024 ** 3), flush=True)
        mark = now
    if step + 1 >= N_STEPS:
        break
torch.cuda.synchronize()
n_timed = step - 10
t_steps = time.time() - t0
per_step = t_steps / n_timed
print("TRAIN  %d steps in %.1f s  = %.1f ms/step  = %.1f samples/s"
      % (n_timed, t_steps, per_step * 1000, bs / per_step))
print("       -> a full epoch of steps would be %.0f s" % (per_step * steps_per_epoch))
print()

# ---- one validation, exactly as validate() calls it
torch.cuda.synchronize(); t1 = time.time()
_, m = evaluate(model, val_loader, device, criterion, cfg["num_classes"], cfg)
torch.cuda.synchronize()
t_val = time.time() - t1
print("VAL    %d samples in %.1f s  = %.1f samples/s" % (n_val, t_val, n_val / t_val))
print()

epoch_est = per_step * steps_per_epoch + t_val
print("=" * 68)
print("  estimated epoch = %.0f s steps + %.0f s val = %.0f s"
      % (per_step * steps_per_epoch, t_val, epoch_est))
print("  observed epoch  = 1939 s (cand_base) / 2026 s (C_tuev_ctrl live)")
print("  validation share of the estimate: %.0f%%" % (t_val / epoch_est * 100))
print("=" * 68)
