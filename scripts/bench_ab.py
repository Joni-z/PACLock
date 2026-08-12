"""2x2: old Conv1d tokeniser vs new GEMM tokeniser, deterministic on vs off,
timing a FULL training step (forward + backward + optimiser).

Why this and not the earlier benchmarks. The frontend A/B measured 1.05x, but it
timed the forward only, and the forward was never the problem: what the Conv1d
tokenisers cost is the WEIGHT-gradient convolution in the backward pass, over
4096 single-channel signals -- and set_seed() forces
``torch.backends.cudnn.deterministic = True``, so MIOpen has to pick an
atomics-free backward-weights algorithm for exactly that shape. The determinism
A/B measured only 1.29x because by then the convolutions had already been
replaced, so there was nothing left for the flag to be slow on.

The two changes are not independent, and every benchmark so far has varied one
while the other was already in its cheap state. This varies both.

Ground truth to land against: the finished TUEV runs are 1939 s/epoch, and the
current code measures ~200 s/epoch on the same config and node type.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend import _triaxial_conv_ref as ref
from paclock_bench.training.train import build_loss

DEV = "cuda"
B, C, T = 32, 16, 1000
STEPS_PER_EPOCH = 2139
cfg = yaml.safe_load(open("configs/_cand/tuev_base.yaml"))
cfg["num_classes"] = 6
mk = cfg["model_kwargs"]


def make(frontend_kind):
    random.seed(0); np.random.seed(0)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = build_model(cfg, (C, T)).to(DEV)
    if frontend_kind == "conv":
        old = ref.TriAxialFrontend(
            n_bands=mk["n_bands"], hidden_dim=mk["d_model"],
            sample_rate=cfg["sample_rate"], kernel_size=mk["kernel_size"],
            patch_len=mk["patch_len"], tokenizer_mode=mk["tokenizer_mode"],
            pac_token_mode=mk["pac_token_mode"],
            interaction_mode=mk["interaction_mode"],
        ).to(DEV)
        old.load_state_dict(model.frontend.state_dict())
        model.frontend = old
    return model


def timed_step(model, n=15, warmup=5):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = build_loss(cfg)
    x = torch.randn(B, C, T, device=DEV)
    y = torch.randint(0, 6, (B,), device=DEV)
    model.train()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x).float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


print("%-22s %-14s %12s %14s %14s"
      % ("tokenizer", "deterministic", "ms/step", "samples/s", "epoch (s)"))
print("-" * 80)
results = {}
for det in [True, False]:
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = False
    for kind in ["conv", "gemm"]:
        model = make(kind)
        ms = timed_step(model)
        results[(kind, det)] = ms
        print("%-22s %-14s %12.2f %14.1f %14.0f"
              % ("Conv1d (old)" if kind == "conv" else "GEMM (new)",
                 str(det), ms, B / ms * 1000, ms / 1000 * STEPS_PER_EPOCH))
        del model
        torch.cuda.empty_cache()

print("-" * 80)
print("  speedup from the tokeniser change, deterministic ON : %.2fx"
      % (results[("conv", True)] / results[("gemm", True)]))
print("  speedup from the tokeniser change, deterministic OFF: %.2fx"
      % (results[("conv", False)] / results[("gemm", False)]))
print("  speedup from dropping determinism, old tokeniser    : %.2fx"
      % (results[("conv", True)] / results[("conv", False)]))
print("  combined (old+det) -> (new+nondet)                  : %.2fx"
      % (results[("conv", True)] / results[("gemm", False)]))
