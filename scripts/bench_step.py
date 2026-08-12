"""Time one full PACLock training step in isolation and compare it to the step
time the finished runs actually achieved.

The finished TUEV runs are unambiguous: 68445 train samples at batch 32 is 2139
steps per epoch, 20 epochs is 42780 steps, and cand_base took 10.77 h = 38772 s.
That is 906 ms per step.

Isolated, the frontend forward measures 2.41 ms and the encoder lands near 16 ms
by the size_large ladder. Two percent of 906 ms is accounted for. This script
exists to find the other 98%, and it does the one thing the earlier profiling
did not: it synchronises around every phase, so an async queue cannot hide the
cost somewhere it did not happen.

Everything runs on random tensors already resident on the GPU, so a difference
between this number and 906 ms is by construction NOT the model -- it is the
loop, the loader, or the host.
"""

from __future__ import annotations

import time

import torch

from paclock_bench.models.build import build_model

DEV = "cuda"
B, C, T = 32, 16, 1000
CFG = {
    "model": "paclock",
    "dataset": "tuev",
    "num_classes": 6,
    "sample_rate": 200,
    "model_kwargs": {
        "arch": "triaxial", "d_model": 128, "depth": 6, "n_bands": 8,
        "n_heads": 4, "dropout": 0.2, "kernel_size": 201, "patch_len": 200,
        "augmentations": [], "freq_mixer": "attention", "band_pe": "index",
        "tokenizer_mode": "pac_interaction", "pac_token_mode": "measured",
        "interaction_mode": "product", "spatial_pe": "index",
    },
}


def sync_time(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


model = build_model(CFG, (C, T)).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

x = torch.randn(B, C, T, device=DEV)
y = torch.randint(0, 6, (B,), device=DEV)

n_par = sum(p.numel() for p in model.parameters()) / 1e6
print("params %.2fM   input (%d, %d, %d)   batch %d" % (n_par, B, C, T, B))
print()

model.train()


def full_step():
    opt.zero_grad(set_to_none=True)
    loss = lossf(model(x), y)
    loss.backward()
    opt.step()


def fwd_only():
    model(x)


t_full = sync_time(full_step, n=20)
t_fwd = sync_time(fwd_only, n=20)

print("=" * 66)
print("  full train step (fwd+bwd+opt)   %8.2f ms   -> %7.1f samples/s"
      % (t_full, B / t_full * 1000))
print("  forward only                    %8.2f ms" % t_fwd)
print("=" * 66)
print("  finished cand_base run          %8.2f ms   -> %7.1f samples/s"
      % (906.0, 35.3))
print("  unexplained factor              %8.1fx" % (906.0 / t_full))
print()

# module-level forward breakdown, each synchronised
sub = {}
with torch.no_grad():
    tok, coupling, hz = model.frontend(x)[:3]
    Bs, Cs, nbs, Ps, Ds = tok.shape
    tok_pe = tok + model.band_pe(hz).view(1, 1, nbs, 1, Ds)
    tok_pe = tok_pe + model.spatial_pe(Cs, tok.device).view(1, Cs, 1, 1, Ds)
    h = model.encoder(tok_pe, coupling, None)
    sub["frontend"] = sync_time(lambda: model.frontend(x), n=20)
    sub["encoder"] = sync_time(lambda: model.encoder(tok_pe, coupling, None), n=20)
    sub["head"] = sync_time(
        lambda: model.head(h.reshape(Bs, Cs * nbs * Ps, Ds), (Cs, nbs, Ps)), n=20)
print("forward breakdown (no_grad):")
for k, v in sorted(sub.items(), key=lambda kv: -kv[1]):
    print("  %-24s %8.2f ms  %5.1f%% of the fwd-only number" % (k, v, v / t_fwd * 100))
print("  %-24s %8.2f ms" % ("(sum)", sum(sub.values())))

# The loop's own per-step cost, with nothing model-shaped in it at all.
print()
print("dataloader check: a real epoch of TUEV is %d steps; the finished run took"
      " %.0f ms/step" % (68445 // B, 906.0))
