"""Inspect gradient/activation statistics at init on the two collapsing
corpora, without a full training run.

    sbatch slurm/run.slurm scripts.diag_gradients

FACED/PhysioNet-MI both sit dead flat near ln(K) train loss for the first
several epochs (FACED) or the whole run (PhysioNet-MI seed2), while the same
recipe trains normally on lower-channel-count corpora. This checks the most
direct mechanical explanation: does gradient reaching the frontend vanish (or
explode) specifically at these corpora's channel/token counts, at both
patch_len=50 (current) and patch_len=200 (pre-search default)?
"""

from __future__ import annotations

import torch
import torch.nn as nn

from paclock_bench.models.build import build_model

CASES = [
    ("faced", 32, 2000, 9),
    ("physionet_mi", 64, 640, 4),
    ("tuev", 16, 1000, 6),          # healthy control
    ("isruc", 6, 6000, 5),          # healthy control
]


def make_cfg(patch_len, C, T, K):
    return {
        "model": "paclock", "dataset": "diag", "num_classes": K,
        "sample_rate": 200,
        "model_kwargs": {
            "arch": "triaxial", "d_model": 128, "depth": 6, "n_bands": 8,
            "n_heads": 4, "dropout": 0.2, "kernel_size": 201,
            "patch_len": patch_len, "pac_patch_len": patch_len,
            "augmentations": [], "freq_mixer": "attention", "band_pe": "index",
            "tokenizer_mode": "pac_interaction", "pac_token_mode": "measured",
            "interaction_mode": "product", "spatial_pe": "index",
        },
    }


def grad_report(model, x, y, n_tokens):
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = nn.functional.cross_entropy(out, y, label_smoothing=0.1)
    loss.backward()

    groups = {"frontend": model.frontend, "band_pe": model.band_pe,
              "spatial_pe": model.spatial_pe, "encoder": model.encoder,
              "head": model.head}
    stats = {}
    for name, mod in groups.items():
        norms = [p.grad.norm().item() for p in mod.parameters()
                 if p.grad is not None]
        stats[name] = (sum(norms), len(norms)) if norms else (0.0, 0)

    return dict(loss=loss.item(), logit_std=out.std().item(),
                logit_absmax=out.abs().max().item(), n_tokens=n_tokens,
                grad=stats)


torch.manual_seed(0)
print("%-13s %-6s %-10s %-8s %-9s %-9s %-38s" % (
    "dataset", "patch", "tokens", "loss", "logit_sd", "logit_max", "grad-norm-sum by group"))
print("-" * 110)

for name, C, T, K in CASES:
    for patch_len in [200, 50]:
        cfg = make_cfg(patch_len, C, T, K)
        torch.manual_seed(0)
        model = build_model(cfg, (C, T))
        model.train()
        B = 8
        x = torch.randn(B, C, T)
        y = torch.randint(0, K, (B,))
        n_bands, P = 8, T // patch_len
        n_tok = C * n_bands * P
        r = grad_report(model, x, y, n_tok)
        gtxt = "  ".join("%s=%.2e(%d)" % (k, v[0], v[1]) for k, v in r["grad"].items())
        print("%-13s %-6d %-10d %-8.4f %-9.4f %-9.4f %s" % (
            name, patch_len, n_tok, r["loss"], r["logit_std"], r["logit_absmax"], gtxt))
    print()
