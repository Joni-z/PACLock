"""Where PACLock's 1.6M parameters actually sit.

Two questions depend on this. Whether 1.6M leaves room for pretraining to help
depends on how much of it is the part pretraining would touch; and whether we
need a separate tokeniser-pretraining stage depends on how much of a *learned*
tokeniser there is to pretrain at all.
"""

from __future__ import annotations

import torch

from paclock_bench.models.build import build_model

CFG = {
    "model": "paclock", "dataset": "tuev", "num_classes": 6, "sample_rate": 200,
    "model_kwargs": {
        "arch": "triaxial", "d_model": 128, "depth": 6, "n_bands": 8,
        "n_heads": 4, "dropout": 0.2, "kernel_size": 201, "patch_len": 50,
        "pac_patch_len": 50, "augmentations": [], "freq_mixer": "attention",
        "band_pe": "index", "tokenizer_mode": "pac_interaction",
        "pac_token_mode": "measured", "interaction_mode": "product",
        "spatial_pe": "index",
    },
}

for tag, over in [("shipped 1.6M", {}),
                  ("large (d256/depth8)", {"d_model": 256, "depth": 8, "n_heads": 8})]:
    cfg = {**CFG, "model_kwargs": {**CFG["model_kwargs"], **over}}
    m = build_model(cfg, (16, 1000))
    tot = sum(p.numel() for p in m.parameters())
    print("=" * 66)
    print("%s   total %.3f M" % (tag, tot / 1e6))
    print("=" * 66)
    groups = {}
    for n, p in m.named_parameters():
        top = n.split(".")[0]
        groups.setdefault(top, 0)
        groups[top] += p.numel()
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print("  %-16s %9d  %5.1f%%" % (k, v, 100 * v / tot))
    # inside the frontend, separate the physics from the learned tokenisers
    fe = {}
    for n, p in m.frontend.named_parameters():
        fe.setdefault(n.split(".")[0], 0)
        fe[n.split(".")[0]] += p.numel()
    print("  frontend detail:")
    for k, v in sorted(fe.items(), key=lambda kv: -kv[1]):
        print("      %-20s %8d" % (k, v))
    print()
