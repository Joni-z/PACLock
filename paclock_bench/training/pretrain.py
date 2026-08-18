"""PACLock self-supervised pretraining: masked cross-frequency reconstruction,
pooled across every corpus this benchmark has already preprocessed.

    python -m paclock_bench.training.pretrain --config <config.yaml>

Reuses TriAxialPACLock.crossfreq_aux_loss verbatim -- the same masked
band-amplitude reconstruction objective already built and validated as a
supervised auxiliary (models/paclock/build.py). Pretraining is that objective
run alone, with no classification head and no labels, over every listed
corpus's *train* split with labels discarded.

Why one training step reads one corpus at a time
--------------------------------------------------
`patch_len` sets the kernel size of the tokenizer's Conv1d, which is a weight
SHAPE, not a runtime argument -- one model instance cannot serve two different
patch_len values. So every corpus in the pool is resampled through the same
fixed `patch_len`, and corpora cannot be mixed within a batch (channel counts
differ), only across steps. Each step samples one corpus (weighted by its
training-split size, so a big corpus is not swamped by many small ones) and
draws one batch from it.

Why per-corpus loss is logged separately, not just pooled
------------------------------------------------------------
The whole reason patch_len is a live question is that FACED (32ch) and
PhysioNet-MI (64ch) collapsed under `patch_len=50` supervised finetuning --
loss frozen at the chance floor for most seeds (docs/FINDINGS.md). That
was cross-entropy against a chance floor of ln(K); this objective is MSE
against a moving target, which has no equivalent floor to name in advance.
A pooled/averaged loss would hide a single corpus flatlining behind the
others still descending, which is exactly the failure this run exists to
catch early -- so every corpus's running loss is printed on its own line.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import yaml

from ..data.datasets import WindowDataset
from ..paths import expand
from .train import set_seed


def build_corpus_loaders(cfg: dict):
    """One (name, DataLoader, n_samples) per pretraining corpus."""
    out = []
    for entry in cfg["corpora"]:
        root = expand(entry["data_root"])
        flatten = bool(entry.get("flatten_sequences", False))
        ds = WindowDataset(root, "train", flatten_sequences=flatten)
        bs = entry.get("batch_size", cfg.get("batch_size", 32))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=cfg.get("num_workers", 4),
            pin_memory=True, drop_last=True, persistent_workers=True,
        )
        out.append((entry["name"], loader, len(ds), ds.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="pretrain_runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda")

    corpora = build_corpus_loaders(cfg)
    print("pretraining corpora:")
    for name, loader, n, shape in corpora:
        print("  %-14s n=%-8d shape=%s batch=%d" % (name, n, shape, loader.batch_size))
    weights = np.array([n for _, _, n, _ in corpora], dtype=np.float64)
    weights /= weights.sum()

    from ..models.paclock.build import build_model as build_paclock

    # SpatialPE(index mode) allocates an nn.Embedding sized to n_channels; every
    # corpus in the pool indexes into it with its own real channel count at
    # forward time, so it has to be sized to the LARGEST channel count in the
    # pool, not any single corpus's. xyz mode is not an option here -- montage
    # coordinates are corpus-specific and this pool mixes montages.
    max_channels = max(shape[0] for _, _, _, shape in corpora)
    mk = dict(cfg["model_kwargs"])
    mk["aux_recon_weight"] = 1.0          # forces mask_token + recon head to exist
    mk.setdefault("aux_mask_mode", "random")
    mk.setdefault("aux_mask_ratio", 0.5)
    mk["spatial_pe"] = "index"
    pac_cfg = {**mk, "num_classes": 2, "n_channels": max_channels, "seq_len": 1,
              "sample_rate": cfg.get("sample_rate", 200), "dataset": "pretrain"}
    model = build_paclock(pac_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("model: %.3fM params  patch_len=%s d_model=%s depth=%s" % (
        n_params, mk["patch_len"], mk["d_model"], mk["depth"]))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4),
                            weight_decay=cfg.get("weight_decay", 0.01))
    steps = cfg["steps"]
    warmup = cfg.get("warmup_steps", steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup)) *
        0.5 * (1 + np.cos(np.pi * max(0, s - warmup) / max(1, steps - warmup))))

    iters = [iter(loader) for _, loader, _, _ in corpora]
    running = {name: [] for name, _, _, _ in corpora}
    os.makedirs(os.path.join(args.out, cfg["name"]), exist_ok=True)
    ckpt_path = os.path.join(args.out, cfg["name"], "checkpoint.pt")

    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        ci = np.random.choice(len(corpora), p=weights)
        name, loader, _, _ = corpora[ci]
        try:
            x, _ = next(iters[ci])
        except StopIteration:
            iters[ci] = iter(loader)
            x, _ = next(iters[ci])
        x = x.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        loss = model.crossfreq_aux_loss(x)
        loss.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        running[name].append(loss.item())
        log_every = cfg.get("log_every_steps", 50)
        if step % log_every == 0:
            elapsed = time.time() - t0
            parts = []
            for nm, _, _, _ in corpora:
                v = running[nm]
                parts.append("%s=%.4f(%d)" % (nm, np.mean(v) if v else float("nan"), len(v)))
                running[nm] = []
            print("step %6d/%d  %5.1fs  lr=%.2e  %s" % (
                step, steps, elapsed, sched.get_last_lr()[0], "  ".join(parts)), flush=True)
            t0 = time.time()

        save_every = cfg.get("save_every_steps", steps)
        if step % save_every == 0 or step == steps:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                      ckpt_path)
            print("  -> saved %s (step %d)" % (ckpt_path, step), flush=True)

    print("pretraining done -> %s" % ckpt_path)


if __name__ == "__main__":
    main()
