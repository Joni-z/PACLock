"""CBraMod masked-patch pretraining on our pool, for the native model and for
CBraMod-with-CroFreMo-tokenizer under one identical loop (2026-09-10).

    python -m paclock_bench.training.pretrain_cbramod --config <cfg.yaml> [--out DIR] [--resume]

Objective, masking and optimizer follow vendor/cbramod/pretrain_trainer.py: MSE on the
masked (electrode, patch) cells, Bernoulli(0.5) mask, AdamW lr 5e-4 wd 5e-2, cosine to
1e-5, grad clip 1. Inputs are our processed arrays (already in units of 100 uV, the same
scaling as CBraMod's x/100). One corpus per step, corpora weighted by size, as in
training/pretrain.py. Checkpoints are saved atomically every `save_every_steps` and at
the end, as {"model": export_state_dict(), "step", "cfg"}.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml

from ..data.datasets import WindowDataset
from ..paths import expand
from .train import set_seed
from ..models.foundation.cbramod_adapter import PATCH
from ..models.foundation.cbramod_pretrain import build_mpm, generate_mask


def build_corpus_loaders(cfg):
    out = []
    for entry in cfg["corpora"]:
        ds = WindowDataset(expand(entry["data_root"]), "train",
                           flatten_sequences=bool(entry.get("flatten_sequences", False)))
        bs = entry.get("batch_size", cfg.get("batch_size", 128))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=cfg.get("num_workers", 8),
            pin_memory=True, drop_last=True, persistent_workers=True)
        out.append((entry["name"], loader, len(ds), ds.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="pretrain_runs_cbramod")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override cfg steps (probe runs)")
    ap.add_argument("--dp", action="store_true", help="DataParallel over all visible GPUs (amd: 4 x MI210 per exclusive node)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.steps:
        cfg["steps"] = args.steps
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda")
    corpora = build_corpus_loaders(cfg)
    for name, loader, n, shape in corpora:
        print("  %-12s n=%-8d shape=%s batch=%d" % (name, n, shape, loader.batch_size), flush=True)
    weights = np.array([n for _, _, n, _ in corpora], dtype=np.float64); weights /= weights.sum()

    model = build_mpm(cfg["kind"], **(cfg.get("model_kwargs") or {})).to(device)
    core = model
    if args.dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model); print("DataParallel over %d GPUs" % torch.cuda.device_count(), flush=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("model: kind=%s %.2fM params" % (cfg["kind"], n_params), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-4), weight_decay=cfg.get("weight_decay", 5e-2))
    steps = int(cfg["steps"]); warmup = int(cfg.get("warmup_steps", 0)); eta_min = cfg.get("eta_min", 1e-5)
    lr0 = cfg.get("lr", 5e-4)
    def lr_at(s):
        if s < warmup:
            return (s + 1) / warmup
        t = (s - warmup) / max(1, steps - warmup)
        return (eta_min + (lr0 - eta_min) * 0.5 * (1 + np.cos(np.pi * t))) / lr0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    mask_ratio = float(cfg.get("mask_ratio", 0.5)); clip = cfg.get("grad_clip", 1.0)

    out_dir = os.path.join(args.out, cfg["name"]); os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt"); start = 1
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        core.load_state_dict(ck["full_model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = int(ck["step"]) + 1; print("resumed from step %d" % ck["step"], flush=True)

    iters = [iter(l) for _, l, _, _ in corpora]; running = []; t0 = time.time(); model.train()
    log_every = cfg.get("log_every_steps", 100); save_every = cfg.get("save_every_steps", 2000)
    for step in range(start, steps + 1):
        ci = int(np.random.choice(len(corpora), p=weights)); name, loader, _, _ = corpora[ci]
        try:
            x, _ = next(iters[ci])
        except StopIteration:
            iters[ci] = iter(loader); x, _ = next(iters[ci])
        x = x.to(device, non_blocking=True).float()
        B, C, T = x.shape
        if T % PATCH:
            x = x[..., : (T // PATCH) * PATCH]
        x = x.reshape(B, C, -1, PATCH)
        mask = generate_mask(B, C, x.shape[2], mask_ratio, device)
        opt.zero_grad(set_to_none=True)
        y = model(x, mask)
        sel = mask == 1
        loss = torch.nn.functional.mse_loss(y[sel], x[sel])
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); sched.step(); running.append(loss.item())
        if step % log_every == 0:
            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print("step %6d/%d  %.3fs/step  %4.1fGiB  lr=%.2e  loss=%.5f" % (
                step, steps, el / log_every, peak, sched.get_last_lr()[0], float(np.mean(running))), flush=True)
            running = []; t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        if step % save_every == 0 or step == steps:
            torch.save({"model": core.export_state_dict(), "full_model": core.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "cfg": cfg},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print("  -> saved %s (step %d)" % (ckpt_path, step), flush=True)
    print("pretraining done -> %s" % ckpt_path, flush=True)


if __name__ == "__main__":
    main()
