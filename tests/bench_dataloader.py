"""Measure whether the dataloader, not the GPU, is the bottleneck.

    python -m tests.bench_dataloader [dataset ...]

The reference repo hit a ~15 it/s ceiling on TUAB because BIOT-style
preprocessing left one pickle per window and every __getitem__ was a separate
open(). This repo writes one consolidated npy per split instead, so that
specific problem is gone by construction -- but "consolidated" does not
automatically mean "fast": these files live on WekaFS, TUAB's train split is a
single 38 GB file, and shuffled training reads it in random order.

So measure rather than assume. Reports samples/s for shuffled and sequential
reads and compares against the rate a GPU step actually needs.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from paclock_bench.data.datasets import WindowDataset

PROC = "/work1/chenyuyou/yifanwang/Zhizhe/processed"


def bench(ds, batch_size: int, workers: int, shuffle: bool, n_batches: int):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    it = iter(loader)
    next(it)                                     # warm up workers
    t0 = time.time()
    n = 0
    for _ in range(n_batches):
        try:
            X, _y = next(it)
        except StopIteration:
            break
        n += len(X)
    dt = time.time() - t0
    del it, loader
    return n / dt if dt > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["tuev", "tuab", "chbmit"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batches", type=int, default=40)
    args = ap.parse_args()

    print(f"batch_size={args.batch_size}  batches={args.batches}\n")
    print("%-10s %-22s %10s %10s %10s" % (
        "dataset", "train shape", "seq/s", "shuf/s", "shuf w=8/s"))
    print("-" * 68)

    for name in args.datasets:
        root = os.path.join(PROC, name)
        if not os.path.exists(os.path.join(root, "train_signals.npy")):
            print("%-10s %-22s   (not preprocessed)" % (name, "-"))
            continue
        ds = WindowDataset(root, "train", flatten_sequences=True)
        shape = "x".join(str(s) for s in ds.shape)
        seq = bench(ds, args.batch_size, 0, False, args.batches)
        shuf = bench(ds, args.batch_size, 0, True, args.batches)
        shuf8 = bench(ds, args.batch_size, 8, True, args.batches)
        print("%-10s %-22s %10.0f %10.0f %10.0f" % (
            name, f"{len(ds)}x{shape}", seq, shuf, shuf8))

    # What rate does the GPU actually consume? Compare against the slowest
    # loader above; if the loader is faster, IO is not the bottleneck.
    if torch.cuda.is_available():
        from paclock_bench.models.baselines.light_supervised import REGISTRY

        print("\nGPU step rate (samples/s), TUEV shape 16x1000:")
        x = torch.randn(64, 16, 1000, device="cuda")
        y = torch.randint(0, 6, (64,), device="cuda")
        lossf = torch.nn.CrossEntropyLoss()
        for mname in ("sparcnet", "ffcl", "st_transformer"):
            m = REGISTRY[mname](in_channels=16, seq_len=1000,
                                num_classes=6, sample_rate=200).cuda()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            for _ in range(3):                    # warm up kernels
                opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
            torch.cuda.synchronize()
            t0 = time.time()
            steps = 20
            for _ in range(steps):
                opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
            torch.cuda.synchronize()
            rate = steps * 64 / (time.time() - t0)
            print("  %-18s %8.0f samples/s" % (mname, rate))
            del m, opt


if __name__ == "__main__":
    main()
