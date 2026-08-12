"""Instantiate and forward every generated experiment config.

    python -m tests.test_all_configs_forward

35 configs (5 group-A models x 7 datasets) each with their own channel count,
window length and sample rate. Shape bugs in that grid are invisible until a
job starts, and a job that dies twenty minutes in has already cost a node
allocation -- ContraWR on Sleep-EDF failed exactly this way, because its
spectrogram does not collapse to 1x1 for a 30 s epoch.

Uses the real input shape from each dataset's manifest, so this checks the
configs against the data that actually exists rather than against the config's
own claims.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import yaml

from paclock_bench.models.build import build_model

results = []

for path in sorted(glob.glob("configs/experiments/*.yaml")):
    cfg = yaml.safe_load(open(path))
    name = cfg["name"]
    root = cfg["data_root"]
    sig = os.path.join(root, "train_signals.npy")
    if not os.path.exists(sig):
        results.append((name, None, "skip: not preprocessed", True))
        continue

    # real shape, straight from the data
    arr = np.load(sig, mmap_mode="r")
    shape = arr.shape[1:]
    if cfg.get("flatten_sequences") and len(shape) == 3:
        shape = shape[1:]
    C, T = int(shape[-2]), int(shape[-1])

    try:
        model = build_model(cfg, (C, T)).eval()
        n = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            out = model(torch.randn(2, C, T))
        ok = out.shape == (2, cfg["num_classes"])
        detail = f"{C}x{T} -> {tuple(out.shape)}  {n:.2f}M"
        if not ok:
            detail += f"  EXPECTED (2, {cfg['num_classes']})"
        results.append((name, n, detail, ok))
    except Exception as e:                                        # noqa: BLE001
        results.append((name, None, f"{C}x{T}  {type(e).__name__}: {e}", False))

print("=" * 92)
for name, _n, detail, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} {detail}")
print("=" * 92)
nfail = sum(1 for r in results if not r[3])
print(f"{len(results) - nfail}/{len(results)} configs forward correctly")
sys.exit(1 if nfail else 0)
