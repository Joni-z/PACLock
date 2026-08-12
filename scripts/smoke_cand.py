"""Build and run one forward pass of every candidate config.

Exists because a candidate sweep was submitted with a head.py that used PEP 604
`int | None` without `from __future__ import annotations`. The cluster runs
Python 3.9, where that annotation is evaluated at class-definition time, so every
PACLock job died on import after 13 seconds -- eight jobs, all preventable by one
import. ast.parse does not catch it; only an actual import does.
"""
import glob
import sys

import torch
import yaml

sys.path.insert(0, ".")
from paclock_bench.models.build import build_model, count_params  # noqa: E402

SHAPE = {"tuev": (16, 1000), "isruc": (6, 6000)}

fail = 0
for path in sorted(glob.glob("configs/_cand/*.yaml")):
    cfg = yaml.safe_load(open(path))
    shape = SHAPE[cfg["dataset"]]
    try:
        model = build_model(cfg, shape)
        with torch.no_grad():
            out = model(torch.randn(2, *shape))
        assert out.shape[0] == 2, out.shape
        print(f"  ok   {path.split('/')[-1]:26s} out {tuple(out.shape)}  "
              f"{count_params(model):.3f}M")
    except Exception as e:                                  # noqa: BLE001
        fail += 1
        print(f"  FAIL {path.split('/')[-1]:26s} {type(e).__name__}: {e}")

print(f"\n{fail} candidate configs failed")
sys.exit(1 if fail else 0)
