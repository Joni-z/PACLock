"""End-to-end check that the path refactor did not break loading.

The eight queued V2 jobs read code and configs only when they start, so they
will pick up the refactor. This builds a real loader and a real model from a
real config, which is the only thing that proves the import chain and the
$PACLOCK_PROC expansion survive contact with the training entry point.
"""

from __future__ import annotations

import sys

import yaml

from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model, count_params
from paclock_bench.paths import DATA, PROC_ROOT, REPO, expand, vendored

print("REPO      ", REPO)
print("PACLOCK_DATA", DATA)
print("PACLOCK_PROC", PROC_ROOT)
print("vendor/biot ", vendored("biot"))
print()

fail = 0
for cfg_path in ["configs/deliverable/tuev_paclock_v2.yaml",
                 "configs/deliverable/isruc_paclock_v2.yaml",
                 "configs/_cand/tuev_base.yaml"]:
    cfg = yaml.safe_load(open(cfg_path))
    try:
        print("%-42s data_root=%s" % (cfg_path, cfg["data_root"]))
        print("%-42s expands to %s" % ("", expand(cfg["data_root"])))
        tr, va, te, info = build_dataloaders(cfg)
        m = build_model(cfg, info["input_shape"])
        X, y = next(iter(tr))
        out = m(X[:2])
        print("%-42s loaders OK  input %s  out %s  %.3fM params"
              % ("", tuple(info["input_shape"]), tuple(out.shape), count_params(m)))
    except Exception as e:                                    # noqa: BLE001
        fail += 1
        print("%-42s FAILED  %s: %s" % ("", type(e).__name__, e))
    print()

print("%d of 3 configs failed" % fail)
sys.exit(1 if fail else 0)
