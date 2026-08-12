"""Completeness and health audit over runs/.

    python -m scripts.audit_runs

Complements collect_results (which reports metrics) by answering the questions
that decide whether a cell may enter the matrix at all:

* is every (dataset, model) cell at the 3 seeds hard rule 4 requires?
* did any seed trip hard rule 3 (val peak at epoch 0, or a flat val curve)?
* which cells have a seed spread large enough that the mean is not meaningful?

The last one is not in the xlsx rules but matters for the same reason they do:
ISRUC ST-Transformer came out at kappa 0.549 / 0.322 / 0.308 across seeds while
every other model on that dataset sat in 0.72-0.76. A mean over that is not a
description of anything.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

GROUP_A = ["sparcnet", "contrawr", "cnn_transformer", "ffcl", "st_transformer"]
DATASETS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
            "physionet_mi", "bci_iv_2a", "faced"]

# A cell whose seed spread exceeds this fraction of its mean is flagged: the
# mean stops being a useful summary well before it becomes formally invalid.
SPREAD_WARN = 0.15


def main() -> None:
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join("runs", "*", "seed*", "result.json"))):
        with open(f) as fh:
            r = json.load(fh)
        runs[(r["dataset"], r["model"])].append(r)

    print("=" * 92)
    print("A 组完成度与健康检查")
    print("=" * 92)
    print("%-14s %-17s %-6s %-22s %s" % ("dataset", "model", "seeds", "primary (mean±std)", "flags"))
    print("-" * 92)

    missing, misconf, unstable = [], [], []
    for ds in DATASETS:
        present = any((ds, m) in runs for m in GROUP_A)
        if not present:
            print("%-14s %-17s %-6s %-22s %s" % (ds, "-", "0/15", "-", "未运行"))
            missing.append((ds, "*"))
            continue
        for m in GROUP_A:
            rs = runs.get((ds, m), [])
            if not rs:
                print("%-14s %-17s %-6s %-22s %s" % (ds, m, "0/3", "-", "缺失"))
                missing.append((ds, m))
                continue
            key = rs[0]["primary_metric"]
            vals = [r["test"][key] for r in rs]
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=0))
            flags = []
            if len(rs) < 3:
                flags.append("n<3 (规则4)")
                missing.append((ds, m))
            bad = [r["seed"] for r in rs if not r["verdict"]["ok"]]
            if bad:
                flags.append(f"seed{bad} mis-configured (规则3)")
                misconf.append((ds, m, bad))
            if mean > 0 and std / abs(mean) > SPREAD_WARN:
                flags.append(f"seed 间离散 {std/abs(mean):.0%}")
                unstable.append((ds, m, mean, std, sorted(vals)))
            print("%-14s %-17s %-6s %-22s %s" % (
                ds, m, f"{len(rs)}/3", f"{mean:.4f}±{std:.4f}",
                "; ".join(flags) if flags else "ok"))

    print("=" * 92)
    total = sum(len(v) for v in runs.values())
    print(f"总计 {total} 个 run,{len(runs)} 个 (dataset, model) 单元格")
    print(f"  未达 3 seeds : {len(missing)}")
    print(f"  mis-configured: {len(misconf)}")
    print(f"  seed 不稳定  : {len(unstable)}")

    if unstable:
        print("\n" + "=" * 92)
        print("seed 间离散过大的单元格 —— 均值不足以描述,附各 seed 值与 val 曲线峰值位置")
        print("=" * 92)
        for ds, m, mean, std, vals in unstable:
            rs = runs[(ds, m)]
            key = rs[0]["primary_metric"]
            print(f"\n{ds} / {m}   {key} = {mean:.4f} ± {std:.4f}")
            for r in sorted(rs, key=lambda x: x["seed"]):
                curve = r["val_curve"]
                peak = int(np.argmax(curve)) if curve else -1
                print("  seed%-2d test=%.4f  best_val=%.4f  epochs=%-3d "
                      "val峰值在第%d/%d次评估  %s" % (
                          r["seed"], r["test"][key], r["best_val"],
                          r["epochs_run"], peak, len(curve),
                          r["verdict"]["status"]))


if __name__ == "__main__":
    main()
