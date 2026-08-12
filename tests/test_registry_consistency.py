"""Every dataset the configs mention must be wired into every lookup table.

    python -m tests.test_registry_consistency

Adding FACED took a config file, a preprocessing script and an entry in
gen_configs -- but not an entry in PRIMARY_METRIC, and that omission only
surfaced when 15 GPU jobs had already started and died on

    KeyError: unknown dataset 'faced'

Costing a node allocation to learn that a dict has one fewer key than another is
a bad trade. This test cross-checks the tables against each other so the failure
happens in a second, on any machine, before anything is submitted.
"""

from __future__ import annotations

import glob
import os
import sys

import yaml

from paclock_bench.training.metrics import PRIMARY_METRIC
from scripts.collect_results import DATASET_TO_SHEET

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# Datasets that have a frozen protocol config, minus anything explicitly retired.
proto_files = sorted(glob.glob("configs/datasets/*.yaml"))
datasets = {}
for p in proto_files:
    name = os.path.basename(p)[:-5]
    if name.startswith("_"):
        continue
    head = open(p).readline()
    if head.startswith("# DEPRECATED"):
        continue
    datasets[name] = yaml.safe_load(open(p))

check("found dataset protocols", len(datasets) >= 8, f"{sorted(datasets)}")

# 1. every protocol has a primary metric, and it agrees with the protocol file
for ds, cfg in datasets.items():
    in_table = ds in PRIMARY_METRIC
    check(f"{ds}: in PRIMARY_METRIC", in_table,
          "" if in_table else "training will KeyError at startup")
    if in_table and "primary_metric" in cfg:
        agree = PRIMARY_METRIC[ds] == cfg["primary_metric"]
        check(f"{ds}: primary metric agrees with protocol", agree,
              f"metrics.py={PRIMARY_METRIC[ds]} config={cfg['primary_metric']}")

# 2. no stale entries pointing at datasets that no longer exist
for ds in PRIMARY_METRIC:
    check(f"PRIMARY_METRIC['{ds}'] has a protocol", ds in datasets,
          "" if ds in datasets else "stale entry (dataset retired or renamed)")

# 3. results collection can name every dataset
for ds in datasets:
    check(f"{ds}: in DATASET_TO_SHEET", ds in DATASET_TO_SHEET,
          "" if ds in DATASET_TO_SHEET else "results would be filed under a raw key")

# 4. the metric a protocol names must be one compute_metrics actually returns
BINARY = {"auroc", "pr_auc", "balanced_acc"}
MULTI = {"balanced_acc", "cohen_kappa", "weighted_f1"}
for ds, cfg in datasets.items():
    if ds not in PRIMARY_METRIC:
        continue
    allowed = BINARY if cfg.get("num_classes") == 2 else MULTI
    m = PRIMARY_METRIC[ds]
    check(f"{ds}: '{m}' is computed for {cfg.get('num_classes')} classes",
          m in allowed, "" if m in allowed else f"expected one of {sorted(allowed)}")

# 5. gen_configs must know about the same set
gen = open("scripts/gen_configs.py").read()
for ds in datasets:
    check(f"{ds}: in gen_configs DATASETS", f'"{ds}"' in gen,
          "" if f'"{ds}"' in gen else "no experiment configs would be generated")

print("=" * 84)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
print("=" * 84)
nfail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - nfail}/{len(results)} checks passed")
sys.exit(1 if nfail else 0)
