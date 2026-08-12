"""Verify a processed dataset against its own manifest before training on it.

    python -m scripts.verify_processed <dataset>

Exits non-zero on any mismatch so a Slurm chain can gate training with
--dependency=afterok. Added after a transfer-in-progress directory looked
"ready" because manifest.json (alphabetically early) had landed while
train_signals.npy was still 11% uploaded.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ds = sys.argv[1]
root = f"/work1/chenyuyou/yifanwang/Zhizhe/processed/{ds}"
man = json.load(open(os.path.join(root, "manifest.json")))

problems = []
print(f"verifying {root}")
for split, v in man["splits"].items():
    sig_p = os.path.join(root, f"{split}_signals.npy")
    lab_p = os.path.join(root, f"{split}_labels.npy")
    for p in (sig_p, lab_p):
        if not os.path.exists(p):
            problems.append(f"{split}: missing {os.path.basename(p)}")
    if problems:
        continue
    sig = np.load(sig_p, mmap_mode="r")
    lab = np.load(lab_p)
    n = v["n_windows"]
    shape = tuple(v["shape"])
    if len(sig) != n:
        problems.append(f"{split}: signals has {len(sig)} rows, manifest says {n}")
    if len(lab) != n:
        problems.append(f"{split}: labels has {len(lab)} rows, manifest says {n}")
    if tuple(sig.shape[1:]) != shape:
        problems.append(f"{split}: shape {sig.shape[1:]} != manifest {shape}")
    # touch both ends: a truncated upload reads as zeros or raises here
    _ = np.asarray(sig[0]), np.asarray(sig[-1])
    counts = {str(k): int(c) for k, c in
              zip(*np.unique(np.asarray(lab).ravel(), return_counts=True))}
    if counts != v["class_counts"]:
        problems.append(f"{split}: class counts {counts} != manifest {v['class_counts']}")
    print(f"  {split:5s} {sig.shape} labels={lab.shape} classes={len(counts)}  OK")

qc = man.get("qc", {})
print("qc:", {k: qc.get(k) for k in
               ("n_subjects_found", "n_subjects_expected", "n_excluded",
                "split_is_subject_disjoint", "filtering") if k in qc})

if problems:
    print("\nFAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nVERIFIED: manifest and npy agree")
