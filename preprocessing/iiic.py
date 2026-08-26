"""IIIC (BDSP SPaRCNet release) -> windowed npy + manifest.

    python -m preprocessing.iiic --config configs/datasets/iiic.yaml

Uses the official prepared arrays (``sparcnet_data/all_train_*.npy``) rather
than re-deriving from raw EDF -- the FACED/Medformer precedent
(docs/PROTOCOLS.md sec.8): when the authors publish a preprocessed release,
we take it, because silent divergence from their cleaning is worse than
inheriting it. Facts verified against the authors' own runSPaRCNet.py:
the 16 rows are the bipolar double banana in LL/RL/LP/RP order -- exactly
this benchmark's TUH montage order (montage.py _BIPOLAR_16) -- at 200 Hz,
10 s windows, 60 Hz notch + 0.5-40 Hz bandpass already applied, microvolts.

What this script adds on top of the release:
  * dedupe by sample key -- 134,450 rows carry only 111,095 unique keys, and
    a duplicated key crossing a split boundary would be leakage;
  * label = majority expert vote (argmax of the per-class vote counts);
    tied rows are excluded and counted in the manifest;
  * patient-disjoint 70/15/15 split with the class mix BALANCED across
    splits. A plain sorted-by-ID cut -- this benchmark's default rule --
    is wrong here: patient IDs correlate with pattern type, so the first
    version of this loader produced train 44% class-1 / 0.9% class-2
    against test 8% / 18%, and every model scored 0.15-0.23 kappa against
    ~0.45 published. BIOT (NeurIPS 2023) splits IIIC "by patient groups
    60:20:20" at random, whose expected class mix is identical across
    splits; the greedy assignment below reaches the same property without
    an RNG, keeping the no-random-seed rule the rest of the pipeline uses.
    The release's own 10_test split has no public labels, so it cannot
    serve as eval.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import yaml

from .common import Manifest, assert_finite, norm_div100, save_split
from paclock_bench.paths import expand


def balanced_patient_split(patients, labels, train_ratio, val_ratio_of_rest,
                           n_classes):
    """Patient-disjoint split whose class mix matches across the three splits.

    Deterministic by construction: patients are visited largest-first (ties
    by ID) and each is placed in whichever split is currently furthest below
    its target on that patient's own dominant class, subject to the split's
    window quota. Largest-first matters -- placing the big patients while
    every split is still empty is what lets the small ones correct the mix.
    """
    order = sorted(set(patients.tolist()),
                   key=lambda p: (-(patients == p).sum(), p))
    targets = {"train": train_ratio,
               "val": (1 - train_ratio) * val_ratio_of_rest,
               "test": (1 - train_ratio) * (1 - val_ratio_of_rest)}
    total = len(patients)
    counts = {k: np.zeros(n_classes, dtype=np.int64) for k in targets}
    sizes = {k: 0 for k in targets}
    groups = {k: set() for k in targets}
    for p in order:
        sel = patients == p
        hist = np.bincount(labels[sel], minlength=n_classes)
        dom = int(hist.argmax())
        best, best_key = None, None
        for k, ratio in targets.items():
            room = ratio * total - sizes[k]
            if room <= 0:                       # quota full: heavy penalty
                room = -abs(room) - total
            # want the split that is furthest behind on this patient's
            # dominant class, tie-broken by remaining room
            deficit = ratio * total * (
                np.bincount(labels, minlength=n_classes)[dom] / total) \
                - counts[k][dom]
            score = deficit + 1e-6 * room
            if room < -total / 2:
                score -= total
            if best is None or score > best:
                best, best_key = score, k
        groups[best_key].add(p)
        counts[best_key] += hist
        sizes[best_key] += int(sel.sum())
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/iiic.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    keys = np.load(os.path.join(root, "all_train_key.npy"))
    votes = np.load(os.path.join(root, "all_train_Y.npy"))

    # first occurrence of each key wins; later duplicates are dropped
    _, first_idx = np.unique(keys, return_index=True)
    first_idx = np.sort(first_idx)
    n_dupes = len(keys) - len(first_idx)

    # majority vote; ties excluded
    v = votes[first_idx]
    top = v.max(axis=1, keepdims=True)
    tied = (v == top).sum(axis=1) > 1
    keep_idx = first_idx[~tied]
    labels = v[~tied].argmax(axis=1).astype(np.int64)
    print(f"{len(keys)} rows -> {len(first_idx)} unique keys "
          f"({n_dupes} duplicates dropped) -> {len(keep_idx)} after "
          f"excluding {int(tied.sum())} vote ties", flush=True)
    man.exclude("(vote ties)", f"{int(tied.sum())} samples with tied majority vote")
    man.exclude("(duplicate keys)", f"{n_dupes} rows repeating an already-seen key")

    patients = np.array([k.split("_")[0] for k in keys[keep_idx]])
    groups = balanced_patient_split(
        patients, labels, cfg["split"]["train_ratio"],
        cfg["split"]["val_ratio_of_rest"], n_classes=votes.shape[1])
    for name, g in groups.items():
        sel = np.isin(patients, sorted(g))
        frac = np.bincount(labels[sel], minlength=votes.shape[1]) / max(sel.sum(), 1)
        print("patients[%s]=%d windows=%d class mix %s"
              % (name, len(g), int(sel.sum()),
                 " ".join("%.3f" % f for f in frac)), flush=True)

    X_all = np.load(os.path.join(root, "all_train_X.npy"), mmap_mode="r")
    for split, keep in groups.items():
        sel = np.isin(patients, sorted(keep))
        idx = keep_idx[sel]
        X = np.asarray(X_all[idx], dtype=np.float32)
        X = norm_div100(X).astype(np.float32)
        assert_finite(X, split)
        y = labels[sel]
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(patients[sel].tolist())),
                      n_windows=len(X), class_counts=Counter(y.tolist()),
                      n_recordings=len(set(patients[sel].tolist())),
                      shape=list(X.shape[1:]))
        print(f"[{split}] {sel.sum()} windows, {len(set(patients[sel].tolist()))} "
              f"patients, classes {sorted(Counter(y.tolist()).items())}", flush=True)

    man.qc = {"n_excluded": len(man.excluded),
              "note": "official SPaRCNet prepared arrays; dedupe by key; "
                      "majority-vote labels, ties excluded; patient-disjoint "
                      "CLASS-BALANCED split (a sorted cut put 44% of class 1 "
                      "in train against 8% in test); release test labels are "
                      "not public"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
