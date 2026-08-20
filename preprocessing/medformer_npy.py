"""ADFD / APAVA (Medformer .npy release) -> windowed npy + manifest.

    python -m preprocessing.medformer_npy --config configs/datasets/adfd.yaml
    python -m preprocessing.medformer_npy --config configs/datasets/apava.yaml

Both corpora ship in Medformer's preprocessed release: one
``Feature/feature_XX.npy`` per subject shaped (n_segments, 256, n_channels) --
consecutive ONE-SECOND segments at 256 Hz -- plus ``Label/label.npy`` shaped
(n_subjects, 2) = [class, subject_id]. Using the published release rather than
re-deriving from raw follows the FACED precedent (docs/PROTOCOLS.md sec.8:
when an official preprocessed release exists, we take it, because bit-exact
reproduction of upstream cleaning is impossible and any divergence would be
silent).

Per subject: the 1 s segments are concatenated back into a continuous stream
(they were cut consecutively, so this restores continuity up to the original
segment boundaries), resampled 256 -> 200 Hz, band-passed and notch-filtered
(both corpora are European: 50 Hz mains), cut into windows, div100-normalised.
ADFD's arrays are in VOLTS (~1e-4 range) and are scaled to microvolts first;
APAVA's are already in microvolts -- the config's ``unit_scale`` records the
factor rather than leaving it implicit in code.

Split: subject-disjoint sorted, stratified by class -- the tuep.py rule; the
label is a property of the subject, so no recording can straddle splits.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import yaml

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    preprocess_signal,
    save_split,
    window_signal,
)
from .tuh_common import sort_subject_split
from paclock_bench.paths import expand


def load_subject(root: str, sid: int, unit_scale: float) -> np.ndarray:
    a = np.load(os.path.join(root, "Feature", f"feature_{sid:02d}.npy"))
    # (n_seg, 256, C) consecutive 1 s segments -> continuous (C, T)
    sig = a.transpose(2, 0, 1).reshape(a.shape[2], -1).astype(np.float64)
    return sig * unit_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    lab = np.load(os.path.join(root, "Label", "label.npy")).astype(int)
    subjects = {int(sid): int(y) for y, sid in lab}

    # subject-disjoint sorted split, stratified by class (tuep.py rule)
    groups = {"train": set(), "val": set(), "test": set()}
    for klass in sorted(set(subjects.values())):
        subs = sorted(s for s, y in subjects.items() if y == klass)
        tr, rest = sort_subject_split([str(s) for s in subs],
                                      cfg["split"]["train_ratio"])
        va, te = sort_subject_split(rest, cfg["split"]["val_ratio_of_rest"])
        groups["train"] |= {int(s) for s in tr}
        groups["val"] |= {int(s) for s in va}
        groups["test"] |= {int(s) for s in te}
    print("subjects: %d train / %d val / %d test"
          % (len(groups["train"]), len(groups["val"]), len(groups["test"])),
          flush=True)

    win = int(cfg["window_sec"] * cfg["sample_rate"])
    for split, keep in groups.items():
        X_all, y_all, used = [], [], []
        for sid in sorted(keep):
            try:
                sig = load_subject(root, sid, float(cfg["unit_scale"]))
                sig = preprocess_signal(
                    sig, cfg["source_rate"], fs_out=cfg["sample_rate"],
                    band=tuple(cfg["band"]), notch_freq=cfg.get("notch"),
                )
                X, _ = window_signal(sig, win, win)
                if len(X) == 0:
                    man.exclude(f"subject_{sid:02d}", "shorter than one window",
                                split=split)
                    continue
                X = norm_div100(X).astype(np.float32)
                assert_finite(X, f"subject_{sid:02d}")
                X_all.append(X)
                y_all.append(np.full(len(X), subjects[sid], dtype=np.int64))
                used.append(str(sid))
            except Exception as e:                            # noqa: BLE001
                man.exclude(f"subject_{sid:02d}",
                            f"{type(e).__name__}: {e}", split=split)
        X = np.concatenate(X_all).astype(np.float32)
        y = np.concatenate(y_all)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=used, n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(used), shape=list(X.shape[1:]))
        print(f"[{split}] {len(used)} subjects -> {len(X)} windows", flush=True)

    man.qc = {"n_excluded": len(man.excluded),
              "note": "Medformer preprocessed release; 1 s segments "
                      "concatenated per subject, resampled 256->200 Hz; "
                      "subject-disjoint sorted split stratified by class"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
