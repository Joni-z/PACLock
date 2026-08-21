"""EEGMat (PhysioNet "EEG During Mental Arithmetic Tasks" v1.0.0) -> npy.

    sbatch slurm/preprocess_new2.slurm  (or)
    python -m preprocessing.eegmat --config configs/datasets/eegmat.yaml

Why this corpus joins the suite: recording-level cognitive-state binary
(background rest vs mental arithmetic), on CBraMod's downstream list (their
"stress"/MentalArithmetic set), so a published FM reference number exists.

Source: ``Subject<NN>_1.edf`` (background, label 0) and ``Subject<NN>_2.edf``
(arithmetic, label 1) for 36 subjects -- the label is the recording suffix.
Every subject holds both classes, so a subject-disjoint sorted split needs no
stratification. 19 referential 10-20 channels (``EEG Xx``; the A2-A1 reference
derivation and ECG are dropped -- SpatialPE wants electrodes with positions).
500 Hz -> 200 Hz; the provider distributes 0.5-45 Hz filtered signals, so the
band here is [0.3, 45] and no notch is applied (50 Hz mains sits above the
provider's low-pass). 5 s windows (CBraMod's window), div100.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import mne
import numpy as np
import yaml

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    preprocess_signal,
    save_split,
    sha256_file,
    window_signal,
)
from .tuh_common import sort_subject_split
from paclock_bench.paths import expand

FNAME_RE = re.compile(r"^(Subject\d+)_([12])\.edf$")


def collect(root: str) -> list[tuple[str, str, int]]:
    """(edf_path, subject, label) walking the extracted release."""
    out = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            m = FNAME_RE.match(f)
            if m:
                out.append((os.path.join(dirpath, f), m.group(1),
                            int(m.group(2)) - 1))
    return sorted(out)


def load_recording(path: str, cfg: dict) -> np.ndarray:
    """(C, T) microvolt array in the config's canonical channel order."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    lower = {c.lower(): c for c in raw.ch_names}
    picks = []
    for want in cfg["channels_edf"]:
        got = lower.get(want.lower())
        if got is None:
            raise KeyError(f"missing channel {want}")
        picks.append(got)
    sig = raw.get_data(picks=picks, units="uV")
    fs = float(raw.info["sfreq"])
    return preprocess_signal(sig, fs, fs_out=cfg["sample_rate"],
                             band=tuple(cfg["band"]),
                             notch_freq=cfg.get("notch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/eegmat.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    items = collect(root)
    if not items:
        raise SystemExit(f"no Subject*_[12].edf under {root}")

    subjects = sorted({s for _, s, _ in items})
    tr, rest = sort_subject_split(subjects, cfg["split"]["train_ratio"])
    va, te = sort_subject_split(rest, cfg["split"]["val_ratio_of_rest"])
    groups = {"train": set(tr), "val": set(va), "test": set(te)}
    print("subjects: %d train / %d val / %d test"
          % (len(tr), len(va), len(te)), flush=True)

    win = int(cfg["window_sec"] * cfg["sample_rate"])
    for split, keep in groups.items():
        X_all, y_all, used = [], [], set()
        for path, sub, lab in items:
            if sub not in keep:
                continue
            try:
                sig = load_recording(path, cfg)
                X, _ = window_signal(sig, win, win)
                if len(X) == 0:
                    man.exclude(os.path.basename(path),
                                "shorter than one window", split=split)
                    continue
                X = norm_div100(X).astype(np.float32)
                assert_finite(X, os.path.basename(path))
                X_all.append(X)
                y_all.append(np.full(len(X), lab, dtype=np.int64))
                used.add(sub)
                man.raw_sha256[os.path.basename(path)] = sha256_file(path)
            except Exception as e:                            # noqa: BLE001
                man.exclude(os.path.basename(path),
                            f"{type(e).__name__}: {e}", split=split)
        X = np.concatenate(X_all).astype(np.float32)
        y = np.concatenate(y_all)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(used), n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(used), shape=list(X.shape[1:]))
        print(f"[{split}] {len(used)} subjects -> {len(X)} windows", flush=True)

    man.qc = {"n_excluded": len(man.excluded),
              "note": "label = recording suffix (_1 rest / _2 arithmetic); "
                      "subject-disjoint sorted split, both classes per subject"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
