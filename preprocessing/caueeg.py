"""CAUEEG (dementia benchmark) -> windowed npy + manifest.

    python -m preprocessing.caueeg --config configs/datasets/caueeg.yaml

External Korean clinical cohort (Kim, Youn & Paik, NeuroImage 2023),
committee-approved 2026-08-24. The 3-class dementia benchmark
(Normal / MCI / Dementia) ships with the AUTHORS' OWN splits; this loader
uses their ``dementia-no-overlap.json`` variant -- official and
subject-disjoint at once, so the external-cohort claim carries no leakage
caveat. CEEDNet's published numbers use the (older) overlapping
``dementia.json`` split; the workbook cites them as reference with that
caveat rather than re-running them.

Signals: 19-channel average-reference 10-20 EEG plus EKG/Photic (dropped).
Channels are reordered to this benchmark's canonical 19-electrode order
(montage.py _MONO_19). 200 Hz native (asserted per file), 0.3-75 Hz band,
60 Hz notch (Korean mains), 10 s windows, div100.
"""

from __future__ import annotations

import argparse
import json
import os
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
    window_signal,
)
from paclock_bench.paths import expand


def load_recording(path: str, cfg: dict) -> np.ndarray:
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
    ap.add_argument("--config", default="configs/datasets/caueeg.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    task = json.load(open(os.path.join(root, cfg["task_json"])))
    print("task:", task["task_name"], "| classes:", task["class_label_to_name"],
          flush=True)
    splits = {"train": task["train_split"], "val": task["validation_split"],
              "test": task["test_split"]}

    win = int(cfg["window_sec"] * cfg["sample_rate"])
    for split, entries in splits.items():
        X_all, y_all, used = [], [], []
        for e in entries:
            serial, label = e["serial"], int(e["class_label"])
            path = os.path.join(root, "signal", "edf", f"{serial}.edf")
            if not os.path.exists(path):
                man.exclude(serial, "no EDF in release", split=split)
                continue
            try:
                sig = load_recording(path, cfg)
                X, _ = window_signal(sig, win, win)
                if len(X) == 0:
                    man.exclude(serial, "shorter than one window", split=split)
                    continue
                X = norm_div100(X).astype(np.float32)
                assert_finite(X, serial)
                X_all.append(X)
                y_all.append(np.full(len(X), label, dtype=np.int64))
                used.append(serial)
            except Exception as ex:                          # noqa: BLE001
                man.exclude(serial, f"{type(ex).__name__}: {ex}", split=split)
        X = np.concatenate(X_all).astype(np.float32)
        y = np.concatenate(y_all)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=used, n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(used), shape=list(X.shape[1:]))
        print(f"[{split}] {len(used)}/{len(entries)} recordings -> "
              f"{len(X)} windows, classes "
              f"{sorted(Counter(y.tolist()).items())}", flush=True)

    man.qc = {"n_excluded": len(man.excluded),
              "note": "authors' dementia-no-overlap splits (official AND "
                      "subject-disjoint); serials without an EDF in the "
                      "release are excluded per split above"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
