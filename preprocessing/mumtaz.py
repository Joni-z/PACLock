"""Mumtaz2016 (MDD vs healthy resting EEG) -> windowed npy + manifest.

    sbatch slurm/preprocess_new2.slurm  (or)
    python -m preprocessing.mumtaz --config configs/datasets/mumtaz.yaml

Why this corpus joins the suite: clinical recording-level binary -- the task
shape this architecture is measured to be strong in -- and it is on CBraMod's
downstream list, so a published FM reference number exists.

Source (figshare 4244171): one flat directory of EDFs named
``{H|MDD} S<n> {EC|EO|TASK}.edf``. Following CBraMod's protocol the TASK
condition is dropped and only resting EC/EO recordings are used; unlike
CBraMod, whose published split sorts *files* (one subject's EC and EO can land
in different splits), the split here is subject-disjoint sorted and stratified
by diagnosis -- the label is a property of the subject (docs/PROTOCOLS.md).

19 referential 10-20 channels (``EEG Xx-LE``), matched case-insensitively and
reordered to the canonical order in the config. 256 Hz -> 200 Hz, 0.3-75 Hz
band, 50 Hz notch (Malaysia), 5 s windows (CBraMod's window), div100.
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

FNAME_RE = re.compile(r"^(H|MDD)\s?S(\d+)\s?(EC|EO|TASK)\.edf$", re.IGNORECASE)


def collect(root: str, conditions: list[str]) -> list[tuple[str, str, int]]:
    """(edf_path, subject_key, label) with label MDD=1 / H=0."""
    out = []
    for f in sorted(os.listdir(root)):
        m = FNAME_RE.match(f)
        if not m:
            continue
        group, num, cond = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        if cond not in conditions:
            continue
        out.append((os.path.join(root, f), f"{group}-S{num:02d}",
                    1 if group == "MDD" else 0))
    return out


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
    ap.add_argument("--config", default="configs/datasets/mumtaz.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    items = collect(root, cfg["conditions"])
    if not items:
        raise SystemExit(f"no EC/EO EDFs under {root}")

    # subject-disjoint sorted split, stratified by diagnosis (tuep.py rule)
    by_label = {0: set(), 1: set()}
    for _, sub, lab in items:
        by_label[lab].add(sub)
    groups = {"train": set(), "val": set(), "test": set()}
    for lab, subs in by_label.items():
        tr, rest = sort_subject_split(sorted(subs), cfg["split"]["train_ratio"])
        va, te = sort_subject_split(rest, cfg["split"]["val_ratio_of_rest"])
        groups["train"] |= set(tr)
        groups["val"] |= set(va)
        groups["test"] |= set(te)
    print("subjects: %d train / %d val / %d test"
          % (len(groups["train"]), len(groups["val"]), len(groups["test"])),
          flush=True)

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
              "note": "EC+EO resting only (TASK dropped, CBraMod precedent); "
                      "subject-disjoint sorted split stratified by diagnosis"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
