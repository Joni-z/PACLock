"""PhysioNet EEG Motor Movement/Imagery v1.0.0 -> trial npy + manifest.
Protocol: docs/PROTOCOLS.md sec.7.

    python -m preprocessing.physionet_mi --config configs/datasets/physionet_mi.yaml

Motor-imagery runs only (04, 06, 08, 10, 12, 14). The T1/T2 event codes mean
different things depending on the run, so the four-class mapping is per-run:
runs 4/8/12 give left/right fist, runs 6/10/14 give both fists / both feet.
T0 (rest) is dropped.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import mne
import numpy as np

from paclock_bench.paths import expand
import yaml

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    preprocess_signal,
    save_split,
    sha256_file,
)


def process_run(edf_path: str, run: int, cfg: dict):
    """One run EDF -> (n_trials, 64, 800) and labels."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if cfg.get("reference") == "average":
        raw.set_eeg_reference("average", verbose="ERROR")
    sig = raw.get_data() * 1e6                                  # volts -> uV
    ann = raw.annotations
    del raw

    if sig.shape[0] != cfg["n_channels"]:
        raise ValueError(f"expected {cfg['n_channels']} channels, got {sig.shape[0]}")

    fs_out = cfg["sample_rate"]
    sig = preprocess_signal(sig, fs, fs_out=fs_out, band=None,
                            hp=cfg["hp"], notch_freq=cfg["notch"])

    run_map = cfg["label_map"][str(run)]
    drop = set(cfg["drop_labels"])
    n = int(cfg["trial_sec"] * fs_out)
    off = int(cfg["trial_offset_sec"] * fs_out)
    T = sig.shape[1]

    X, y = [], []
    for onset, desc in zip(ann.onset, ann.description):
        code = desc.strip()
        if code in drop or code not in run_map:
            continue
        a = int(round(onset * fs_out)) + off                    # no baseline correction
        b = a + n
        if a < 0 or b > T:
            continue
        X.append(sig[:, a:b])
        y.append(run_map[code])
    if not X:
        raise ValueError("no usable trials")
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # Accepted for interface parity: slurm/preprocess.slurm passes --jobs to
    # every dataset. This script is single-process (one pass over the files),
    # so the value is recorded in the manifest but not used to fan out.
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    sp = cfg["split"]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    def split_of(n: int) -> str | None:
        for name in ("train", "val", "test"):
            lo, hi = sp[name]
            if lo <= n <= hi:
                return name
        return None

    buckets = {k: {"X": [], "y": [], "subs": []} for k in ("train", "val", "test")}
    subjects = sorted(d for d in os.listdir(root) if d.startswith("S") and d[1:].isdigit())

    for s in subjects:
        n = int(s[1:])
        split = split_of(n)
        if split is None:
            man.exclude(s, f"subject {n} outside all split ranges")
            continue
        got = False
        for run in cfg["runs"]:
            f = os.path.join(root, s, f"{s}R{run:02d}.edf")
            if not os.path.exists(f):
                man.exclude(f"{s}R{run:02d}", "file not found", split=split)
                continue
            try:
                X, y = process_run(f, run, cfg)
            except Exception as e:                              # noqa: BLE001
                man.exclude(f"{s}R{run:02d}", f"{type(e).__name__}: {e}", split=split)
                continue
            buckets[split]["X"].append(X)
            buckets[split]["y"].append(y)
            man.raw_sha256[f"{s}R{run:02d}.edf"] = sha256_file(f)
            got = True
        if got:
            buckets[split]["subs"].append(s)
            print(f"  {s} -> {split}", flush=True)

    for split, b in buckets.items():
        if not b["X"]:
            raise RuntimeError(f"{split}: no trials")
        X = norm_div100(np.concatenate(b["X"])).astype(np.float32)
        y = np.concatenate(b["y"]).astype(np.int64)
        assert_finite(X, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=b["subs"], n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    man.check_disjoint()
    man.qc = {"n_excluded": len(man.excluded), "n_subjects": len(subjects),
              "runs_used": cfg["runs"]}
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
