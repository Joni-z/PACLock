"""BCI Competition IV-2a -> trial npy + manifest. Protocol: docs/PROTOCOLS.md sec.9.

    python -m preprocessing.bci_iv_2a --config configs/datasets/bci_iv_2a.yaml

Loaded through MOABB (BNCI2014_001) so the raw .mat parsing and event coding
come from a maintained reader rather than a hand-rolled one.

Split is by session and run: session T runs 1-5 are train and run 6 is val;
session E is test and is evaluated exactly once. The trial window is cue onset
+ 0..4 s with no pre-cue baseline.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import yaml

from .common import (
    highpass,
    Manifest,
    assert_finite,
    bandpass,
    norm_div100,
    notch,
    resample_to,
    save_split,
)


def epochs_from_run(raw, cfg: dict):
    """One MOABB run -> (n_trials, 22, 800) and labels, filtered and resampled."""
    import mne

    fs = float(raw.info["sfreq"])
    if cfg.get("drop_eog"):
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
        ch = [raw.ch_names[i] for i in picks]
    else:
        ch = raw.ch_names
    if cfg.get("strict_channels") and len(ch) != cfg["n_channels"]:
        raise ValueError(f"expected {cfg['n_channels']} EEG channels, got {len(ch)}")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    label_map = cfg["label_map"]
    # MOABB names the four cues directly; keep only those
    keep = {code: label_map[name] for name, code in event_id.items()
            if name in label_map}
    if not keep:
        raise ValueError(f"no MI events in {sorted(event_id)}")

    sig = raw.get_data(picks=ch) * 1e6                          # volts -> uV
    # PAC protocol drops the band-pass for a high-pass; see
    # scripts/make_pac_protocol.py. One implementation serves both so
    # only the filter settings can differ between the two protocols.
    _b = cfg.get("band")
    sig = (bandpass(sig.astype(np.float64), fs, _b[0], _b[1]) if _b
           else highpass(sig.astype(np.float64), fs, cfg["hp"]))
    if cfg.get("notch"):
        sig = notch(sig, fs, cfg["notch"])
    sig = resample_to(sig, fs, cfg["sample_rate"])

    fs_out = cfg["sample_rate"]
    scale = fs_out / fs
    n = int(cfg["trial_sec"] * fs_out)
    off = int(cfg["trial_offset_sec"] * fs_out)
    T = sig.shape[1]

    X, y = [], []
    for sample, _, code in events:
        if code not in keep:
            continue
        a = int(round(sample * scale)) + off                    # cue onset, no baseline
        b = a + n
        if a < 0 or b > T:
            continue
        X.append(sig[:, a:b])
        y.append(keep[code])
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
    out_dir = cfg["out_dir"]
    sp = cfg["split"]

    os.environ.setdefault("MNE_DATA", cfg["raw_root"])
    from moabb.datasets import BNCI2014_001

    ds = BNCI2014_001()
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)
    buckets = {k: {"X": [], "y": [], "subs": []} for k in ("train", "val", "test")}
    train_runs = {str(r) for r in sp["train_runs"]}
    val_runs = {str(r) for r in sp["val_runs"]}

    for subject in ds.subject_list:
        data = ds.get_data(subjects=[subject])[subject]
        for session, runs in data.items():
            for run_name, raw in runs.items():
                # MOABB run keys are like '0', '1', ... or 'run_0'; take the digits
                digits = "".join(c for c in run_name if c.isdigit())
                if session == sp["test_session"]:
                    split = "test"
                elif digits in train_runs:
                    split = "train"
                elif digits in val_runs:
                    split = "val"
                else:
                    man.exclude(f"S{subject}/{session}/{run_name}",
                                "run not in any split")
                    continue
                try:
                    X, y = epochs_from_run(raw, cfg)
                except Exception as e:                          # noqa: BLE001
                    man.exclude(f"S{subject}/{session}/{run_name}",
                                f"{type(e).__name__}: {e}", split=split)
                    continue
                buckets[split]["X"].append(X)
                buckets[split]["y"].append(y)
                buckets[split]["subs"].append(f"S{subject:02d}")
        print(f"  subject {subject} done", flush=True)

    exp = cfg.get("expected", {})
    for split, b in buckets.items():
        if not b["X"]:
            raise RuntimeError(f"{split}: no trials")
        X = norm_div100(np.concatenate(b["X"])).astype(np.float32)
        y = np.concatenate(b["y"]).astype(np.int64)
        assert_finite(X, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(b["subs"])), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    # subjects appear in every split by design (within-subject, cross-session),
    # so check_disjoint() is deliberately not called here
    n_sub = len(ds.subject_list)
    man.qc = {
        "n_excluded": len(man.excluded),
        "n_subjects": n_sub,
        "split_is_subject_disjoint": False,
        "split_basis": "session/run (cross-session, within-subject)",
        "expected_train_trials": exp.get("train_trials_per_subject", 0) * n_sub,
        "actual_train_trials": man.splits["train"]["n_windows"],
        "expected_val_trials": exp.get("val_trials_per_subject", 0) * n_sub,
        "actual_val_trials": man.splits["val"]["n_windows"],
        "expected_test_trials": exp.get("trials_per_session", 0) * n_sub,
        "actual_test_trials": man.splits["test"]["n_windows"],
    }
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
