"""Sleep-EDF Expanded v1.0.0 (SC subset) -> epoch npy + manifest.
Protocol: docs/PROTOCOLS.md sec.5.

    python -m preprocessing.sleepedf --config configs/datasets/sleepedf.yaml

Stays at the native 100 Hz -- the protocol explicitly does not resample. The
normalisation is per-channel mean/std computed over the retained *train* epochs
only and then applied unchanged to val/test, so this script processes all splits
before normalising anything.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import mne
import numpy as np

from paclock_bench.paths import expand
import yaml

from .common import (
    highpass,
    Manifest,
    assert_finite,
    bandpass,
    compute_train_stats,
    norm_with_stats,
    save_split,
    sha256_file,
)

# SC4ssNE0: 'ss' is the subject number, 'N' the night. Two nights per subject.
FNAME_RE = re.compile(r"^SC4(\d{2})(\d)")


def subject_night(fname: str) -> tuple[int, int]:
    m = FNAME_RE.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"unexpected Sleep-EDF filename: {fname}")
    return int(m.group(1)), int(m.group(2))


def pair_files(root: str) -> list[tuple[int, str, str]]:
    """(subject, psg_path, hypnogram_path) for every recording in the SC subset."""
    psgs = sorted(f for f in os.listdir(root) if f.endswith("-PSG.edf"))
    hyps = sorted(f for f in os.listdir(root) if f.endswith("-Hypnogram.edf"))
    # the hypnogram's 8th character differs from the PSG's, so match on the first 7
    by_stem = {h[:7]: h for h in hyps}
    out = []
    for p in psgs:
        h = by_stem.get(p[:7])
        if h is None:
            continue
        sub, _night = subject_night(p)
        out.append((sub, os.path.join(root, p), os.path.join(root, h)))
    return out


def epochs_from_recording(psg: str, hyp: str, cfg: dict):
    """One PSG+hypnogram pair -> (N, C, 3000) epochs and labels.

    Crops to 30 min either side of the sleep period before epoching, per protocol:
    the SC recordings contain many hours of pre/post-bed wake that would otherwise
    dominate class W.
    """
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR",
                              stim_channel=None)
    fs = float(raw.info["sfreq"])
    missing = [c for c in cfg["channels"] if c not in raw.ch_names]
    if missing:
        raise KeyError(f"missing channels {missing}")
    sig = raw.get_data(picks=cfg["channels"]) * 1e6                # volts -> uV
    del raw

    # PAC protocol drops the band-pass for a high-pass; see
    # scripts/make_pac_protocol.py. One implementation serves both so
    # only the filter settings can differ between the two protocols.
    band = cfg.get("band")
    sig = (bandpass(sig.astype(np.float64), fs, band[0], band[1]) if band
           else highpass(sig.astype(np.float64), fs, cfg["hp"]))

    ann = mne.read_annotations(hyp)
    epoch_len = int(cfg["window_sec"] * fs)
    label_map = cfg["label_map"]
    drop = set(cfg["drop_labels"])

    # hypnogram annotations are variable-length runs of one stage; expand to epochs
    items: list[tuple[int, int]] = []                               # (start_sample, label)
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if desc in drop or desc not in label_map:
            continue
        lab = label_map[desc]
        n = int(round(dur / cfg["window_sec"]))
        base = int(round(onset * fs))
        for k in range(n):
            items.append((base + k * epoch_len, lab))
    if not items:
        raise ValueError("no scored epochs")

    crop = cfg.get("crop_wake", {})
    if crop.get("enabled"):
        pad = int(crop["minutes"] * 60 * fs)
        sleep = [s for s, lab in items if lab != 0]
        if sleep:
            lo, hi = min(sleep) - pad, max(sleep) + pad + epoch_len
            items = [(s, lab) for s, lab in items if lo <= s < hi]

    X, y = [], []
    T = sig.shape[1]
    for s, lab in items:
        if s < 0 or s + epoch_len > T:
            continue
        X.append(sig[:, s:s + epoch_len])
        y.append(lab)
    if not X:
        raise ValueError("no epochs survived cropping")
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

    want = {s: name for name in ("train", "val", "test") for s in sp[name]}
    buckets: dict[str, dict[str, list]] = {
        k: {"X": [], "y": [], "subs": []} for k in ("train", "val", "test")
    }

    for sub, psg, hyp in pair_files(root):
        split = want.get(sub)
        if split is None:
            man.exclude(os.path.basename(psg), f"subject {sub} not in any split")
            continue
        try:
            X, y = epochs_from_recording(psg, hyp, cfg)
        except Exception as e:                                     # noqa: BLE001
            man.exclude(os.path.basename(psg), f"{type(e).__name__}: {e}",
                        split=split)
            continue
        buckets[split]["X"].append(X)
        buckets[split]["y"].append(y)
        buckets[split]["subs"].append(sub)
        man.raw_sha256[os.path.basename(psg)] = sha256_file(psg)
        print(f"  sub {sub:02d} [{split}] {X.shape}", flush=True)

    data = {}
    for split, b in buckets.items():
        if not b["X"]:
            raise RuntimeError(f"{split}: no recordings")
        data[split] = (np.concatenate(b["X"]), np.concatenate(b["y"]), b["subs"])

    # train statistics only; val/test are transformed with them, never their own
    mean, std = compute_train_stats(data["train"][0])
    man.qc = {
        "norm_mean_per_channel": mean.ravel().tolist(),
        "norm_std_per_channel": std.ravel().tolist(),
        "norm_source": "train split only",
        "resampled": False,
        "n_excluded": len(man.excluded),
    }

    for split, (X, y, subs) in data.items():
        X = norm_with_stats(X, mean, std).astype(np.float32)
        assert_finite(X, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    man.check_disjoint()
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
