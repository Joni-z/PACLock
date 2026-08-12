"""FACED -> 10 s window npy + manifest. Protocol: docs/PROTOCOLS.md sec.8.

    python -m preprocessing.faced --config configs/datasets/faced.yaml

Consumes the **official pre-processed** release (Synapse syn50614194), one
``.pkl`` per subject holding ``(28 videos, 32 channels, 7500 samples)`` at
250 Hz. The official pipeline already applied the 0.05-47 Hz band-pass, bad
channel interpolation, ICA eye-movement removal and common-average reference,
and already cropped each video to its final 30 s -- so this script only
resamples 250 -> 200 Hz and cuts windows. **Do not add another filter here**;
filtering twice would silently change the data every model sees, and we cannot
reproduce the official ICA step anyway.

If you have the *raw* FACED release instead, this script is the wrong tool: the
protocol pins the pre-processed version, and replicating the official ICA
cleanup bit-for-bit is not feasible.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import Counter

import numpy as np
import yaml

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    resample_to,
    save_split,
    sha256_file,
)

FNAME_RE = re.compile(r"sub(\d+)", re.IGNORECASE)


def subject_id(fname: str) -> int:
    """Subject index from a filename like ``sub000.pkl``."""
    m = FNAME_RE.search(os.path.basename(fname))
    if not m:
        raise ValueError(f"cannot read a subject id from {fname}")
    return int(m.group(1))


def build_video_labels(cfg: dict) -> np.ndarray:
    """Video index -> emotion class.

    The 28 videos are laid out in blocks, one block per emotion, with sizes
    ``videos_per_label`` = 3,3,3,3,4,3,3,3,3 (neutral has the extra one). The
    protocol forbids reordering these to suit a model.
    """
    counts = cfg["videos_per_label"]
    labels = np.concatenate([np.full(c, i, dtype=np.int64)
                             for i, c in enumerate(counts)])
    if len(labels) != cfg["n_videos"]:
        raise ValueError(
            f"videos_per_label sums to {len(labels)}, expected {cfg['n_videos']}")
    return labels


def load_subject(path: str, cfg: dict) -> np.ndarray:
    """One subject .pkl -> (28, 32, 7500) float array, shape-checked."""
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (video, channel, time) array, got {arr.shape}")
    n_vid, n_ch, n_t = arr.shape
    if n_vid != cfg["n_videos"] or n_ch != cfg["n_channels"]:
        raise ValueError(
            f"expected ({cfg['n_videos']}, {cfg['n_channels']}, T), got {arr.shape}")
    expected_t = int(cfg["trial_sec"] * cfg["source_rate"])
    if abs(n_t - expected_t) > cfg["source_rate"]:            # tolerate < 1 s slack
        raise ValueError(f"expected ~{expected_t} samples at {cfg['source_rate']} Hz, "
                         f"got {n_t}")
    return arr


def windows_for_subject(arr: np.ndarray, cfg: dict):
    """(28, 32, 7500) @250 Hz -> (n_win, 32, 2000) @200 Hz plus per-window video id."""
    fs_out = cfg["sample_rate"]
    win = int(cfg["window_sec"] * fs_out)
    stride = int(cfg["stride_sec"] * fs_out)

    X, vid = [], []
    for v in range(arr.shape[0]):
        sig = resample_to(arr[v], cfg["source_rate"], fs_out)   # (32, 6000)
        n = (sig.shape[1] - win) // stride + 1 if sig.shape[1] >= win else 0
        for k in range(n):
            X.append(sig[:, k * stride: k * stride + win])
            vid.append(v)
    if not X:
        raise ValueError("no windows produced")
    return np.stack(X).astype(np.float32), np.array(vid, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # Path overrides so this can run off-cluster (e.g. on a laptop that has the
    # download) without editing the frozen config. Everything that affects the
    # data itself still comes from the config.
    ap.add_argument("--raw-root", default=None,
                    help="override cfg['raw_root'] (directory holding the .pkl)")
    ap.add_argument("--out-dir", default=None,
                    help="override cfg['out_dir'] (where the npy are written)")
    # Accepted for interface parity: slurm/preprocess.slurm passes --jobs to
    # every dataset. This script is single-process (one pass over the files),
    # so the value is recorded in the manifest but not used to fan out.
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = args.raw_root or cfg["raw_root"]
    out_dir = args.out_dir or cfg["out_dir"]
    cfg["raw_root"], cfg["out_dir"] = root, out_dir     # keep the manifest honest
    sp = cfg["split"]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    if not os.path.isdir(root):
        raise SystemExit(f"{root} not found -- fetch the official pre-processed "
                         f"release first (see docs/STATUS.md)")

    pkls = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(".pkl"):
                pkls.append(os.path.join(dirpath, f))
    pkls.sort()
    if not pkls:
        raise SystemExit(f"no .pkl under {root}. If you only have the raw release, "
                         f"see this module's docstring -- the protocol needs the "
                         f"pre-processed one.")

    video_labels = build_video_labels(cfg)

    def split_of(sub: int) -> str | None:
        for name in ("train", "val", "test"):
            lo, hi = sp[name]
            if lo <= sub <= hi:
                return name
        return None

    buckets = {k: {"X": [], "y": [], "subs": []} for k in ("train", "val", "test")}
    per_subject_windows = {}

    for path in pkls:
        try:
            sub = subject_id(path)
        except ValueError as e:
            man.exclude(os.path.basename(path), str(e))
            continue
        # Subject-level split happens BEFORE windowing, so every window of a
        # subject -- and therefore of a video -- lands in exactly one split.
        split = split_of(sub)
        if split is None:
            man.exclude(os.path.basename(path), f"subject {sub} outside all split ranges")
            continue
        try:
            arr = load_subject(path, cfg)
            X, vid = windows_for_subject(arr, cfg)
        except Exception as e:                                  # noqa: BLE001
            man.exclude(os.path.basename(path), f"{type(e).__name__}: {e}", split=split)
            print(f"  sub{sub:03d}: EXCLUDED ({e})", flush=True)
            continue
        y = video_labels[vid]
        buckets[split]["X"].append(X)
        buckets[split]["y"].append(y)
        buckets[split]["subs"].append(f"sub{sub:03d}")
        per_subject_windows[f"sub{sub:03d}"] = int(len(X))
        man.raw_sha256[os.path.basename(path)] = sha256_file(path)
        print(f"  sub{sub:03d} [{split}] {X.shape}", flush=True)

    for split, b in buckets.items():
        if not b["X"]:
            raise RuntimeError(f"{split}: no subjects processed")
        X = norm_div100(np.concatenate(b["X"])).astype(np.float32)
        y = np.concatenate(b["y"]).astype(np.int64)
        assert_finite(X, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(b["subs"]), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    man.check_disjoint()          # subject-disjoint by construction; verify it

    exp = cfg.get("expected", {})
    per_sub = exp.get("trials_per_subject", 0) * exp.get("windows_per_trial", 0)
    odd = {s: n for s, n in per_subject_windows.items() if per_sub and n != per_sub}
    man.qc = {
        "n_excluded": len(man.excluded),
        "n_subjects_found": len(per_subject_windows),
        "n_subjects_expected": cfg["n_subjects"],
        "windows_per_subject_expected": per_sub,
        "subjects_with_unexpected_window_count": odd,
        "split_is_subject_disjoint": True,
        "filtering": "none (official release is already filtered; resample only)",
        "labels": cfg["labels"],
    }
    man.save(os.path.join(out_dir, "manifest.json"))
    if odd:
        print(f"WARNING: {len(odd)} subjects have an unexpected window count", flush=True)


if __name__ == "__main__":
    main()
