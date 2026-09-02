"""TUSZ v2.0.6 -> windowed npy + manifest. Protocol: docs/PROTOCOLS.md sec.3.

    python -m preprocessing.tusz --config configs/datasets/tusz.yaml [--jobs 32]

Official subject-disjoint train/dev/eval map to train/val/test. Windows are
labelled positive iff they overlap any seizure interval by more than zero
samples, on half-open [start, end). All negatives are kept.
"""

from __future__ import annotations

def _filter_args(cfg):
    """Filtering arguments, supporting both the frozen and the PAC protocols.

    The frozen configs carry ``band: [lo, hi]``; the PAC-methodology configs
    (scripts/make_pac_protocol.py) drop ``band`` and carry ``hp`` instead,
    because a low-pass ceiling truncates PACLock's filterbank just as the notch
    punctures it. Reading both here keeps one preprocessing implementation for
    the two protocols, so nothing but the filter settings can differ between
    them -- which is the whole point of the contrast.
    """
    band = cfg.get("band")
    return {
        "band": tuple(band) if band else None,
        "hp": cfg.get("hp"),
        "notch_freq": cfg.get("notch"),
    }


import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np

from paclock_bench.paths import expand
import yaml

from .common import (
    Manifest,
    assert_finite,
    intervals_overlap_labels,
    norm_div100,
    preprocess_signal,
    save_split,
    sha256_file,
    window_signal,
)
from .tuh_common import MissingChannels, load_bipolar_uv, subject_of


def read_csv_bi(path: str) -> list[tuple[float, float]]:
    """Return seizure intervals in seconds from a ``.csv_bi`` annotation file."""
    intervals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("channel,"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            label = parts[3].strip().lower()
            if label in ("seiz", "seizure"):
                intervals.append((float(parts[1]), float(parts[2])))
    return intervals


def process_one(edf_path: str, cfg: dict):
    ann = edf_path[:-4] + ".csv_bi"
    try:
        if not os.path.exists(ann):
            return edf_path, None, None, 0, None, "no .csv_bi annotation"
        intervals = read_csv_bi(ann)

        sig, fs = load_bipolar_uv(edf_path)
        fs_out = cfg["sample_rate"]
        sig = preprocess_signal(sig, fs, fs_out=fs_out,
                                **_filter_args(cfg))

        win = int(cfg["window_sec"] * fs_out)
        stride = int(cfg["stride_sec"] * fs_out)
        X, tail = window_signal(sig, win, stride)
        if len(X) == 0:
            return edf_path, None, None, 0, None, "shorter than one window"

        y = intervals_overlap_labels(len(X), win, stride, intervals, fs_out)
        X = norm_div100(X).astype(np.float32)
        assert_finite(X, os.path.basename(edf_path))
        return edf_path, X, y, tail, sha256_file(edf_path), None
    except MissingChannels as e:
        # protocol: exclude the whole recording and log it
        return edf_path, None, None, 0, None, str(e)
    except Exception as e:                              # noqa: BLE001
        return edf_path, None, None, 0, None, f"{type(e).__name__}: {e}"


def list_edfs(split_root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(split_root):
        for f in files:
            if f.endswith(".edf"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def run_group(paths, cfg, jobs, man, tag):
    X_all, y_all, subs, tail_total = [], [], [], 0
    with Pool(jobs) as pool:
        for path, X, y, tail, sha, err in pool.imap_unordered(
            partial(process_one, cfg=cfg), paths, chunksize=2
        ):
            if err is not None:
                man.exclude(os.path.basename(path), err, split=tag)
                continue
            X_all.append(X)
            y_all.append(y)
            subs.append(subject_of(path))
            tail_total += tail
            man.raw_sha256[os.path.basename(path)] = sha
    if not X_all:
        raise RuntimeError(f"{tag}: every recording was excluded")
    return (np.concatenate(X_all).astype(np.float32),
            np.concatenate(y_all).astype(np.int64), subs, tail_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    sp = cfg["split"]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    groups = {
        "train": list_edfs(os.path.join(root, sp["train"])),
        "val": list_edfs(os.path.join(root, sp["val"])),
        "test": list_edfs(os.path.join(root, sp["test"])),
    }

    for split, paths in groups.items():
        print(f"[{split}] {len(paths)} recordings", flush=True)
        X, y, subs, tail = run_group(paths, cfg, args.jobs, man, split)
        save_split(out_dir, split, X, y)
        counts = Counter(y.tolist())
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=counts, discarded_tail=tail,
                      n_recordings=len(paths), shape=list(X.shape[1:]),
                      positive_rate=float(counts.get(1, 0) / max(len(y), 1)))

    man.check_disjoint()
    man.qc = {
        "n_excluded": len(man.excluded),
        "n_recordings_total": sum(len(v) for v in groups.values()),
        "n_subjects_total": sum(v["n_subjects"] for v in man.splits.values()),
    }
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
