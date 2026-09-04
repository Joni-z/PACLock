"""TUSZ v2.0.6 -> seizure-TYPE classification windows (morphology task on the same
recordings, windows and protocol as the detection task in preprocessing/tusz.py).

    python -m preprocessing.tusz_type --config configs/datasets/tusz_type.yaml [--jobs 16]

Why this exists (2026-09-05). The detection task labels a window by STATE
(seizure vs background); this task labels the same 10 s seizure windows by the
MORPHOLOGY of the seizure (focal, generalized, complex partial, absence, ...)
using TUSZ's per-channel ``.csv`` term annotations. Everything else -- filtering,
montage, windowing, splits, normalization -- is identical to tusz.py, so the
contrast "does coupling help the state label / the morphology label" moves one
variable: the label type.

Window label = the seizure type with the largest total overlap (seconds x
channels) among the annotated seizure intervals overlapping the window. Windows
with no seizure overlap are dropped (they belong to the detection task). Types
not in ``classes`` are dropped and counted in the manifest.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
import yaml

from paclock_bench.paths import expand

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    preprocess_signal,
    save_split,
    sha256_file,
    window_signal,
)
from .tuh_common import MissingChannels, load_bipolar_uv, subject_of
from .tusz import _filter_args, list_edfs

BCKG = {"bckg", "background", ""}


def read_csv_types(path: str) -> list[tuple[float, float, str]]:
    """Per-channel (start, stop, type) rows of a ``.csv`` term annotation file."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("channel,"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            lab = parts[3].strip().lower()
            if lab in BCKG:
                continue
            out.append((float(parts[1]), float(parts[2]), lab))
    return out


def window_types(n_win: int, win: int, stride: int, rows, fs: int) -> list[str | None]:
    """Majority seizure type per window by overlap (seconds x channels); None if no seizure."""
    labels = []
    for w in range(n_win):
        a, b = w * stride / fs, (w * stride + win) / fs
        acc: Counter = Counter()
        for s, e, lab in rows:
            ov = min(b, e) - max(a, s)
            if ov > 0:
                acc[lab] += ov
        labels.append(acc.most_common(1)[0][0] if acc else None)
    return labels


def process_one(edf_path: str, cfg: dict):
    ann = edf_path[:-4] + ".csv"
    try:
        if not os.path.exists(ann):
            return edf_path, None, None, None, "no .csv annotation"
        rows = read_csv_types(ann)
        if not rows:
            return edf_path, None, None, None, None          # no seizures: nothing for this task
        sig, fs = load_bipolar_uv(edf_path)
        fs_out = cfg["sample_rate"]
        sig = preprocess_signal(sig, fs, fs_out=fs_out, **_filter_args(cfg))
        win = int(cfg["window_sec"] * fs_out)
        stride = int(cfg["stride_sec"] * fs_out)
        X, _ = window_signal(sig, win, stride)
        if len(X) == 0:
            return edf_path, None, None, None, "shorter than one window"
        types = window_types(len(X), win, stride, rows, fs_out)
        keep = [i for i, t in enumerate(types) if t is not None]
        if not keep:
            return edf_path, None, None, None, None
        X = norm_div100(X[keep]).astype(np.float32, copy=False)
        assert_finite(X, os.path.basename(edf_path))
        return edf_path, X, [types[i] for i in keep], sha256_file(edf_path), None
    except MissingChannels as e:
        return edf_path, None, None, None, str(e)
    except Exception as e:                              # noqa: BLE001
        return edf_path, None, None, None, f"{type(e).__name__}: {e}"


def run_group(paths, cfg, jobs, man, tag, classes):
    idx = {c: i for i, c in enumerate(classes)}
    X_all, y_all, subs, dropped = [], [], [], Counter()
    with Pool(jobs) as pool:
        for path, X, types, sha, err in pool.imap_unordered(partial(process_one, cfg=cfg), paths, chunksize=2):
            if err is not None:
                man.exclude(os.path.basename(path), err, split=tag)
                continue
            if X is None:
                continue
            keep = [i for i, t in enumerate(types) if t in idx]
            for t in (types[i] for i in range(len(types)) if types[i] not in idx):
                dropped[t] += 1
            if not keep:
                continue
            X_all.append(X[keep])
            y_all.append(np.array([idx[types[i]] for i in keep], dtype=np.int64))
            subs.append(subject_of(path))
            man.raw_sha256[os.path.basename(path)] = sha
    if not X_all:
        raise RuntimeError(f"{tag}: no seizure windows of the requested classes")
    return (np.concatenate(X_all).astype(np.float32, copy=False),
            np.concatenate(y_all), subs, dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    sp = cfg["split"]
    classes = [c.lower() for c in cfg["classes"]]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)
    groups = {"train": list_edfs(os.path.join(root, sp["train"])),
              "val": list_edfs(os.path.join(root, sp["val"])),
              "test": list_edfs(os.path.join(root, sp["test"]))}
    for split, paths in groups.items():
        print(f"[{split}] {len(paths)} recordings", flush=True)
        X, y, subs, dropped = run_group(paths, cfg, args.jobs, man, split, classes)
        save_split(out_dir, split, X, y)
        counts = Counter(y.tolist())
        print(f"[{split}] {X.shape} -> {out_dir}  labels={[counts.get(i, 0) for i in range(len(classes))]}"
              f"  dropped_types={dict(dropped)}", flush=True)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts={classes[k]: v for k, v in counts.items()},
                      dropped_types=dict(dropped), n_recordings=len(paths), shape=list(X.shape[1:]))
    man.check_disjoint()
    man.qc = {"n_excluded": len(man.excluded), "classes": classes}
    man.save(os.path.join(out_dir, "manifest.json"))
    print(f"[manifest] {out_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
