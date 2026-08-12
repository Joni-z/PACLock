"""TUEV v2.0.1 -> event-centred npy + manifest. Protocol: docs/PROTOCOLS.md sec.2.

    python -m preprocessing.tuev --config configs/datasets/tuev.yaml [--jobs 32]

Not a sliding window: each ``.rec`` annotation row produces exactly one 5 s
sample (2 s before the event + the ~1 s event + 2 s after). Label codes 1..6 in
the file map to 0..5.
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
import yaml

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    preprocess_signal,
    save_split,
    sha256_file,
)
from .tuh_common import MissingChannels, load_bipolar_uv, sort_subject_split


def read_rec(path: str) -> np.ndarray:
    """``.rec`` rows are ``channel,start_sec,stop_sec,label``."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            rows.append((int(float(parts[0])), float(parts[1]),
                         float(parts[2]), int(float(parts[3]))))
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, 4))


def process_one(edf_path: str, cfg: dict):
    """One EDF + its .rec -> (N, 16, 1000) samples with labels 0..5."""
    rec_path = edf_path[:-4] + ".rec"
    try:
        if not os.path.exists(rec_path):
            return edf_path, None, None, None, 0, "no .rec annotation file"
        events = read_rec(rec_path)
        if len(events) == 0:
            return edf_path, None, None, None, 0, "empty .rec"

        sig, fs = load_bipolar_uv(edf_path)
        fs_out = cfg["sample_rate"]
        sig = preprocess_signal(sig, fs, fs_out=fs_out,
                                **_filter_args(cfg))

        win = int(cfg["window_sec"] * fs_out)          # 1000 @ 200 Hz
        pre = int(cfg["pre_sec"] * fs_out)
        T = sig.shape[1]

        X_list, y_list, odd = [], [], 0
        for _chan, start_s, stop_s, label in events:
            if not (1 <= label <= 6):
                continue
            if abs((stop_s - start_s) - 1.0) > 1e-6:
                odd += 1                                # QC: protocol assumes ~1 s
            # protocol: 2 s before the event start, 2 s after the event end.
            #
            # Events near either edge of the recording wrap around rather than
            # being dropped. That is what the reference implementation does --
            # it tiles the signal three times and indexes into the middle copy
            # (`np.concatenate([signals, signals, signals], axis=1)`), so an
            # event 1 s into the recording takes its leading context from the
            # tail. Modular indexing here is equivalent and does not triple the
            # memory.
            #
            # Skipping those events instead cost 5.4% of the corpus (106,394
            # against the 112,491 the protocol cites) and biased the sample:
            # recording-edge events are not a random subset.
            #
            # Anchor on the event start with a fixed `win` so every sample has
            # the same length. For the exactly-1 s annotations that make up the
            # corpus this is identical to the reference's start-2s..end+2s; for
            # the 881 that are not, the reference would produce a ragged array,
            # while this stays well defined.
            a = int(round(start_s * fs_out)) - pre
            idx = np.arange(a, a + win) % T
            seg = sig[:, idx]
            X_list.append(seg)
            y_list.append(int(label) - 1)               # 1..6 -> 0..5

        if not X_list:
            return edf_path, None, None, None, 0, "no usable events"
        X = norm_div100(np.stack(X_list)).astype(np.float32)
        assert_finite(X, os.path.basename(edf_path))
        y = np.array(y_list, dtype=np.int64)
        return edf_path, X, y, sha256_file(edf_path), odd, None
    except MissingChannels as e:
        return edf_path, None, None, None, 0, str(e)
    except Exception as e:                              # noqa: BLE001
        return edf_path, None, None, None, 0, f"{type(e).__name__}: {e}"


def list_subject_edfs(split_root: str) -> dict[str, list[str]]:
    """TUEV lays out one directory per subject."""
    out: dict[str, list[str]] = {}
    for sub in sorted(os.listdir(split_root)):
        d = os.path.join(split_root, sub)
        if not os.path.isdir(d):
            continue
        edfs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".edf"))
        if edfs:
            out[sub] = edfs
    return out


def run_group(paths, cfg, jobs, man, tag):
    X_all, y_all, subs, odd_total = [], [], [], 0
    with Pool(jobs) as pool:
        for path, X, y, sha, odd, err in pool.imap_unordered(
            partial(process_one, cfg=cfg), paths, chunksize=2
        ):
            if err is not None:
                man.exclude(os.path.basename(path), err, split=tag)
                continue
            X_all.append(X)
            y_all.append(y)
            subs.append(os.path.basename(os.path.dirname(path)))
            odd_total += odd
            man.raw_sha256[os.path.basename(path)] = sha
    if not X_all:
        raise RuntimeError(f"{tag}: every recording was excluded")
    return (np.concatenate(X_all).astype(np.float32),
            np.concatenate(y_all).astype(np.int64), subs, odd_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = cfg["raw_root"], cfg["out_dir"]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    train_subs = list_subject_edfs(os.path.join(root, "train"))
    test_subs = list_subject_edfs(os.path.join(root, "eval"))

    tr, va = sort_subject_split(list(train_subs), cfg["split"]["ratio"])
    groups = {
        "train": [p for s in tr for p in train_subs[s]],
        "val": [p for s in va for p in train_subs[s]],
        "test": [p for s in sorted(test_subs) for p in test_subs[s]],
    }

    odd_all = 0
    for split, paths in groups.items():
        print(f"[{split}] {len(paths)} recordings", flush=True)
        X, y, subs, odd = run_group(paths, cfg, args.jobs, man, split)
        odd_all += odd
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(paths), shape=list(X.shape[1:]))

    man.check_disjoint()
    total = sum(v["n_windows"] for v in man.splits.values())
    man.qc = {
        "n_excluded": len(man.excluded),
        "n_events_not_1s": odd_all,
        "edge_events": "wrapped, not dropped (reference tiling behaviour)",
        "expected_n_samples": cfg.get("expected", {}).get("n_samples"),
        "actual_n_samples": total,
        "n_train_subjects_official": len(train_subs),
        "n_eval_subjects_official": len(test_subs),
    }
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
