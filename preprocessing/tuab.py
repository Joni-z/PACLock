"""TUAB v3.0.1 -> windowed npy + manifest. Protocol: docs/PROTOCOLS.md sec.1.

    python -m preprocessing.tuab --config configs/datasets/tuab.yaml [--jobs 32]

Label is recording-level (normal=0 / abnormal=1) and comes from the directory,
not from any annotation file.
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
    window_signal,
)
from .tuh_common import MissingChannels, load_bipolar_uv, sort_subject_split, subject_of


def process_one(path_label: tuple[str, int], cfg: dict):
    """Worker: one EDF -> (windows, labels, tail, sha, error)."""
    path, label = path_label
    try:
        sig, fs = load_bipolar_uv(path)
        sig = preprocess_signal(
            sig, fs,
            fs_out=cfg["sample_rate"],
            **_filter_args(cfg),
        )
        win = int(cfg["window_sec"] * cfg["sample_rate"])
        stride = int(cfg["stride_sec"] * cfg["sample_rate"])
        X, tail = window_signal(sig, win, stride)
        if len(X) == 0:
            return path, None, None, 0, None, "shorter than one window"
        X = norm_div100(X).astype(np.float32)
        assert_finite(X, os.path.basename(path))
        y = np.full(len(X), label, dtype=np.int64)
        return path, X, y, tail, sha256_file(path), None
    except MissingChannels as e:
        return path, None, None, 0, None, str(e)
    except Exception as e:                                   # noqa: BLE001
        return path, None, None, 0, None, f"{type(e).__name__}: {e}"


def collect(root: str, split_dir: str, cfg: dict) -> list[tuple[str, int]]:
    """(edf_path, label) for one official split directory. abnormal=1, normal=0."""
    out = []
    for name, label in (("normal", 0), ("abnormal", 1)):
        d = os.path.join(root, split_dir, name, cfg["channel_std"])
        if not os.path.isdir(d):
            raise FileNotFoundError(d)
        for f in sorted(os.listdir(d)):
            if f.endswith(".edf"):
                out.append((os.path.join(d, f), label))
    return out


def run_group(items, cfg, jobs, manifest, tag):
    """Process a list of (path, label) and concatenate. Exclusions are logged."""
    X_all, y_all, subs, tail_total = [], [], [], 0
    with Pool(jobs) as pool:
        for path, X, y, tail, sha, err in pool.imap_unordered(
            partial(process_one, cfg=cfg), items, chunksize=4
        ):
            if err is not None:
                manifest.exclude(os.path.basename(path), err, split=tag)
                continue
            X_all.append(X)
            y_all.append(y)
            subs.append(subject_of(path))
            tail_total += tail
            manifest.raw_sha256[os.path.basename(path)] = sha
    if not X_all:
        raise RuntimeError(f"{tag}: every recording was excluded")
    X = np.concatenate(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.int64)
    return X, y, subs, tail_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = cfg["raw_root"], cfg["out_dir"]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    # ---- split by subject inside the official train dir, per class ---------- #
    train_items = collect(root, "train", cfg)
    test_items = collect(root, "eval", cfg)

    # per_class: normal and abnormal are sorted and cut independently, so the
    # 80/20 ratio holds within each class rather than only overall.
    assigned: dict[str, list] = {"train": [], "val": []}
    for label in (0, 1):
        cls = [it for it in train_items if it[1] == label]
        tr_sub, va_sub = sort_subject_split([subject_of(p) for p, _ in cls],
                                            cfg["split"]["ratio"])
        tr_set, va_set = set(tr_sub), set(va_sub)
        assigned["train"] += [it for it in cls if subject_of(it[0]) in tr_set]
        assigned["val"] += [it for it in cls if subject_of(it[0]) in va_set]

    for split, items in (("train", assigned["train"]),
                         ("val", assigned["val"]),
                         ("test", test_items)):
        print(f"[{split}] {len(items)} recordings", flush=True)
        X, y, subs, tail = run_group(items, cfg, args.jobs, man, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=Counter(y.tolist()), discarded_tail=tail,
                      n_recordings=len(items) - sum(
                          1 for e in man.excluded if e.get("split") == split),
                      shape=list(X.shape[1:]))

    # The protocol splits normal and abnormal independently. 54 TUAB subjects
    # have both a normal and an abnormal recording (the same patient scored
    # differently at different sessions), so a handful land in train via one
    # class and val via the other. This is not our deviation: BIOT's
    # `datasets/TUAB/process.py` and CBraMod's `preprocessing_tuab.py` both
    # split per class, and every published TUAB number we calibrate against was
    # produced this way. Matching them is the point of the group-A calibration,
    # so the overlap is kept and recorded rather than silently repaired.
    #
    # It is train<->val only: the official eval split shares no subject with
    # train, so the reported test metric is unaffected. The overlap can
    # slightly optimistically bias checkpoint selection.
    man.qc = {
        "n_excluded": len(man.excluded),
        "expected_train_recordings": 2718,
        "expected_eval_recordings": 276,
        "actual_train_recordings": len(train_items),
        "actual_eval_recordings": len(test_items),
        "split_rule": "per-class subject sort, 80/20 (BIOT/CBraMod convention)",
        "subject_overlap_expected": (
            "train<->val only; consequence of per-class splitting, matches "
            "published practice. test (official eval) is subject-disjoint."
        ),
    }
    overlap = man.check_disjoint(strict=False)
    man.qc["n_overlapping_subjects"] = sum(f["n_subjects"] for f in overlap)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
