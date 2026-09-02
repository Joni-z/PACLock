"""CHB-MIT v1.0.0 -> windowed npy + manifest. Protocol: docs/PROTOCOLS.md sec.4.

    python -m preprocessing.chbmit --config configs/datasets/chbmit.yaml [--jobs 32]

Two things here differ from the published TFM setup, both deliberate and both
required by the protocol:

* **Strict subject-disjoint split.** chb21 is the same person as chb01, so both
  go to train; val is chb20/chb22 and test is chb23/chb24.
* **Corrected overlap labelling.** A window is positive iff it intersects a
  seizure by more than zero samples. TFM's original test missed seizures that
  fully covered a window.

The protocol specifies no filtering for this corpus -- only resampling 256->200
-- so none is applied. Do not add a band-pass "for consistency" with the TUH
scripts; that would silently change the data all models see.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from functools import partial
from multiprocessing import Pool

import mne
import numpy as np

from paclock_bench.paths import expand
import yaml

from .common import (
    Manifest,
    assert_finite,
    intervals_overlap_labels,
    norm_q95,
    resample_to,
    save_split,
    sha256_file,
    window_signal,
)


def parse_summary(path: str) -> dict[str, list[tuple[float, float]]]:
    """chbXX-summary.txt -> {edf_filename: [(start_sec, end_sec), ...]}.

    Handles both layouts in the corpus: the common
    ``Seizure Start Time: N seconds`` and chb24's numbered
    ``Seizure 1 Start Time: N seconds``.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    cur = None
    starts: list[float] = []
    ends: list[float] = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                if cur is not None:
                    out[cur] = list(zip(starts, ends))
                cur = line.split(":", 1)[1].strip()
                starts, ends = [], []
            elif "Start Time:" in line and "Seizure" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    starts.append(float(m.group(1)))
            elif "End Time:" in line and "Seizure" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    ends.append(float(m.group(1)))
    if cur is not None:
        out[cur] = list(zip(starts, ends))
    return out


def pick_channels(raw: mne.io.BaseRaw, wanted: list[str]) -> np.ndarray:
    """CHB-MIT EDFs are already bipolar. Select the 16 named channels in order.

    Most CHB-MIT files list 'T8-P8' twice; MNE de-duplicates by appending an
    index, so the channels arrive as 'T8-P8-0' and 'T8-P8-1'. Matching only on
    the exact name would drop the montage entirely, so a wanted name also
    accepts a '<name>-<digit>' variant, lowest index first (the two copies are
    identical signals).
    """
    def norm(s: str) -> str:
        return s.upper().replace(" ", "")

    index: dict[str, int] = {}
    for i, name in enumerate(raw.ch_names):
        index.setdefault(norm(name), i)

    picked, missing = [], []
    for c in wanted:
        key = norm(c)
        if key in index:
            picked.append(index[key])
            continue
        variants = sorted(k for k in index
                          if k.startswith(key + "-") and k[len(key) + 1:].isdigit())
        if variants:
            picked.append(index[variants[0]])
        else:
            missing.append(c)
    if missing:
        raise KeyError(f"missing channels: {missing}")
    data = raw.get_data()
    return np.stack([data[i] for i in picked])


def process_one(item: tuple[str, list[tuple[float, float]]], cfg: dict):
    edf_path, seizures = item
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        sig = pick_channels(raw, cfg["channels"]) * 1e6          # volts -> uV
        del raw

        fs_out = cfg["sample_rate"]
        sig = resample_to(sig.astype(np.float64), fs, fs_out)    # no filtering: see docstring
        sig = np.ascontiguousarray(sig, dtype=np.float32)

        win = int(cfg["window_sec"] * fs_out)
        stride = int(cfg["stride_sec"] * fs_out)
        X, tail = window_signal(sig, win, stride)
        if len(X) == 0:
            return edf_path, None, None, 0, 0, None, "shorter than one window"
        y = intervals_overlap_labels(len(X), win, stride, seizures, fs_out)

        # Extra positives densely sampled around each seizure, all forced positive.
        n_aug = 0
        aug_cfg = cfg.get("augment_seizure", {})
        if aug_cfg.get("enabled") and seizures:
            a_stride = int(aug_cfg["stride_sec"] * fs_out)
            extra = []
            for (s, e) in seizures:
                a = int((s - aug_cfg["pre_sec"]) * fs_out)
                b = int((e + aug_cfg["post_sec"]) * fs_out)
                a = max(a, 0)
                for st in range(a, min(b, sig.shape[1]) - win + 1, a_stride):
                    extra.append(sig[:, st:st + win])
            if extra:
                extra_arr = np.stack(extra)
                n_aug = len(extra_arr)
                X = np.concatenate([X, extra_arr])
                y = np.concatenate([y, np.ones(n_aug, dtype=np.int64)])

        X = norm_q95(X).astype(np.float32)
        assert_finite(X, os.path.basename(edf_path))
        return edf_path, X, y, tail, n_aug, sha256_file(edf_path), None
    except Exception as e:                                       # noqa: BLE001
        return edf_path, None, None, 0, 0, None, f"{type(e).__name__}: {e}"


def collect_case(root: str, case: str) -> list[tuple[str, list]]:
    """All EDFs for one chbXX case, paired with their seizure intervals."""
    d = os.path.join(root, case)
    summary = os.path.join(d, f"{case}-summary.txt")
    ann = parse_summary(summary) if os.path.exists(summary) else {}
    items = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".edf"):
            items.append((os.path.join(d, f), ann.get(f, [])))
    return items


def run_group(items, cfg, jobs, man, tag):
    X_all, y_all, cases, tail_total, aug_total = [], [], [], 0, 0
    with Pool(jobs) as pool:
        for path, X, y, tail, n_aug, sha, err in pool.imap_unordered(
            partial(process_one, cfg=cfg), items, chunksize=1
        ):
            if err is not None:
                man.exclude(os.path.basename(path), err, split=tag)
                continue
            X_all.append(X)
            y_all.append(y)
            cases.append(os.path.basename(os.path.dirname(path)))
            tail_total += tail
            aug_total += n_aug
            man.raw_sha256[os.path.basename(path)] = sha
    if not X_all:
        raise RuntimeError(f"{tag}: every recording was excluded")
    return (np.concatenate(X_all).astype(np.float32, copy=False),
            np.concatenate(y_all).astype(np.int64, copy=False), cases, tail_total, aug_total)


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

    for split in ("train", "val", "test"):
        items = [it for case in sp[split] for it in collect_case(root, case)]
        print(f"[{split}] {len(items)} recordings from {sp[split]}", flush=True)
        X, y, cases, tail, n_aug = run_group(items, cfg, args.jobs, man, split)
        save_split(out_dir, split, X, y)
        counts = Counter(y.tolist())
        man.add_split(split, subjects=sp[split], n_windows=len(X),
                      class_counts=counts, discarded_tail=tail,
                      n_recordings=len(items), shape=list(X.shape[1:]),
                      n_augmented_positive=n_aug,
                      positive_rate=float(counts.get(1, 0) / max(len(y), 1)))

    man.check_disjoint()
    man.qc = {
        "n_excluded": len(man.excluded),
        "note_chb01_chb21_same_subject": "both in train, per protocol",
        "filtering": "none (protocol specifies resample only)",
    }
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
