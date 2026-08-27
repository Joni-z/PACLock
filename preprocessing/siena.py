"""Siena Scalp EEG v1.0.0 -> windowed npy + manifest.

    python -m preprocessing.siena --config configs/datasets/siena.yaml

External seizure cohort (PhysioNet; 14 subjects, Unit of Neurology, Siena).
CHB-MIT's task shape on an out-of-corpus population: binary
seizure-vs-background windows, labels from the per-subject
``Seizures-list-PNxx.txt`` files, windows positive iff they overlap a
seizure interval (the same half-open overlap rule the TUSZ/CHB-MIT
protocols use -- common.intervals_overlap_labels).

The annotation format gives clock times (registration start, seizure
start/end, ``hh.mm.ss``); offsets are computed against each recording's own
registration start, wrapping past midnight where end < start. 19-channel
referential 10-20 subset in canonical order; 512 Hz native -> 200 Hz,
0.3-75 Hz band, 50 Hz notch (Italy), 10 s windows, div100.
Subject-disjoint sorted split.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict

import mne
import numpy as np
import yaml

from .common import (
    Manifest,
    assert_finite,
    intervals_overlap_labels,
    norm_div100,
    preprocess_signal,
    save_split,
    window_signal,
)
from .tuh_common import sort_subject_split
from paclock_bench.paths import expand

TIME_RE = r"(\d{1,2})[.:](\d{2})[.:](\d{2})"


def _secs(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_seizure_list(path: str) -> dict[str, list[tuple[float, float]]]:
    """filename -> [(seizure_start_sec, seizure_end_sec)] relative to file start.

    The lists interleave ``File name:``, ``Registration start time:``,
    ``Seizure start time:`` and ``Seizure end time:`` lines; a file can carry
    several seizures. Times are wall-clock; crossing midnight adds 24 h.
    """
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    fname, reg0 = None, None
    for line in open(path, errors="replace"):
        low = line.lower()
        m = re.search(r"file\s*name\s*:\s*(\S+\.edf)", low)
        if m:
            fname, reg0 = m.group(1), None
            continue
        m = re.search(r"registration\s+start\s+time\s*:?\s*" + TIME_RE, low)
        if m and fname:
            reg0 = _secs(*m.groups())
            continue
        m = re.search(r"seizure\s+start\s+time\s*:?\s*" + TIME_RE, low)
        if m and fname and reg0 is not None:
            t = _secs(*m.groups())
            out[fname].append([(t - reg0) % 86400, None])
            continue
        m = re.search(r"seizure\s+end\s+time\s*:?\s*" + TIME_RE, low)
        if m and fname and out.get(fname) and out[fname][-1][1] is None:
            t = _secs(*m.groups())
            start = out[fname][-1][0]
            out[fname][-1][1] = ((t - reg0) % 86400)
            if out[fname][-1][1] < start:                    # crossed midnight
                out[fname][-1][1] += 86400
    bad = [(f, iv) for f, ivs in out.items() for iv in ivs if iv[1] is None]
    if bad:
        raise ValueError(f"unpaired seizure annotations: {bad}")
    return {f: [tuple(iv) for iv in ivs] for f, ivs in out.items()}


def load_recording(path: str, cfg: dict) -> np.ndarray:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    lower = {c.lower().replace("eeg ", "").strip(): c for c in raw.ch_names}
    # legacy (T3/T4/T5/T6) and modern (T7/T8/P7/P8) temporal names both occur
    alias = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8",
             "t7": "t3", "t8": "t4", "p7": "t5", "p8": "t6"}
    picks = []
    for want in cfg["channels"]:
        w = want.lower()
        got = lower.get(w) or lower.get(alias.get(w, w))
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
    ap.add_argument("--config", default="configs/datasets/siena.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    subjects = sorted(d for d in os.listdir(root)
                      if re.fullmatch(r"PN\d+", d)
                      and os.path.isdir(os.path.join(root, d)))
    if not subjects:
        raise SystemExit(f"no PNxx subject dirs under {root}")
    tr, rest = sort_subject_split(subjects, cfg["split"]["train_ratio"])
    va, te = sort_subject_split(rest, cfg["split"]["val_ratio_of_rest"])
    groups = {"train": tr, "val": va, "test": te}
    print("subjects: %d train / %d val / %d test" % (len(tr), len(va), len(te)),
          flush=True)

    win = int(cfg["window_sec"] * cfg["sample_rate"])
    stride = int(cfg["stride_sec"] * cfg["sample_rate"])
    for split, subs in groups.items():
        X_all, y_all, used = [], [], set()
        sid_all, rec_all = [], []
        for sub in subs:
            sdir = os.path.join(root, sub)
            szlist = os.path.join(sdir, f"Seizures-list-{sub}.txt")
            try:
                intervals = parse_seizure_list(szlist)
            except Exception as ex:                          # noqa: BLE001
                man.exclude(sub, f"seizure list: {type(ex).__name__}: {ex}",
                            split=split)
                continue
            for f in sorted(os.listdir(sdir)):
                if not f.endswith(".edf"):
                    continue
                try:
                    sig = load_recording(os.path.join(sdir, f), cfg)
                    X, _ = window_signal(sig, win, stride)
                    if len(X) == 0:
                        man.exclude(f, "shorter than one window", split=split)
                        continue
                    X = norm_div100(X).astype(np.float32)
                    assert_finite(X, f)
                    y = intervals_overlap_labels(
                        len(X), win, stride,
                        intervals.get(f.lower(), []), cfg["sample_rate"])
                    X_all.append(X)
                    y_all.append(y)
                    sid_all.extend([sub] * len(X))
                    rec_all.extend([f] * len(X))
                    used.add(sub)
                except Exception as ex:                      # noqa: BLE001
                    man.exclude(f, f"{type(ex).__name__}: {ex}", split=split)
        X = np.concatenate(X_all).astype(np.float32)
        y = np.concatenate(y_all)
        save_split(out_dir, split, X, y)
        # per-window subject/recording sidecar: the b32x4 incident (val PR
        # 0.43, test PR 0.065) was unattributable without knowing which test
        # subject owns which window
        np.save(os.path.join(out_dir, f"{split}_subject_ids.npy"),
                np.array(sid_all))
        np.save(os.path.join(out_dir, f"{split}_recording_ids.npy"),
                np.array(rec_all))
        man.add_split(split, subjects=sorted(used), n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(used), shape=list(X.shape[1:]))
        print(f"[{split}] {sorted(used)} -> {len(X)} windows, "
              f"pos {int(y.sum())}", flush=True)

    man.qc = {"n_excluded": len(man.excluded),
              "note": "seizure intervals from Seizures-list-PNxx.txt against "
                      "each recording's registration start (midnight-wrapped); "
                      "half-open overlap labelling; subject-disjoint sorted split"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
