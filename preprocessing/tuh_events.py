"""TUSL / TUAR -> event-centred npy + manifest.

    python -m preprocessing.tuh_events --config configs/datasets/tusl.yaml

Why these two corpora exist in this benchmark: the PAC interaction tokenizer's
only decisive win so far is TUEV (+0.172 Cohen's kappa over the plain
filterbank tokenizer), which is transient-event *morphology* classification.
Everywhere else -- seizure detection, sleep staging, motor imagery, emotion --
the plain tokenizer matches or beats it. One dataset is not a mechanism, so
TUSL and TUAR are added as same-type tasks to test whether the TUEV advantage
generalises to event-morphology classification or was idiosyncratic to TUEV.

TUSL is the sharper test of the two: its three classes are seizure, slowing
and background, and the corpus exists precisely because slowing is a source of
*false positives* for seizure detectors -- the two are similar in band power
and differ in waveform structure, which is what a coupling-based tokenizer
should be able to exploit and a band-power one should not.

Differences from preprocessing/tuev.py, which this otherwise mirrors:

  * annotations are per-channel CSV (`channel,start_time,stop_time,label`)
    rather than TUEV's `.rec`, and one event spans many channel rows -- rows
    are collapsed to unique (start, stop, label) intervals so an event is
    counted once, not once per electrode.
  * event durations vary (TUEV's are ~1 s by construction), so the window is
    centred on the event midpoint rather than anchored 2 s before its start.
    A fixed window keeps every sample the same length; centring keeps the
    event inside it whatever its length.
  * labels are strings, mapped to indices by the config, and compound TUAR
    labels (`eyem_musc`) are dropped rather than assigned to either parent --
    keeping them would make the class definition depend on annotation style.
"""

from __future__ import annotations

import argparse
import csv
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
from paclock_bench.paths import expand


def read_events(csv_path: str, label_map: dict) -> list:
    """Unique (start_s, stop_s, label_idx) intervals from a per-channel CSV.

    Every electrode carries its own row for the same event, so the same
    interval appears up to 16-22 times; collapsing to unique intervals counts
    the event once. Labels outside `label_map` (compound TUAR classes, and the
    stray seizure types in TUAR) are dropped.
    """
    seen, out = set(), []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "channel":
                continue
            if len(row) < 4:
                continue
            try:
                start_s, stop_s = float(row[1]), float(row[2])
            except ValueError:
                continue
            label = row[3].strip()
            if label not in label_map:
                continue
            key = (round(start_s, 4), round(stop_s, 4), label)
            if key in seen:
                continue
            seen.add(key)
            out.append((start_s, stop_s, label_map[label]))
    return out


def find_csv(edf_path: str) -> str | None:
    """The annotation beside an EDF, preferring the per-channel file over the
    binary (`_bi`) one -- the latter collapses every class to seizure/background
    and would silently turn a multi-class task into a binary one."""
    base = edf_path[:-4]
    for ext in (".csv", ".tse"):
        p = base + ext
        if os.path.exists(p):
            return p
    return None


def process_one(edf_path: str, cfg: dict):
    try:
        csv_path = find_csv(edf_path)
        if csv_path is None:
            return edf_path, None, None, None, 0, "no annotation file"
        events = read_events(csv_path, cfg["label_map"])
        if not events:
            return edf_path, None, None, None, 0, "no events in mapped classes"

        sig, fs = load_bipolar_uv(edf_path)
        fs_out = cfg["sample_rate"]
        sig = preprocess_signal(
            sig, fs, fs_out=fs_out,
            band=tuple(cfg["band"]) if cfg.get("band") else None,
            hp=cfg.get("hp"), notch_freq=cfg.get("notch"),
        )
        win = int(cfg["window_sec"] * fs_out)
        T = sig.shape[1]
        if T < win:
            return edf_path, None, None, None, 0, "recording shorter than one window"

        X_list, y_list, short = [], [], 0
        for start_s, stop_s, y in events:
            if stop_s - start_s < cfg.get("min_event_sec", 0.0):
                short += 1
                continue
            mid = 0.5 * (start_s + stop_s) * fs_out
            a = int(round(mid - win / 2))
            # Same modular indexing as tuev.py: events near a recording edge
            # wrap rather than being dropped, so edge events -- which are not a
            # random subset -- do not bias the sample.
            idx = np.arange(a, a + win) % T
            X_list.append(sig[:, idx])
            y_list.append(y)

        if not X_list:
            return edf_path, None, None, None, 0, "all events shorter than min_event_sec"
        X = norm_div100(np.stack(X_list)).astype(np.float32)
        assert_finite(X, os.path.basename(edf_path))
        return edf_path, X, np.array(y_list, dtype=np.int64), sha256_file(edf_path), short, None
    except MissingChannels as e:
        return edf_path, None, None, None, 0, str(e)
    except Exception as e:                                  # noqa: BLE001
        return edf_path, None, None, None, 0, f"{type(e).__name__}: {e}"


def subject_of(edf_path: str) -> str:
    """TUSL nests <subject>/<session>/<montage>/file.edf; TUAR is flat under a
    montage directory with the subject as the filename's first field. Both
    reduce to the leading `aaaaaXXX` token of the basename."""
    return os.path.basename(edf_path).split("_")[0]


def list_edfs(root: str) -> list:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".edf"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def run_group(paths, cfg, jobs, man, tag):
    X_all, y_all, subs, n_short = [], [], [], 0
    with Pool(jobs) as pool:
        for path, X, y, sha, short, err in pool.imap_unordered(
            partial(process_one, cfg=cfg), paths, chunksize=2
        ):
            if err is not None:
                man.exclude(os.path.basename(path), err, split=tag)
                continue
            X_all.append(X); y_all.append(y); subs.append(subject_of(path))
            n_short += short
            man.raw_sha256[os.path.basename(path)] = sha
    if not X_all:
        raise RuntimeError("%s: every recording was excluded" % tag)
    return (np.concatenate(X_all), np.concatenate(y_all), subs, n_short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    paths = list_edfs(root)
    if not paths:
        raise SystemExit("no EDFs under %s" % root)
    subs = sorted({subject_of(p) for p in paths})
    # No official split ships with either corpus, so hold out whole subjects,
    # sorted by ID for determinism -- the same rule TUAB uses (tuh_common's
    # sort_subject_split), and subject-disjoint so nothing leaks across splits.
    tr, rest = sort_subject_split(subs, cfg["split"]["train_ratio"])
    va, te = sort_subject_split(rest, cfg["split"]["val_ratio_of_rest"])
    groups = {"train": set(tr), "val": set(va), "test": set(te)}
    print("subjects: %d train / %d val / %d test" % (len(tr), len(va), len(te)), flush=True)

    for split, keep in groups.items():
        sel = [p for p in paths if subject_of(p) in keep]
        print("[%s] %d recordings" % (split, len(sel)), flush=True)
        X, y, s, n_short = run_group(sel, cfg, args.jobs, man, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(s)), n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(sel), shape=list(X.shape[1:]),
                      events_dropped_too_short=n_short)

    man.qc = {"n_excluded": len(man.excluded),
              "label_map": cfg["label_map"],
              "note": "event-centred windows; per-channel CSV rows collapsed to "
                      "unique intervals so an event counts once"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
