"""TUEP v2.0.1 -> windowed npy + manifest.

    sbatch slurm/preprocess.slurm  (or)  python -m preprocessing.tuep --config configs/datasets/tuep.yaml

Why this corpus joins the suite: the 12-set downstream slate needs corpora in
the regimes this architecture is measured to be strong in, and TUEP is the
closest sibling of two of them -- a recording-level clinical binary
(epilepsy=1 / no_epilepsy=0), TUAB's task shape on TUSZ's clinical question.
The label comes from the top-level directory, exactly like TUAB; unlike TUAB
there is NO official train/eval split, so whole subjects are held out by
sorted ID, the same rule every unofficial-split corpus in this benchmark uses
(docs/PROTOCOLS.md).

Differences from preprocessing/tuab.py, which this otherwise mirrors:
  * recordings live under <label_dir>/<subject>/<session>/<montage>/*.edf
    rather than <split>/<label>/<montage>/*.edf -- collected by os.walk;
  * one subject can carry many sessions; all of a subject's recordings follow
    the subject into its split (subject-disjoint by construction);
  * only 01_tcp_ar-montage recordings are used, matching channel_std
    everywhere else in the suite.
"""

from __future__ import annotations

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
from .tuab import _filter_args
from .tuh_common import MissingChannels, load_bipolar_uv, sort_subject_split, subject_of
from paclock_bench.paths import expand


def process_one(path_label: tuple[str, int], cfg: dict):
    path, label = path_label
    try:
        sig, fs = load_bipolar_uv(path)
        sig = preprocess_signal(
            sig, fs, fs_out=cfg["sample_rate"], **_filter_args(cfg),
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


def collect(root: str, cfg: dict) -> list[tuple[str, int]]:
    """(edf_path, label) for the whole corpus. epilepsy=1, no_epilepsy=0.

    Only recordings under the standard montage directory are taken, so the
    16-bipolar channel set is constructible for every file that enters.
    """
    out = []
    for name, label in (("01_no_epilepsy", 0), ("00_epilepsy", 1)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            raise FileNotFoundError(d)
        for dirpath, _, files in os.walk(d):
            if os.path.basename(dirpath) != cfg["channel_std"]:
                continue
            for f in sorted(files):
                if f.endswith(".edf"):
                    out.append((os.path.join(dirpath, f), label))
    return sorted(out)


def run_group(items, cfg, jobs, manifest, tag):
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
    return (np.concatenate(X_all).astype(np.float32),
            np.concatenate(y_all).astype(np.int64), subs, tail_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/tuep.yaml")
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = expand(cfg["raw_root"]), expand(cfg["out_dir"])
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    items = collect(root, cfg)
    if not items:
        raise SystemExit(f"no EDFs under {root}")
    # Subject-disjoint sorted split, stratified BY LABEL: epilepsy and
    # no_epilepsy subjects are split 70/15/15 separately so both classes appear
    # in every split whatever the class ratio -- a subject never contributes a
    # label conflict because the label is a property of the subject's top dir.
    by_label = {0: set(), 1: set()}
    for p, lab in items:
        by_label[lab].add(subject_of(p))
    groups = {"train": set(), "val": set(), "test": set()}
    for lab, subs in by_label.items():
        tr, rest = sort_subject_split(sorted(subs), cfg["split"]["train_ratio"])
        va, te = sort_subject_split(rest, cfg["split"]["val_ratio_of_rest"])
        groups["train"] |= set(tr)
        groups["val"] |= set(va)
        groups["test"] |= set(te)
    print("subjects: %d train / %d val / %d test"
          % (len(groups["train"]), len(groups["val"]), len(groups["test"])),
          flush=True)

    for split, keep in groups.items():
        sel = [(p, l) for p, l in items if subject_of(p) in keep]
        print(f"[{split}] {len(sel)} recordings", flush=True)
        X, y, s, tail = run_group(sel, cfg, args.jobs, man, split)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(s)), n_windows=len(X),
                      class_counts=Counter(y.tolist()),
                      n_recordings=len(sel), shape=list(X.shape[1:]),
                      seconds_dropped_at_tails=tail)

    man.qc = {"n_excluded": len(man.excluded),
              "note": "recording-level label from the top directory; "
                      "subject-disjoint sorted split stratified by label"}
    man.check_disjoint(strict=True)
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
