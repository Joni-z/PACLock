"""TUEG -> a bounded, subject-diverse slice, windowed for self-supervised
pretraining. Protocol: docs/PRETRAIN_PLAN.md sec 2/4 -- pretraining pool
`patch_len` decision, and the deliberate choice NOT to chase TUEG's full
27,074h against a model 5-30x smaller than the baselines that use it whole.

    python -m preprocessing.tueg --config configs/datasets/tueg_slice.yaml

Unlike tuab.py/tuev.py/tusz.py, TUEG carries no classification label -- this
produces unlabeled windows for paclock_bench.training.pretrain, which
discards labels anyway. `y` is written as all-zeros purely so the output
shape matches the same (signals.npy, labels.npy) pair every other corpus
uses, letting WindowDataset read it with zero special-casing.

Two things this script does that tuab.py's pattern does not need:

1. Session-list exclusion. TUEG's own DOCS/sessions_tueg_common_with_tusz.list
   enumerates every session TUEG shares with TUSZ -- our own downstream eval
   corpus. Pretraining on those sessions and then evaluating a from-scratch
   vs pretrained TUSZ comparison would be exactly the TUAB/TUEV-in-TUEG
   leakage CBraMod's own paper flags about itself (docs/ARCH_SEARCH.md).
   Excluded before sampling, not after -- so the target hour budget is spent
   entirely on eval-clean data, not partially wasted on sessions cut later.

2. Subject-level sampling to a target hour budget, not "take everything".
   TUEG is 27,074h across 24,618 unique subjects; the pretraining pool this
   feeds is ~3-8k hours, matching the rest of the pool's order of magnitude
   rather than jumping to CBraMod's data scale for a model 5-30x smaller
   (docs/PRETRAIN_PLAN.md sec "final plan"). One file is drawn from each of
   as many DISTINCT subjects as the budget allows (deterministic shuffle,
   seeded) before ever taking a second file from the same subject -- so the
   slice buys subject diversity first, not just hours from a few subjects.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import yaml

from .common import (
    Manifest,
    assert_finite,
    norm_div100,
    preprocess_signal,
    sha256_file,
    window_signal,
)
from .tuh_common import MissingChannels, load_bipolar_uv
from paclock_bench.paths import expand


def _session_of(edf_path: str) -> tuple[str, str]:
    """.../edf/<grp>/<subject>/<session>_<year>/<montage>/<file>.edf
    -> (subject, "session_year"), matching the session-list files' own
    "<subject>/<session>_<year>" format."""
    parts = edf_path.rstrip("/").split("/")
    # parts[-1]=file, [-2]=montage, [-3]="s001_2003", [-4]=subject
    return parts[-4], parts[-3]


def load_session_list(path: str) -> set[str]:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_downstream_subjects(manifest_paths: list[str]) -> set[str]:
    """Every subject of every split of every downstream TUH corpus.

    TUAB / TUEV / TUSZ / TUEP / TUAR are all SUBSETS of TUEG, so a slice
    sampled from TUEG re-serves downstream subjects to pretraining unless
    they are removed by subject. The corpus's own
    DOCS/sessions_tueg_common_with_tusz.list removes SESSIONS, which is not
    enough: the same patient's other sessions stay in, and an EEG model can
    learn patient-specific signatures across sessions. Measured before this
    was added: 34.8% of TUAB's test subjects, 39.5% of TUSZ's, 39.3% of
    TUEP's and 28.1% of TUAR's were inside the slice, which is 55% of the
    pretraining sample budget.

    All splits are excluded, not just test/val. Pretraining on a downstream
    TRAIN split is legitimate and the pool does it directly for nine corpora,
    but doing it a second time THROUGH TUEG would silently re-weight those
    subjects; excluding wholesale keeps the pool's composition the thing the
    config says it is. TUEG has ~14.7k eligible subjects against the slice's
    ~5.2k, so removing ~3.2k downstream subjects costs no hours at all.
    """
    subs: set[str] = set()
    for mp in manifest_paths:
        with open(mp) as f:
            man = json.load(f)
        for split in man.get("splits", {}).values():
            subs.update(split.get("subjects", []))
    return subs


def select_slice(index_path: str, exclude_list_path: str, target_hours: float,
                 avg_file_hours: float, seed: int,
                 exclude_subjects: set[str] | None = None) -> list[str]:
    """Subject-diverse file selection under a target-hour budget.

    avg_file_hours comes from the corpus's own AAREADME (0.3813h/file, TUEG
    v2.0.2) rather than measured here -- measuring would mean opening every
    candidate file's header before knowing whether it survives selection,
    which is the expensive step this function exists to defer to the actual
    preprocessing worker, run only on the files actually chosen.
    """
    with open(index_path) as f:
        all_files = [l.strip() for l in f if l.strip()]

    excluded_sessions = load_session_list(exclude_list_path)
    excluded_subjects = exclude_subjects or set()
    by_subject: dict[str, list[str]] = defaultdict(list)
    n_excluded = 0
    n_excluded_subj = 0
    for path in all_files:
        subject, session = _session_of(path)
        if subject in excluded_subjects:
            n_excluded_subj += 1
            continue
        if f"{subject}/{session}" in excluded_sessions:
            n_excluded += 1
            continue
        by_subject[subject].append(path)

    subjects = sorted(by_subject)                  # sort first: deterministic
    random.Random(seed).shuffle(subjects)           # then shuffle: unbiased

    target_files = int(target_hours / avg_file_hours)
    selected: list[str] = []
    # round 1: one file per subject, maximizing distinct subjects first
    for subj in subjects:
        if len(selected) >= target_files:
            break
        selected.append(sorted(by_subject[subj])[0])
    # round 2 (only if the budget exceeds the subject count): second files,
    # same deterministic order, before ever taking a third from anyone
    round_idx = 1
    while len(selected) < target_files and round_idx < 20:
        added = False
        for subj in subjects:
            if len(selected) >= target_files:
                break
            files = sorted(by_subject[subj])
            if len(files) > round_idx:
                selected.append(files[round_idx])
                added = True
        if not added:
            break
        round_idx += 1

    print(f"[tueg select] {len(all_files)} files total, {n_excluded_subj} "
          f"excluded (downstream subject), {n_excluded} excluded "
          f"(TUSZ session list), {len(by_subject)} eligible subjects, "
          f"{len(selected)} files selected (~{len(selected) * avg_file_hours:.0f}h, "
          f"target {target_hours:.0f}h)", flush=True)
    return selected


def process_one(path: str, cfg: dict, tmp_dir: str):
    """Worker: one EDF -> a temp .npy holding its windows, not the array
    itself. No label -- see module docstring; y is filled with zeros by the
    caller once N is known.

    At the target scale here (2000h -> ~720k windows -> ~92GB float32,
    computed directly: 720000*16*2000*4 bytes), returning arrays through the
    multiprocessing pipe and accumulating them in the parent -- the pattern
    every other preprocessing/*.py script uses, fine at TUAB's ~49GB scale --
    OOM-killed the first run of this script (confirmed: `oom-kill event(s)`
    in the job log, 20 workers each holding a filtered array concurrently on
    top of the accumulating parent-side list). Writing each file's result to
    its own temp file here, then streaming those into a preallocated on-disk
    memmap in `main()`, bounds peak RSS to a small, roughly constant number
    of in-flight arrays regardless of corpus size.
    """
    try:
        sig, fs = load_bipolar_uv(path)
        sig = preprocess_signal(
            sig, fs, fs_out=cfg["sample_rate"],
            band=tuple(cfg["band"]) if cfg.get("band") else None,
            notch_freq=cfg.get("notch"),
        )
        win = int(cfg["window_sec"] * cfg["sample_rate"])
        stride = int(cfg["stride_sec"] * cfg["sample_rate"])
        X, tail = window_signal(sig, win, stride)
        if len(X) == 0:
            return path, None, 0, 0, None, "shorter than one window"
        X = norm_div100(X).astype(np.float32)
        assert_finite(X, os.path.basename(path))
        sha = sha256_file(path)
        tmp_path = os.path.join(tmp_dir, sha + ".npy")
        np.save(tmp_path, X)
        return path, tmp_path, len(X), tail, sha, None
    except MissingChannels as e:
        return path, None, 0, 0, None, str(e)
    except Exception as e:                                   # noqa: BLE001
        return path, None, 0, 0, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    # preprocessing/*.py has never called paths.expand() on its own config
    # fields (unlike training/pretrain.py, data/datasets.py, models/build.py)
    # -- a gap left by the portability migration that templated raw_root/
    # out_dir to $PACLOCK_* without updating the readers. Harmless so far
    # because every existing corpus was preprocessed before that templating
    # landed; this script is the first preprocessing run since, so it can't
    # silently inherit the same bug.
    for key in ("index_path", "exclude_list_path", "out_dir"):
        cfg[key] = expand(cfg[key])

    down = [expand(p) for p in cfg.get("exclude_subject_manifests", [])]
    missing = [p for p in down if not os.path.exists(p)]
    if missing:
        # Fail rather than silently under-exclude: a missing manifest here is
        # exactly the leak this option exists to close.
        raise SystemExit("exclude_subject_manifests not found: %s" % missing)
    exclude_subjects = load_downstream_subjects(down)
    if down:
        print("[tueg select] excluding %d downstream subjects from %d corpora"
              % (len(exclude_subjects), len(down)), flush=True)

    selected = select_slice(
        cfg["index_path"], cfg["exclude_list_path"], cfg["target_hours"],
        cfg.get("avg_file_hours", 0.3813), cfg.get("seed", 0),
        exclude_subjects=exclude_subjects,
    )

    win = int(cfg["window_sec"] * cfg["sample_rate"])
    n_ch = 16
    tmp_dir = os.path.join(cfg["out_dir"], "_tmp_windows")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[tueg] starting pool of {args.jobs} workers over {len(selected)} "
          f"files", flush=True)
    man = Manifest(dataset="tueg_slice", protocol=cfg)
    chunks: list[tuple[str, int]] = []       # (tmp_path, n_windows), in order
    subs, tail_total = [], 0
    import time
    t0 = time.time()
    from functools import partial
    from multiprocessing import Pool
    with Pool(args.jobs) as pool:
        for i, (path, tmp_path, n, tail, sha, err) in enumerate(pool.imap_unordered(
            partial(process_one, cfg=cfg, tmp_dir=tmp_dir), selected, chunksize=4
        ), start=1):
            if i % 50 == 0 or i == len(selected):
                print(f"[tueg] {i}/{len(selected)} files done, "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)
            if err is not None:
                man.exclude(os.path.basename(path), err, split="train")
                continue
            chunks.append((tmp_path, n))
            subs.append(_session_of(path)[0])
            tail_total += tail
            man.raw_sha256[os.path.basename(path)] = sha

    if not chunks:
        raise RuntimeError("every selected recording was excluded")

    # Stream each worker's temp file into a preallocated on-disk memmap
    # instead of np.concatenate-ing everything in RAM (see process_one's
    # docstring for why: ~92GB at this corpus's target size). Peak RSS here
    # is one chunk's array plus the memmap's own page cache, not the whole
    # output -- bounded regardless of how large target_hours is set.
    total_n = sum(n for _, n in chunks)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    X_path = os.path.join(cfg["out_dir"], "train_signals.npy")
    X_mmap = np.lib.format.open_memmap(
        X_path, mode="w+", dtype=np.float32, shape=(total_n, n_ch, win))
    offset = 0
    for tmp_path, n in chunks:
        X_mmap[offset:offset + n] = np.load(tmp_path)
        offset += n
        os.remove(tmp_path)
    X_mmap.flush()
    del X_mmap
    os.rmdir(tmp_dir)

    y = np.zeros(total_n, dtype=np.int64)      # no label -- see docstring
    np.save(os.path.join(cfg["out_dir"], "train_labels.npy"), y)
    print(f"[train] ({total_n}, {n_ch}, {win}) -> {cfg['out_dir']}  "
          f"labels={np.bincount(y).tolist()}")

    man.add_split("train", subjects=sorted(set(subs)), n_windows=total_n,
                  class_counts=Counter(y.tolist()), discarded_tail=tail_total,
                  n_recordings=len(selected) - len(man.excluded),
                  shape=[n_ch, win])
    man.qc = {
        "n_excluded": len(man.excluded),
        "n_selected": len(selected),
        "target_hours": cfg["target_hours"],
        "actual_hours_est": total_n * cfg["window_sec"] / 3600.0,
        "purpose": "self-supervised pretraining pool only, no classification "
                   "label -- see module docstring",
    }
    man.save(os.path.join(cfg["out_dir"], "manifest.json"))


if __name__ == "__main__":
    main()
