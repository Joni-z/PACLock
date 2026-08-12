"""LaBraM-native preprocessing for group B. Ported from LaBraM's own scripts.

    python -m preprocessing.labram_native --dataset tuab --out <dir> [--jobs 32]

Ported from ``dataset_maker/make_TUAB.py`` and ``make_TUEV.py`` in
935963004/LaBraM. LaBraM's pipeline differs from both ours and BIOT's on every
line that matters, which is exactly why the protocol makes group B use each
repo's own preprocessing:

                     LaBraM (here)          BIOT (biot_native)     ours (A/C/D)
    montage          23 unipolar -REF       16 bipolar             16 bipolar
    band-pass        0.1-75 Hz              none                   0.3-75 Hz
    notch            50 Hz                  none                   60 Hz
    units            microvolts             volts->uV              microvolts
    normalisation    /100 in the loader     q95 in the loader      /100 at preprocess

The 23-channel unipolar montage is the single biggest difference: LaBraM's
patch embedding and its positional embeddings are built for those electrodes in
that order, so feeding it a bipolar montage would not just change the numbers,
it would mismatch the pretrained weights.

``raw.get_data(units='uV')`` is used verbatim; the /100 that LaBraM applies at
load time stays in the loader, matching upstream's split of responsibilities.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import mne
from paclock_bench.paths import DATA as _DATA
import numpy as np

from .common import Manifest, assert_finite, save_split, sha256_file
from .tuev import read_rec

DATA = _DATA

# verbatim from make_TUAB.py
DROP_CHANNELS = [
    'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF',
    'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF',
    'EEG SP1-REF', 'EEG SP2-REF',
] + [f'EEG {i}-REF' for i in range(20, 129)]

CH_ORDER = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF',
]

L_FREQ, H_FREQ, NOTCH, FS_OUT = 0.1, 75.0, 50.0, 200


def load_labram(edf_path: str) -> np.ndarray:
    """EDF -> (23, T) in microvolts, LaBraM's channel set / filtering / rate."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    useless = [c for c in DROP_CHANNELS if c in raw.ch_names]
    if useless:
        raw.drop_channels(useless)
    if len(CH_ORDER) == len(raw.ch_names):
        raw.reorder_channels(CH_ORDER)
    if raw.ch_names != CH_ORDER:
        raise ValueError(
            f"channel order mismatch: have {len(raw.ch_names)} channels "
            f"{raw.ch_names[:4]}..., need the 23 in LaBraM's chOrder_standard")
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose="ERROR")
    raw.notch_filter(NOTCH, verbose="ERROR")
    raw.resample(FS_OUT, verbose="ERROR")
    return raw.get_data(units="uV")


def _tuab_one(item):
    path, label = item
    try:
        sig = load_labram(path)
        win = 10 * FS_OUT
        n = sig.shape[1] // win
        if n == 0:
            return path, None, None, None, "shorter than one window"
        X = np.stack([sig[:, i * win:(i + 1) * win] for i in range(n)]).astype(np.float32)
        assert_finite(X, os.path.basename(path))
        return path, X, np.full(len(X), label, dtype=np.int64), sha256_file(path), None
    except Exception as e:                                    # noqa: BLE001
        return path, None, None, None, f"{type(e).__name__}: {e}"


def _tuev_one(edf_path: str):
    rec = edf_path[:-4] + ".rec"
    try:
        if not os.path.exists(rec):
            return edf_path, None, None, None, "no .rec"
        events = read_rec(rec)
        if len(events) == 0:
            return edf_path, None, None, None, "empty .rec"
        sig = load_labram(edf_path)
        win, pre, T = 5 * FS_OUT, 2 * FS_OUT, sig.shape[1]
        X, y = [], []
        for _c, start_s, _stop_s, label in events:
            if not (1 <= label <= 6):
                continue
            a = int(round(start_s * FS_OUT)) - pre
            X.append(sig[:, np.arange(a, a + win) % T])
            y.append(int(label) - 1)
        if not X:
            return edf_path, None, None, None, "no usable events"
        Xa = np.stack(X).astype(np.float32)
        assert_finite(Xa, os.path.basename(edf_path))
        return edf_path, Xa, np.array(y, dtype=np.int64), sha256_file(edf_path), None
    except Exception as e:                                    # noqa: BLE001
        return edf_path, None, None, None, f"{type(e).__name__}: {e}"


def _run(groups, worker, jobs, man, out_dir, subject_of_path):
    for split, items in groups.items():
        print(f"[{split}] {len(items)} recordings", flush=True)
        Xs, ys, subs = [], [], []
        with Pool(jobs) as pool:
            for path, X, y, sha, err in pool.imap_unordered(worker, items, chunksize=2):
                if err:
                    man.exclude(os.path.basename(path), err, split=split)
                    continue
                Xs.append(X); ys.append(y); subs.append(subject_of_path(path))
                man.raw_sha256[os.path.basename(path)] = sha
        if not Xs:
            raise RuntimeError(f"{split}: everything was excluded")
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))


def tuab(out_dir: str, jobs: int):
    from .tuh_common import subject_of
    root = f"{DATA}/tuh/tuab/edf"
    man = Manifest(dataset="tuab_labram", protocol={
        "source": "LaBraM dataset_maker/make_TUAB.py",
        "montage": "23 unipolar -REF, chOrder_standard",
        "filtering": "0.1-75 Hz band-pass + 50 Hz notch",
        "sample_rate": FS_OUT, "units": "uV",
        "normalisation": "divide by 100, applied in the loader",
        "split": "np.random.shuffle, per class, 80/20",
    })

    def collect(d):
        out = []
        for name, label in (("normal", 0), ("abnormal", 1)):
            p = os.path.join(root, d, name, "01_tcp_ar")
            for f in sorted(os.listdir(p)):
                if f.endswith(".edf"):
                    out.append((os.path.join(p, f), label))
        return out

    train_items, test_items = collect("train"), collect("eval")
    np.random.seed(4523)          # LaBraM's dataset seed
    assigned = {"train": [], "val": []}
    for label in (1, 0):
        cls = [it for it in train_items if it[1] == label]
        subs = list({subject_of(p) for p, _ in cls})
        np.random.shuffle(subs)
        cut = int(len(subs) * 0.8)
        tr, va = set(subs[:cut]), set(subs[cut:])
        assigned["train"] += [it for it in cls if subject_of(it[0]) in tr]
        assigned["val"] += [it for it in cls if subject_of(it[0]) in va]

    _run({"train": assigned["train"], "val": assigned["val"], "test": test_items},
         _tuab_one, jobs, man, out_dir, subject_of)
    man.qc = {"n_excluded": len(man.excluded), "n_channels": len(CH_ORDER)}
    man.check_disjoint(strict=False)
    man.save(os.path.join(out_dir, "manifest.json"))


def tuev(out_dir: str, jobs: int):
    root = f"{DATA}/tuh/tuev/edf"
    man = Manifest(dataset="tuev_labram", protocol={
        "source": "LaBraM dataset_maker/make_TUEV.py",
        "montage": "23 unipolar -REF, chOrder_standard",
        "filtering": "0.1-75 Hz band-pass + 50 Hz notch",
        "sample_rate": FS_OUT, "units": "uV",
        "normalisation": "divide by 100, applied in the loader",
    })

    def subject_files(split_root):
        out = {}
        for sub in sorted(os.listdir(split_root)):
            d = os.path.join(split_root, sub)
            if os.path.isdir(d):
                e = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".edf"))
                if e:
                    out[sub] = e
        return out

    train_subs = subject_files(os.path.join(root, "train"))
    test_subs = subject_files(os.path.join(root, "eval"))
    np.random.seed(4523)
    subs = list(train_subs)
    np.random.shuffle(subs)
    cut = int(len(subs) * 0.8)
    groups = {
        "train": [p for s in subs[:cut] for p in train_subs[s]],
        "val": [p for s in subs[cut:] for p in train_subs[s]],
        "test": [p for s in sorted(test_subs) for p in test_subs[s]],
    }
    _run(groups, _tuev_one, jobs, man, out_dir,
         lambda p: os.path.basename(os.path.dirname(p)))
    man.qc = {"n_excluded": len(man.excluded), "n_channels": len(CH_ORDER)}
    man.check_disjoint()
    man.save(os.path.join(out_dir, "manifest.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tuab", "tuev", "tusz"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"tuab": tuab, "tuev": tuev, "tusz": tusz}[args.dataset](args.out, args.jobs)




# --------------------------------------------------------------------------- #
# TUSZ
# --------------------------------------------------------------------------- #
# LaBraM ships makers for TUAB and TUEV only. This applies LaBraM's own
# conditioning -- the 23 unipolar -REF montage, 0.1-75 Hz, 50 Hz notch, 200 Hz,
# microvolts, /100 deferred to the loader -- to TUSZ's official-directory split
# and csv_bi seizure annotations, which come from preprocessing/tusz.py.
# The montage is the part that matters: LaBraM's positional embedding is indexed
# by electrode identity on the TUH path, so TUSZ must supply the same 23
# channels make_TUAB.py does, not the 16 bipolar ones the frozen protocol uses.
def _tusz_one(edf_path: str):
    from .common import intervals_overlap_labels, window_signal   # noqa: PLC0415
    from .tusz import read_csv_bi                                 # noqa: PLC0415

    ann = edf_path[:-4] + ".csv_bi"
    try:
        if not os.path.exists(ann):
            return edf_path, None, None, None, "no .csv_bi annotation"
        intervals = read_csv_bi(ann)
        sig = load_labram(edf_path)               # 23 unipolar, 0.1-75, 50 Hz, uV
        win = stride = 10 * FS_OUT
        X, _tail = window_signal(sig, win, stride)
        if len(X) == 0:
            return edf_path, None, None, None, "shorter than one window"
        y = intervals_overlap_labels(len(X), win, stride, intervals, FS_OUT)
        X = X.astype(np.float32)          # no /100: LaBraM scales in the loader
        assert_finite(X, os.path.basename(edf_path))
        return edf_path, X, y.astype(np.int64), sha256_file(edf_path), None
    except Exception as e:                                    # noqa: BLE001
        return edf_path, None, None, None, f"{type(e).__name__}: {e}"


def tusz(out_dir: str, jobs: int):
    from .tuh_common import subject_of                           # noqa: PLC0415
    from .tusz import list_edfs                                  # noqa: PLC0415

    root = f"{DATA}/tuh/tusz/edf"
    man = Manifest(dataset="tusz_labram", protocol={
        "source": "LaBraM conditioning (make_TUAB.py) on TUSZ's own "
                  "official-directory split and csv_bi annotations",
        "montage": "23 unipolar -REF, chOrder_standard",
        "filtering": "0.1-75 Hz band-pass + 50 Hz notch",
        "sample_rate": FS_OUT, "units": "uV",
        "normalisation": "divide by 100, applied in the loader",
        "split": "official dirs: train / dev / eval",
        "label_rule": "overlap_gt_zero",
        "window_sec": 10,
    })

    groups = {split: list_edfs(os.path.join(root, sub))
              for split, sub in (("train", "train"), ("val", "dev"), ("test", "eval"))}
    _run(groups, _tusz_one, jobs, man, out_dir, subject_of)
    man.qc = {"n_excluded": len(man.excluded), "n_channels": len(CH_ORDER)}
    man.check_disjoint()
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
