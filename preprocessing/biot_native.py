"""BIOT-native preprocessing for group B. Ported from BIOT's own scripts.

    python -m preprocessing.biot_native --dataset tuab --out <dir> [--jobs 32]

Group B's protocol rule is "必须用各自 repo 的预处理 + 归一化 + finetune recipe.
不要塞进我们的预处理" -- feeding BIOT our CBraMod-protocol data is exactly the
mistake the xlsx cites (0.6772 -> 0.4436). So this reproduces BIOT's pipeline
instead of ours, and the differences are deliberate:

                    BIOT (this file, group B)     our frozen protocol (A/C/D)
    filtering       none, resample to 200 Hz only  0.3-75 Hz band-pass + 60 Hz notch
    normalisation   per-window per-channel q95     divide by 100
                    (applied in the loader)        (applied at preprocessing)
    TUAB split      np.random.shuffle, seed 12345  subject IDs sorted, 80/20

Ported from ``datasets/TUAB/process.py`` and ``datasets/TUEV/process.py`` in
ycq091044/BIOT. The channel montage and the 10 s / 5 s windowing match ours
already, so only the three rows above actually differ.

Normalisation is deliberately NOT applied here: BIOT applies q95 inside its
Dataset (utils.py), per window at load time, so it stays in the loader
(paclock_bench/data/biot_dataset.py) to keep the split of responsibilities the
same as upstream.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

from paclock_bench.paths import DATA as _DATA
import numpy as np
import yaml

from .common import Manifest, assert_finite, resample_to, save_split, sha256_file
from .tuh_common import MissingChannels, load_bipolar_uv, subject_of
from .tuev import read_rec

DATA = _DATA


# --------------------------------------------------------------------------- #
# TUAB
# --------------------------------------------------------------------------- #
def _tuab_one(item, fs_out: int):
    path, label = item
    try:
        sig, fs = load_bipolar_uv(path)
        # BIOT: raw.resample(200) and nothing else -- no filtering at all
        sig = resample_to(sig.astype(np.float64), fs, fs_out)
        win = 10 * fs_out
        n = sig.shape[1] // win
        if n == 0:
            return path, None, None, None, "shorter than one window"
        X = np.stack([sig[:, i * win:(i + 1) * win] for i in range(n)]).astype(np.float32)
        assert_finite(X, os.path.basename(path))
        y = np.full(len(X), label, dtype=np.int64)
        return path, X, y, sha256_file(path), None
    except MissingChannels as e:
        return path, None, None, None, str(e)
    except Exception as e:                                    # noqa: BLE001
        return path, None, None, None, f"{type(e).__name__}: {e}"


def tuab(out_dir: str, jobs: int, fs_out: int = 200):
    root = f"{DATA}/tuh/tuab/edf"
    man = Manifest(dataset="tuab_biot", protocol={
        "source": "BIOT datasets/TUAB/process.py",
        "filtering": "none (resample to 200 Hz only)",
        "normalisation": "per-window q95, applied in the loader",
        "split": "np.random.shuffle with seed 12345, per class, 80/20",
        "window_sec": 10, "sample_rate": fs_out,
    })

    def collect(split_dir):
        out = []
        for name, label in (("normal", 0), ("abnormal", 1)):
            d = os.path.join(root, split_dir, name, "01_tcp_ar")
            for f in sorted(os.listdir(d)):
                if f.endswith(".edf"):
                    out.append((os.path.join(d, f), label))
        return out

    train_items, test_items = collect("train"), collect("eval")

    # BIOT's exact split: seed 12345, shuffle subject ids per class, first 80%.
    # Reproducing the RNG call order matters -- abnormal is shuffled before
    # normal in process.py, and np.random is stateful.
    np.random.seed(12345)
    assigned = {"train": [], "val": []}
    for label, cls_name in ((1, "abnormal"), (0, "normal")):
        cls = [it for it in train_items if it[1] == label]
        subs = list({subject_of(p) for p, _ in cls})
        np.random.shuffle(subs)
        cut = int(len(subs) * 0.8)
        tr, va = set(subs[:cut]), set(subs[cut:])
        assigned["train"] += [it for it in cls if subject_of(it[0]) in tr]
        assigned["val"] += [it for it in cls if subject_of(it[0]) in va]

    for split, items in (("train", assigned["train"]), ("val", assigned["val"]),
                         ("test", test_items)):
        print(f"[{split}] {len(items)} recordings", flush=True)
        Xs, ys, subs = [], [], []
        with Pool(jobs) as pool:
            for path, X, y, sha, err in pool.imap_unordered(
                partial(_tuab_one, fs_out=fs_out), items, chunksize=4
            ):
                if err:
                    man.exclude(os.path.basename(path), err, split=split)
                    continue
                Xs.append(X); ys.append(y); subs.append(subject_of(path))
                man.raw_sha256[os.path.basename(path)] = sha
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    man.qc = {"n_excluded": len(man.excluded),
              "note": "BIOT-native: no filtering; q95 normalisation happens in the loader"}
    man.check_disjoint(strict=False)
    man.save(os.path.join(out_dir, "manifest.json"))


# --------------------------------------------------------------------------- #
# TUEV
# --------------------------------------------------------------------------- #
def _tuev_one(edf_path: str, fs_out: int):
    rec = edf_path[:-4] + ".rec"
    try:
        if not os.path.exists(rec):
            return edf_path, None, None, None, "no .rec"
        events = read_rec(rec)
        if len(events) == 0:
            return edf_path, None, None, None, "empty .rec"
        sig, fs = load_bipolar_uv(edf_path)
        sig = resample_to(sig.astype(np.float64), fs, fs_out)   # no filtering
        win, pre, T = 5 * fs_out, 2 * fs_out, sig.shape[1]

        X, y = [], []
        for _c, start_s, _stop_s, label in events:
            if not (1 <= label <= 6):
                continue
            a = int(round(start_s * fs_out)) - pre
            # BIOT tiles the signal 3x and indexes the middle copy; modular
            # indexing is equivalent without the memory
            X.append(sig[:, np.arange(a, a + win) % T])
            y.append(int(label) - 1)
        if not X:
            return edf_path, None, None, None, "no usable events"
        Xa = np.stack(X).astype(np.float32)
        assert_finite(Xa, os.path.basename(edf_path))
        return edf_path, Xa, np.array(y, dtype=np.int64), sha256_file(edf_path), None
    except MissingChannels as e:
        return edf_path, None, None, None, str(e)
    except Exception as e:                                    # noqa: BLE001
        return edf_path, None, None, None, f"{type(e).__name__}: {e}"


def tuev(out_dir: str, jobs: int, fs_out: int = 200):
    root = f"{DATA}/tuh/tuev/edf"
    man = Manifest(dataset="tuev_biot", protocol={
        "source": "BIOT datasets/TUEV/process.py",
        "filtering": "none (resample to 200 Hz only)",
        "normalisation": "per-window q95, applied in the loader",
        "split": "np.random.shuffle with seed 12345 over train subjects, 80/20",
        "window_sec": 5, "sample_rate": fs_out,
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

    np.random.seed(12345)
    subs = list(train_subs)
    np.random.shuffle(subs)
    cut = int(len(subs) * 0.8)
    groups = {
        "train": [p for s in subs[:cut] for p in train_subs[s]],
        "val": [p for s in subs[cut:] for p in train_subs[s]],
        "test": [p for s in sorted(test_subs) for p in test_subs[s]],
    }

    for split, paths in groups.items():
        print(f"[{split}] {len(paths)} recordings", flush=True)
        Xs, ys, subs_seen = [], [], []
        with Pool(jobs) as pool:
            for path, X, y, sha, err in pool.imap_unordered(
                partial(_tuev_one, fs_out=fs_out), paths, chunksize=2
            ):
                if err:
                    man.exclude(os.path.basename(path), err, split=split)
                    continue
                Xs.append(X); ys.append(y)
                subs_seen.append(os.path.basename(os.path.dirname(path)))
                man.raw_sha256[os.path.basename(path)] = sha
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs_seen)), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    man.qc = {"n_excluded": len(man.excluded),
              "note": "BIOT-native: no filtering; q95 normalisation happens in the loader"}
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
# BIOT ships no TUSZ maker, so this applies BIOT's *conditioning* (no filtering,
# resample to 200 Hz, q95 deferred to the loader) to TUSZ's own annotation and
# split conventions, which come from preprocessing/tusz.py: the official
# train/dev/eval directories and the "any overlap with a seizure interval" label
# rule. Only the conditioning differs from the frozen protocol -- the montage,
# window length and labelling are shared, so the BIOT row stays comparable to
# the group-A TUSZ rows.
def _tusz_one(edf_path: str, fs_out: int):
    from .common import intervals_overlap_labels, window_signal   # noqa: PLC0415
    from .tusz import read_csv_bi                                 # noqa: PLC0415

    ann = edf_path[:-4] + ".csv_bi"
    try:
        if not os.path.exists(ann):
            return edf_path, None, None, None, "no .csv_bi annotation"
        intervals = read_csv_bi(ann)
        sig, fs = load_bipolar_uv(edf_path)
        # BIOT: resample only, no band-pass and no notch
        sig = resample_to(sig.astype(np.float64), fs, fs_out)
        win = stride = 10 * fs_out
        X, _tail = window_signal(sig, win, stride)
        if len(X) == 0:
            return edf_path, None, None, None, "shorter than one window"
        y = intervals_overlap_labels(len(X), win, stride, intervals, fs_out)
        X = X.astype(np.float32)          # no div100: BIOT normalises in the loader
        assert_finite(X, os.path.basename(edf_path))
        return edf_path, X, y.astype(np.int64), sha256_file(edf_path), None
    except MissingChannels as e:
        return edf_path, None, None, None, str(e)
    except Exception as e:                                    # noqa: BLE001
        return edf_path, None, None, None, f"{type(e).__name__}: {e}"


def tusz(out_dir: str, jobs: int, fs_out: int = 200):
    from .tusz import list_edfs                                   # noqa: PLC0415

    root = f"{DATA}/tuh/tusz/edf"
    man = Manifest(dataset="tusz_biot", protocol={
        "source": "BIOT conditioning (no filtering, resample only) on TUSZ's "
                  "own official-directory split and csv_bi annotations",
        "filtering": "none (resample to 200 Hz only)",
        "normalisation": "per-window q95, applied in the loader",
        "split": "official dirs: train / dev / eval",
        "label_rule": "overlap_gt_zero",
        "window_sec": 10, "sample_rate": fs_out,
    })

    for split, sub in (("train", "train"), ("val", "dev"), ("test", "eval")):
        paths = list_edfs(os.path.join(root, sub))
        print(f"[{split}] {len(paths)} recordings", flush=True)
        Xs, ys, subs = [], [], []
        with Pool(jobs) as pool:
            for path, X, y, sha, err in pool.imap_unordered(
                partial(_tusz_one, fs_out=fs_out), paths, chunksize=2
            ):
                if err:
                    man.exclude(os.path.basename(path), err, split=split)
                    continue
                Xs.append(X); ys.append(y); subs.append(subject_of(path))
                man.raw_sha256[os.path.basename(path)] = sha
        if not Xs:
            raise RuntimeError(f"{split}: every recording was excluded")
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=sorted(set(subs)), n_windows=len(X),
                      class_counts=Counter(y.tolist()), shape=list(X.shape[1:]))

    man.qc = {"n_excluded": len(man.excluded),
              "note": "BIOT-native: no filtering; q95 normalisation in the loader"}
    man.check_disjoint()
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
