"""TFM-Tokenizer-native preprocessing for group B.

    python -m preprocessing.tfm_native --dataset tuab --out <dir> [--jobs 32]

Ported from ``datasets_processing/TUAB`` and ``datasets_processing/TUEV`` in
Jathurshan0330/TFM-Tokenizer. TFM matters more than the other group-B repos
because the xlsx's group-A published anchors come from its paper -- so its
pipeline is the one those calibration numbers were produced with.

It is a fourth distinct pipeline:

                  TFM (here)       LaBraM          BIOT        ours (A/C/D)
    band-pass     0.1-75 Hz        0.1-75 Hz       none        0.3-75 Hz
    notch         50 Hz            50 Hz           none        60 Hz
    sample rate   200 Hz           200 Hz          200 Hz      200 Hz
    montage       16 bipolar       23 unipolar     16 bipolar  16 bipolar
    normalise     q95 in loader    /100 in loader  q95 loader  /100 at preprocess

Its filtering matches LaBraM's rather than ours (0.1 vs 0.3 Hz low cut, 50 vs
60 Hz notch), so it still needs its own preprocessed copy.

Normalisation stays in the loader, matching upstream:
    X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method='linear', keepdims=True)+1e-8)
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
import yaml

from .common import (Manifest, assert_finite, bandpass, notch, resample_to,
                     save_split, sha256_file)
from .tuh_common import MissingChannels, load_bipolar_uv, subject_of
from .tuev import read_rec

DATA = "/work1/chenyuyou/yifanwang/data"


# --------------------------------------------------------------------------- #
# TUAB
# --------------------------------------------------------------------------- #
# 200 Hz, not the 256 in configs/dataset_configs.yaml.
#
# That config field is the *source* rate of the corpora; the model input rate is
# --resampling_rate, whose argparse default in downstream_transformer_finetuning.py
# is 200, and get_stft_torch builds its window and n_fft from that same value
# (n_fft=resampling_rate, hop=resampling_rate//2). The README's inference snippet
# also says "x shape (B, C, T) at 200 Hz". An earlier revision of this file used
# 256 and would have produced windows the pretrained tokenizer cannot consume.
L_FREQ, H_FREQ, NOTCH, FS_OUT = 0.1, 75.0, 50.0, 200


def _tuab_one(item, fs_out: int):
    path, label = item
    try:
        sig, fs = load_bipolar_uv(path)
        # TFM: filter at the native rate, then resample to 256 Hz
        sig = bandpass(sig.astype(np.float64), fs, L_FREQ, H_FREQ)
        sig = notch(sig, fs, NOTCH)
        sig = resample_to(sig, fs, fs_out)
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


def tuab(out_dir: str, jobs: int, fs_out: int = FS_OUT):
    root = f"{DATA}/tuh/tuab/edf"
    man = Manifest(dataset="tuab_tfm", protocol={
        "source": "TFM-Tokenizer datasets_processing/TUAB",
        "filtering": "0.1-75 Hz band-pass + 50 Hz notch, 200 Hz",
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
              "note": "TFM-native: 0.1-75 Hz + 50 Hz notch at 256 Hz; q95 in the loader"}
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
        sig = bandpass(sig.astype(np.float64), fs, L_FREQ, H_FREQ)
        sig = notch(sig, fs, NOTCH)
        sig = resample_to(sig, fs, fs_out)
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


def tuev(out_dir: str, jobs: int, fs_out: int = FS_OUT):
    root = f"{DATA}/tuh/tuev/edf"
    man = Manifest(dataset="tuev_tfm", protocol={
        "source": "TFM-Tokenizer datasets_processing/TUEV",
        "filtering": "0.1-75 Hz band-pass + 50 Hz notch, 200 Hz",
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
              "note": "TFM-native: 0.1-75 Hz + 50 Hz notch at 256 Hz; q95 in the loader"}
    man.check_disjoint()
    man.save(os.path.join(out_dir, "manifest.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tuab", "tuev"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"tuab": tuab, "tuev": tuev}[args.dataset](args.out, args.jobs)


if __name__ == "__main__":
    main()
