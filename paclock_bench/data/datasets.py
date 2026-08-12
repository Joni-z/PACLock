"""Datasets over the preprocessed npy pairs.

All normalisation already happened in `preprocessing/` -- exactly once, recorded
in the manifest, and identical for every model. Nothing here rescales the
signal. That is deliberate: the reference repo normalised inside __getitem__,
which made the effective input a property of the loader rather than of the
frozen protocol, and any model that brought its own loader silently got
different data.

The one exception is group B (official pretrained weights), which the protocol
requires to run each repo's own preprocessing and normalisation. Those models
get their own loaders under `models/foundation/`, never this one.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Read the whole split into RAM when it is at most this large; above it, fall
# back to mmap. 24 GB covers every corpus in the matrix except TUAB/TUSZ/CHB-MIT
# (38-42 GB), and the compute nodes have far more than that -- a four-process
# benchmark reported 8.6% memory use on a 128-core node.
PRELOAD_LIMIT_GB = 24.0


class WindowDataset(Dataset):
    """(signals, labels) npy pair produced by `preprocessing/`, in RAM if it fits.

    The original comment here read "mmap rather than a full load: TUAB is ~300k
    windows x 16 x 2000 float32, which is far past comfortable RAM". That is true
    of TUAB and false of the machine: TUEV's training split is 4.4 GB.

    mmap plus ``shuffle=True`` turns an epoch into one random 64 KB read per
    sample -- 68,445 of them for TUEV -- against a shared Lustre filesystem, and
    Lustre is at its worst on small random reads. A four-arm benchmark measured
    the consequence: batch 32, 128 and 256 with and without bf16 all landed
    within 2% of each other at ~31 samples/s, while GPU memory use rose 4.5% ->
    18.3% (so the larger batches really were being used) and GPU power stayed
    flat at 75-80 W on a 300 W part with the host at 3.5% CPU. Neither the device
    nor the host was busy; both were waiting on I/O.

    So the split is read into RAM when it fits, and mmap remains for the corpora
    that genuinely do not. Reading is sequential and happens once.
    """

    def __init__(self, root: str, split: str, *, flatten_sequences: bool = False,
                 preload: bool | None = None):
        sig_path = os.path.join(root, f"{split}_signals.npy")
        lab_path = os.path.join(root, f"{split}_labels.npy")
        for p in (sig_path, lab_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found -- run the preprocessing script for this dataset"
                )
        size_gb = os.path.getsize(sig_path) / 1024 ** 3
        if preload is None:
            preload = size_gb <= PRELOAD_LIMIT_GB
        if preload:
            self.signals = np.load(sig_path)
            print(f"  [{split}] {size_gb:.1f} GB read into RAM", flush=True)
        else:
            self.signals = np.load(sig_path, mmap_mode="r")
            print(f"  [{split}] {size_gb:.1f} GB left on disk (mmap), over the "
                  f"{PRELOAD_LIMIT_GB:.0f} GB preload limit", flush=True)
        self.labels = np.load(lab_path)
        self.split = split

        # ISRUC is stored as (n_seq, 20, C, T) sequences. Models that consume one
        # epoch at a time set flatten_sequences and see (n_seq*20, C, T).
        self.flatten = flatten_sequences and self.signals.ndim == 4
        if self.flatten:
            n, s = self.signals.shape[0], self.signals.shape[1]
            self.signals = self.signals.reshape(n * s, *self.signals.shape[2:])
            self.labels = self.labels.reshape(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        # np.array (a copy) rather than asarray: the mmap is read-only, and
        # torch.from_numpy on a non-writable buffer warns on every call and
        # hands back a tensor that is unsafe to write in place.
        x = torch.from_numpy(np.array(self.signals[i], dtype=np.float32))
        y = self.labels[i]
        y = torch.from_numpy(np.array(y, dtype=np.int64)) if np.ndim(y) else int(y)
        return x, y

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.signals.shape[1:])

    def class_counts(self) -> np.ndarray:
        return np.bincount(np.asarray(self.labels).ravel())


def load_manifest(root: str) -> dict:
    path = os.path.join(root, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- the protocol requires every run to reference the "
            f"manifest its data was built from"
        )
    with open(path) as f:
        return json.load(f)


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """train/val/test loaders plus the manifest the data came from."""
    root = cfg["data_root"]
    manifest = load_manifest(root)
    flatten = bool(cfg.get("flatten_sequences", False))
    # None lets each split decide from its own size; set explicitly to force
    preload = cfg.get("preload")

    sets = {
        s: WindowDataset(root, s, flatten_sequences=flatten, preload=preload)
        for s in ("train", "val", "test")
    }

    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 8)
    common = dict(num_workers=nw, pin_memory=True,
                  persistent_workers=nw > 0, drop_last=False)

    loaders = (
        DataLoader(sets["train"], batch_size=bs, shuffle=True, **common),
        DataLoader(sets["val"], batch_size=bs, shuffle=False, **common),
        DataLoader(sets["test"], batch_size=bs, shuffle=False, **common),
    )
    info = {
        "manifest": manifest,
        "input_shape": sets["train"].shape,
        "class_counts": {s: d.class_counts().tolist() for s, d in sets.items()},
        "n_samples": {s: len(d) for s, d in sets.items()},
    }
    return (*loaders, info)
