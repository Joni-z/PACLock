"""BIOT-native Dataset. Normalisation matches BIOT's utils.py exactly.

BIOT normalises inside ``__getitem__``, per window and per channel, by the 95th
percentile of |x|:

    X = X / (np.quantile(np.abs(X), q=0.95, method="linear",
                         axis=-1, keepdims=True) + 1e-8)

That stays here rather than moving into preprocessing, so the division of
responsibilities is the same as upstream and the stored arrays remain the raw
resampled signal -- which is also what lets one preprocessed copy serve both the
pretrained and the from-scratch BIOT rows.
"""

from __future__ import annotations

import os

from ..paths import expand
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BIOTWindowDataset(Dataset):
    def __init__(self, root: str, split: str, *, flatten_sequences: bool = False):
        sig = os.path.join(root, f"{split}_signals.npy")
        lab = os.path.join(root, f"{split}_labels.npy")
        for p in (sig, lab):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} missing -- run preprocessing.biot_native for this dataset")
        self.signals = np.load(sig, mmap_mode="r")
        self.labels = np.load(lab)

        # ISRUC is stored as (n_seq, 20, C, T) sequences of sleep epochs. Models
        # that consume one epoch at a time -- which is every group-B model on the
        # cross-corpus path -- set flatten_sequences and see (n_seq*20, C, T).
        self.flatten = flatten_sequences and self.signals.ndim == 4
        if self.flatten:
            n, s = self.signals.shape[:2]
            self.signals = np.asarray(self.signals).reshape(n * s, *self.signals.shape[2:])
            self.labels = np.asarray(self.labels).reshape(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        X = np.array(self.signals[i], dtype=np.float32)
        # BIOT utils.py, verbatim
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear",
                             axis=-1, keepdims=True) + 1e-8)
        return torch.from_numpy(X), int(self.labels[i])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.signals.shape[1:])

    def class_counts(self) -> np.ndarray:
        return np.bincount(np.asarray(self.labels).ravel())


def build_biot_dataloaders(cfg: dict):
    root = expand(cfg["data_root"])
    flatten = bool(cfg.get("flatten_sequences", False))
    sets = {s: BIOTWindowDataset(root, s, flatten_sequences=flatten)
            for s in ("train", "val", "test")}
    bs = cfg.get("batch_size", 512)
    nw = cfg.get("num_workers", 16)
    common = dict(num_workers=nw, pin_memory=True,
                  persistent_workers=nw > 0, drop_last=False)
    loaders = (
        DataLoader(sets["train"], batch_size=bs, shuffle=True, **common),
        DataLoader(sets["val"], batch_size=bs, shuffle=False, **common),
        DataLoader(sets["test"], batch_size=bs, shuffle=False, **common),
    )
    info = {
        "manifest": {},
        "input_shape": sets["train"].shape,
        "class_counts": {s: d.class_counts().tolist() for s, d in sets.items()},
        "n_samples": {s: len(d) for s, d in sets.items()},
    }
    return (*loaders, info)
