"""LaBraM-native Dataset. Normalisation matches LaBraM's own loaders.

LaBraM stores microvolts and divides by 100 at load time (its TUABLoader /
TUEVLoader in utils.py), so the division stays here rather than in
preprocessing -- same split of responsibilities as upstream, and it keeps the
stored arrays usable by both the pretrained and the from-scratch rows.
"""

from __future__ import annotations

import os

from ..paths import expand
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LaBraMWindowDataset(Dataset):
    def __init__(self, root: str, split: str, *, flatten_sequences: bool = False,
                 divisor: float = 100.0):
        sig = os.path.join(root, f"{split}_signals.npy")
        lab = os.path.join(root, f"{split}_labels.npy")
        for p in (sig, lab):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} missing -- run preprocessing.labram_native first")
        self.divisor = divisor
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
        X = np.array(self.signals[i], dtype=np.float32) / self.divisor
        return torch.from_numpy(X), int(self.labels[i])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.signals.shape[1:])

    def class_counts(self) -> np.ndarray:
        return np.bincount(np.asarray(self.labels).ravel())


def build_labram_dataloaders(cfg: dict):
    root = expand(cfg["data_root"])
    flatten = bool(cfg.get("flatten_sequences", False))
    # LaBraM's loader divides raw microvolts by 100. That is correct for the
    # TUH rows, whose arrays labram_native.py stores in microvolts. The
    # cross-corpus rows instead read the group-A arrays, which the frozen
    # protocol has *already* normalised -- div100 for most corpora, so they
    # sit at exactly the scale LaBraM expects after its own division.
    # Dividing again put them at std ~1e-3 and the model collapsed to a
    # constant class (balanced_acc exactly 1/K, zero variance over seeds).
    divisor = float(cfg.get("loader_divisor", 100.0))
    sets = {s: LaBraMWindowDataset(root, s, flatten_sequences=flatten,
                                   divisor=divisor)
            for s in ("train", "val", "test")}
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 16)
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
