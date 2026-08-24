"""CSBrain (NeurIPS 2025 spotlight) adapter -- group B row.

Vendored repo: ``vendor/csbrain`` (yuchen2199/CSBrain @ 185aee5); weights:
``vendor/csbrain/weights/CSBrain/pth/CSBrain.pth`` (their Google Drive
release). Mirrors their ``models/model_for_*.py`` wrappers: the CSBrain
backbone with per-corpus brain-region encoding and region-topology channel
sorting, their shape-filtered pretrained load, ``proj_out`` stripped, and
their MLP classifier head; the finetune recipe in the experiment configs is
their ``sh/finetune_CSBrain_*.sh`` (AdamW lr 1e-4, wd 0.01, dropout 0.1,
cosine).

Region maps: the TUH 16-channel bipolar corpora use their TUEV mapping
VERBATIM (same montage, same order). Referential 19-channel corpora build
the same structures from the electrode names with their region/topology
tables -- the identical sorting algorithm, so a corpus they never ran gets
the construction their own wrappers would have produced.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from ...paths import vendored

VENDOR = vendored("csbrain")
CHECKPOINT = os.path.join(VENDOR, "weights", "CSBrain", "pth", "CSBrain.pth")

# their model_for_tuev.py, verbatim (first electrode of each bipolar pair)
TUH16_REGIONS = [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 4, 1, 0, 0, 4, 1]
TUH16_ELECTRODES = ["FP1", "F7", "T3", "T5", "FP2", "F8", "T4", "T6",
                    "FP1", "F3", "C3", "P3", "FP2", "F4", "C4", "P4"]

# Frontal 0 | Parietal 1 | Temporal 2 | Occipital 3 | Central 4
REGION_OF = {"FP1": 0, "FP2": 0, "F3": 0, "F4": 0, "F7": 0, "F8": 0, "FZ": 0,
             "C3": 4, "C4": 4, "CZ": 4,
             "P3": 1, "P4": 1, "PZ": 1,
             "T3": 2, "T4": 2, "T5": 2, "T6": 2,
             "O1": 3, "O2": 3}
TOPOLOGY = {
    0: ["FP1", "F3", "F7", "FZ", "F4", "F8", "FP2"],
    4: ["C3", "CZ", "C4"],
    1: ["P3", "PZ", "P4"],
    2: ["T3", "T5", "T6", "T4"],
    3: ["O1", "O2"],
}

# canonical 19-electrode order shared by adfd / caueeg / siena / mumtaz / eegmat
MONO19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]


def _sorted_indices(regions: list[int], electrodes: list[str]) -> list[int]:
    """Their exact region-topology sort (model_for_tuev.py), stable on ties."""
    groups: dict[int, list[tuple[int, str]]] = {}
    for i, r in enumerate(regions):
        groups.setdefault(r, []).append((i, electrodes[i]))
    out = []
    for r in sorted(groups):
        out.extend(i for i, _ in
                   sorted(groups[r], key=lambda t: TOPOLOGY[r].index(t[1])))
    return out


def _maps_for(n_channels: int, dataset: str):
    if n_channels == 16:
        regions, electrodes = TUH16_REGIONS, TUH16_ELECTRODES
    elif n_channels == 19:
        electrodes = MONO19
        regions = [REGION_OF[e] for e in electrodes]
    else:
        raise ValueError(
            f"csbrain adapter has no region map for {n_channels} channels "
            f"(dataset {dataset!r}); known: 16 (TUH bipolar), 19 (mono 10-20)")
    return regions, _sorted_indices(regions, electrodes)


class CSBrainClassifier(nn.Module):
    """Backbone + their per-corpus MLP head, fed (B, C, T) raw windows."""

    def __init__(self, backbone: nn.Module, n_channels: int, n_patches: int,
                 n_classes: int, dropout: float, d: int = 200):
        super().__init__()
        self.backbone = backbone
        self.n_patches = n_patches
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * n_patches * d, 5 * d),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(5 * d, d),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        P = self.n_patches
        x = x[:, :, : P * 200].reshape(B, C, P, 200)
        feats = self.backbone(x)
        return self.classifier(feats.contiguous().view(B, -1))


def build_csbrain(n_classes: int, n_channels: int, seq_len: int, dataset: str,
                  *, pretrained: bool = True, dropout: float = 0.1,
                  n_layer: int = 12) -> nn.Module:
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from models.CSBrain import CSBrain                      # noqa: PLC0415

    regions, sorted_idx = _maps_for(n_channels, dataset)
    backbone = CSBrain(in_dim=200, out_dim=200, d_model=200,
                       dim_feedforward=800, seq_len=30, n_layer=n_layer,
                       nhead=8, brain_regions=regions,
                       sorted_indices=sorted_idx)
    if pretrained:
        sd = torch.load(CHECKPOINT, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        msd = backbone.state_dict()
        match = {k: v for k, v in sd.items()
                 if k in msd and v.size() == msd[k].size()}
        msd.update(match)
        backbone.load_state_dict(msd)
        print(f"[csbrain] loaded {len(match)}/{len(msd)} tensors from "
              f"{os.path.basename(CHECKPOINT)} (their shape-filtered load)",
              flush=True)
    backbone.proj_out = nn.Sequential()
    return CSBrainClassifier(backbone, n_channels, seq_len // 200,
                             n_classes, dropout)
