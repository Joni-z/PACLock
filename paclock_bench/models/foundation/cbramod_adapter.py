"""CBraMod backbone and weights with an explicitly configured task head.

Architecture comes from the vendored repo (``vendor/cbramod/models/cbramod.py``)
and loads ``pretrained_weights.pth`` from the release. The historical shared
head width is 800; upstream task heads differ (TUEV 1000, TUAB/CHB/FACED 2000).
A shared-width adaptation is not automatically a faithful task reproduction.

    CBraMod(in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    backbone.proj_out = nn.Identity()      # the pretraining head is discarded

These adapters use our frozen arrays for internally matched comparisons.
Upstream TUAB/TUEV share the 16-bipolar montage, 200 Hz sampling and /100
scaling conventions. That alone does not establish bit-exact preprocessing
parity: filtering order, event boundaries and recording exclusions also matter.
Native protocol and task-head fidelity must be checked per dataset.

Its Dataset reshapes each window to (channels, patches, 200) and divides by 100:

    data = data.reshape(16, 10, 200)
    return data/100, label

The reshape happens in the wrapper below; the /100 already happened in our
preprocessing, so the loader must not apply it twice -- see CBraModWrapper.
"""

from __future__ import annotations

import os
import sys

from ...paths import vendored
import torch
import torch.nn as nn

VENDOR = vendored("cbramod")
CHECKPOINT = os.path.join(VENDOR, "pretrained_weights", "pretrained_weights.pth")

# models/model_for_tuab.py, verbatim
BACKBONE_ARGS = dict(in_dim=200, out_dim=200, d_model=200,
                     dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
PATCH = 200


def _import_cbramod():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from models.cbramod import CBraMod            # noqa: PLC0415
    return CBraMod


class CBraModClassifier(nn.Module):
    """CBraMod backbone + an 'all_patch_reps' head.

    The historical adapter default width is 800. Upstream TUEV uses 1000;
    reproduce that head with ``classifier_hidden_dim=1000`` and a new run name.

    The wrapper also does the (batch, channel, time) -> (batch, channel, patch,
    200) reshape that CBraMod's Dataset does, so the shared training loop can
    keep handing every model the same plain tensor.

    Our preprocessed arrays are already divided by 100 (the frozen protocol
    applies it at preprocessing time, CBraMod applies it in its loader), so the
    scaling is deliberately *not* repeated here.
    """

    def __init__(self, backbone: nn.Module, n_channels: int, n_patches: int,
                 n_classes: int, dropout: float = 0.1,
                 classifier_hidden_dim: int = 800):
        super().__init__()
        self.backbone = backbone
        self.backbone.proj_out = nn.Identity()
        self.n_patches = n_patches
        feat = n_channels * n_patches * BACKBONE_ARGS["d_model"]
        # Preserve the historical default while allowing the actual task head.
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feat, classifier_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, BACKBONE_ARGS["d_model"]),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(BACKBONE_ARGS["d_model"], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if T % PATCH:
            raise ValueError(f"CBraMod needs T divisible by {PATCH}, got {T}")
        x = x.reshape(B, C, T // PATCH, PATCH)
        return self.classifier(self.backbone(x))

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.classifier.parameters()


class CBraModSequence(nn.Module):
    """CBraMod for ISRUC, ported from ``models/model_for_isruc.py``.

    ISRUC is the one corpus where CBraMod does not use the flatten-and-classify
    head. Its input is a *sequence* of 20 consecutive 30 s sleep epochs, each
    epoch is encoded independently, and a 1-layer transformer then mixes across
    the sequence before every epoch is classified:

        x = x.view(bz*seq_len, ch, 30, 200)
        f = backbone(x).view(bz, seq_len, ch*30*200)
        f = head(f)                       # Linear(6*30*200, 512) + GELU
        f = sequence_encoder(f)           # TransformerEncoderLayer(512, nhead=4,
                                          #   ff=2048, norm_first=True), 1 layer
        out = classifier(f)               # (bz, seq_len, n_classes)

    Using the TUAB head here instead would discard the cross-epoch context that
    is the whole point of the design -- neighbouring epochs carry most of the
    information for staging transitions.

    The output keeps one prediction per epoch, so it is (B, seq_len, n_classes);
    the training loop flattens it against the (B, seq_len) labels.
    """

    def __init__(self, backbone: nn.Module, n_channels: int, n_patches: int,
                 n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.backbone.proj_out = nn.Identity()
        self.n_patches = n_patches
        feat = n_channels * n_patches * BACKBONE_ARGS["d_model"]
        self.head = nn.Sequential(nn.Linear(feat, 512), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048, batch_first=True,
            activation=nn.functional.gelu, norm_first=True)
        self.sequence_encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bz, seq_len, ch, epoch_size = x.shape
        x = x.contiguous().view(bz * seq_len, ch, epoch_size // PATCH, PATCH)
        f = self.backbone(x).contiguous().view(bz, seq_len, -1)
        f = self.head(f)
        f = self.sequence_encoder(f)
        return self.classifier(f)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        for m in (self.head, self.sequence_encoder, self.classifier):
            yield from m.parameters()


def build_cbramod(n_classes: int, n_channels: int, seq_len: int, *,
                  pretrained: bool = True, dropout: float = 0.1,
                  sequence: bool = False, pretrained_path: str | None = None,
                  classifier_hidden_dim: int = 800) -> nn.Module:
    """Build CBraMod; ``pretrained=False`` is the group-C from-scratch row.

    ``sequence=True`` selects the ISRUC variant (model_for_isruc.py), which is
    the only corpus where upstream classifies a sequence of epochs jointly.
    """
    CBraMod = _import_cbramod()
    backbone = CBraMod(**BACKBONE_ARGS)
    if pretrained_path:
        # our own re-pretrained CBraMod (training/pretrain_cbramod.py, kind=native)
        ck = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=True)
        print(f"  CBraMod: loaded our checkpoint {pretrained_path} (step {ck.get('step') if isinstance(ck, dict) else '?'})", flush=True)
    elif pretrained:
        load_pretrained(backbone)
    n_patches = seq_len // PATCH
    if sequence:
        return CBraModSequence(backbone, n_channels, n_patches, n_classes)
    return CBraModClassifier(backbone, n_channels, n_patches, n_classes, dropout,
                            classifier_hidden_dim=classifier_hidden_dim)


def count_backbone_params(model: nn.Module) -> float:
    """Backbone-only parameter count, in millions.

    The xlsx lists CBraMod at ~4M, which is the foundation model itself. The
    total depends on the corpus and configured hidden width, because the head
    flattens ``channels * patches * 200``. Historical shared-width totals range
    from about 17.9M on TUEV to 56.3M on FACED. Both
    numbers are real; they answer different questions, so both are reported.
    """
    bb = getattr(model, "backbone", None)
    if bb is None:
        return float("nan")
    return sum(p.numel() for p in bb.parameters()) / 1e6


def load_pretrained(backbone: nn.Module, path: str = CHECKPOINT) -> None:
    """Load pretrained_weights.pth strictly, as model_for_tuab.py does.

    Strict because upstream is strict: a partial load would leave part of the
    12-layer backbone random while the row still claims to be pretrained.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- fetch it with slurm/fetch_cbramod.slurm")
    state = torch.load(path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
