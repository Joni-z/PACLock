"""LaBraM for group B: official code, official checkpoint, official recipe.

Nothing is reimplemented -- the architecture comes from the vendored repo
(``vendor/labram/modeling_finetune.py``), registered with timm, and this module
only builds it the way LaBraM's own ``run_class_finetuning.py`` does and loads
``labram-base.pth``.

Model construction, from ``get_models()``:

    create_model("labram_base_patch200_200", pretrained=False,
                 num_classes=..., drop_rate=0.0, drop_path_rate=0.1,
                 attn_drop_rate=0.0, use_mean_pooling=True, init_scale=0.001,
                 use_rel_pos_bias=False, use_abs_pos_emb=True,
                 init_values=0.1, qkv_bias=False)

The README's TUAB command passes ``--disable_rel_pos_bias --abs_pos_emb
--disable_qkv_bias``, which is why rel_pos_bias and qkv_bias are off here.

Checkpoint handling copies ``run_class_finetuning.py`` exactly:

* the state dict lives under ``model`` or ``module`` (``--model_key model|module``)
* keys are prefixed ``student.`` and that prefix is stripped
  (``--model_filter_name gzp`` triggers the filter branch upstream)
* the classification head is not in the checkpoint and stays random

LaBraM also needs the channel names, uppercased and stripped to bare electrode
labels (``'EEG FP1-REF' -> 'FP1'``), because its positional embedding is indexed
by electrode identity rather than by channel position. Feeding it a different
montage than it was pretrained on would silently mis-align those embeddings, so
the montage here must match ``preprocessing/labram_native.py``.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

from ...paths import vendored
import torch
import torch.nn as nn

VENDOR = vendored("labram")
CHECKPOINT = os.path.join(VENDOR, "checkpoints", "labram-base.pth")

# the 23 channels labram_native.py produces, in that order
CH_ORDER_RAW = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF',
]

# LaBraM's own transform: 'EEG FP1-REF' -> 'FP1'
CH_NAMES = [n.split(" ")[-1].split("-")[0].upper() for n in CH_ORDER_RAW]

# run_class_finetuning.py get_models() + the README's TUAB flags
MODEL_ARGS = dict(
    pretrained=False,
    drop_rate=0.0,
    drop_path_rate=0.1,          # --drop_path 0.1
    attn_drop_rate=0.0,
    drop_block_rate=None,
    use_mean_pooling=True,
    init_scale=0.001,
    use_rel_pos_bias=False,      # --disable_rel_pos_bias
    use_abs_pos_emb=True,        # --abs_pos_emb
    init_values=0.1,
    qkv_bias=False,              # --disable_qkv_bias
)


def _load_standard_1020() -> list[str]:
    """Read the ``standard_1020`` list straight out of LaBraM's utils.py.

    utils.py cannot simply be imported: it pulls in pyhealth at module level for
    metrics we do not use. Parsing the file with ast gets the authoritative list
    without executing any of it, and without hand-copying 100+ electrode names
    that would then be able to drift from upstream.
    """
    import ast                                        # noqa: PLC0415

    path = os.path.join(VENDOR, "utils.py")
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "standard_1020" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"standard_1020 not found in {path}")


def get_input_chans(ch_names: list[str]) -> list[int]:
    """Map electrode names to LaBraM's positional-embedding indices.

    Inlined from LaBraM's utils.py:

        input_chans = [0]                       # for cls token
        for ch_name in ch_names:
            input_chans.append(standard_1020.index(ch_name) + 1)

    Copied rather than imported because utils.py imports pyhealth at module
    level for its metrics, which we do not use -- our metrics module implements
    the protocol's own definitions. ``standard_1020`` is still read out of that
    same file, by parsing it, so the index mapping remains upstream's rather
    than a transcription that could drift.
    """
    standard_1020 = _load_standard_1020()

    chans = [0]
    for name in ch_names:
        if name not in standard_1020:
            raise KeyError(f"{name!r} is not in LaBraM's standard_1020 montage")
        chans.append(standard_1020.index(name) + 1)
    return chans


def _import_labram():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import modeling_finetune          # noqa: F401  (registers the timm models)
    from timm.models import create_model     # noqa: PLC0415
    return create_model


class LaBraMWrapper(nn.Module):
    """Adapts LaBraM's (batch, channel, patch, patch_size) + ch_names interface.

    Our loaders hand every model a plain (batch, channel, time) tensor. LaBraM
    instead wants the time axis pre-split into 200-sample patches and the
    channel identities passed alongside, so the reshape and the ``input_chans``
    lookup happen here rather than leaking into the shared training loop.
    """

    def __init__(self, model: nn.Module, input_chans: torch.Tensor,
                 patch_size: int = 200, target_len: int | None = None):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        # set only on the cross-corpus path, where EEGPT's scripts resample and
        # common-average-reference before the reshape
        self.target_len = target_len
        self.register_buffer("input_chans", input_chans, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_len is not None:
            from .channel_adapt import temporal_interpolation   # noqa: PLC0415
            x = temporal_interpolation(x, self.target_len)
        B, C, T = x.shape
        if T % self.patch_size:
            raise ValueError(
                f"LaBraM needs the time axis divisible by {self.patch_size}, got {T}")
        x = x.reshape(B, C, T // self.patch_size, self.patch_size)
        return self.model(x, input_chans=self.input_chans)


# EEGPT's cross-corpus scripts (finetune_LaBraM_SleepEDF.py,
# linear_probe_LaBraM_BCIC2A.py) differ from LaBraM's own TUAB command in one
# argument: they leave the relative position bias ON. Since those scripts are
# the published provenance of the non-TUH LaBraM baselines, the cross-corpus
# path follows them rather than the TUAB README.
POSITIONAL_MODEL_ARGS = dict(MODEL_ARGS, use_rel_pos_bias=True)


def build_labram(n_classes: int, n_channels: int, *,
                 pretrained: bool = True,
                 montage_mode: str = "electrode",
                 target_len: int | None = None,
                 model_name: str = "labram_base_patch200_200") -> nn.Module:
    """Build LaBraM; ``pretrained=False`` is the group-C from-scratch row.

    ``montage_mode`` selects how the positional embedding is indexed:

    ``"electrode"``
        The 23-channel unipolar montage from ``preprocessing/labram_native.py``,
        indexed through ``standard_1020`` by electrode identity. This is
        LaBraM's own dataset maker and is used for the TUH corpora.

    ``"positional"``
        ``input_chans = range(C + 1)`` -- indexed by position, with no electrode
        lookup at all. This is what EEGPT's cross-corpus scripts do, and it is
        the provenance of the published LaBraM numbers on corpora LaBraM ships
        no maker for (Sleep-EDF, ISRUC, CHB-MIT, the BCI sets, FACED). It also
        means a bipolar montage needs no unipolar reconstruction, because
        upstream never aligns electrodes in this path. ``target_len`` (window
        length in samples at 200 Hz) is required here, since the cross-corpus
        path also applies upstream's ``temporal_interpolation``.
    """
    create_model = _import_labram()

    if montage_mode == "electrode":
        if n_channels != len(CH_NAMES):
            raise ValueError(
                f"LaBraM's electrode-indexed path expects the {len(CH_NAMES)}-channel "
                f"montage from preprocessing/labram_native.py, got {n_channels} "
                f"channels; use montage_mode='positional' for other corpora")
        model = create_model(model_name, num_classes=n_classes, **MODEL_ARGS)
        if pretrained:
            load_pretrained(model)
        return LaBraMWrapper(model, torch.IntTensor(get_input_chans(CH_NAMES)))

    if montage_mode != "positional":
        raise ValueError(f"unknown montage_mode {montage_mode!r}")

    if target_len is None:
        raise ValueError("montage_mode='positional' needs target_len "
                         "(window length in samples at 200 Hz)")
    model = create_model(model_name, num_classes=n_classes, **POSITIONAL_MODEL_ARGS)
    if pretrained:
        load_pretrained(model)
    # input_chans=[i for i in range(C+1)], verbatim from EEGPT's scripts
    return LaBraMWrapper(model, torch.IntTensor(list(range(n_channels + 1))),
                         target_len=target_len)


def load_pretrained(model: nn.Module, path: str = CHECKPOINT) -> None:
    """Load labram-base.pth, following run_class_finetuning.py's key handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # weights_only=False: torch >= 2.6 defaults it to True, and this checkpoint
    # carries numpy scalars from LaBraM's training metadata, which the restricted
    # unpickler rejects. Upstream predates that default. The file is the one
    # published in the LaBraM repo, so the trust requirement is satisfied; the
    # narrower alternative (add_safe_globals) would still have to allowlist
    # arbitrary numpy internals, with no real gain.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state = None
    for key in ("model", "module"):          # --model_key model|module
        if key in ckpt:
            state = ckpt[key]
            break
    if state is None:
        state = ckpt

    # upstream strips the 'student.' prefix and drops everything else
    if any(k.startswith("student.") for k in state):
        state = OrderedDict((k[len("student."):], v) for k, v in state.items()
                            if k.startswith("student."))

    # Upstream drops relative_position_index keys and any head whose shape
    # disagrees, then calls its own non-strict utils.load_state_dict. So the
    # pretraining-only tensors (mask_token, lm_head, and the pre-pooling `norm`
    # that use_mean_pooling replaces with fc_norm) are expected to be unused,
    # and fc_norm/head are expected to start random. Replicated here.
    state = {k: v for k, v in state.items() if "relative_position_index" not in k}
    model_sd = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in state and k in model_sd and state[k].shape != model_sd[k].shape:
            del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Only the known-benign names may be missing or unused. Anything else means
    # the weights and the architecture genuinely disagree, which would leave part
    # of the encoder random while still being reported as "pretrained".
    BENIGN_MISSING = ("fc_norm.", "head.")
    BENIGN_UNEXPECTED = ("mask_token", "lm_head.", "norm.")
    # The cross-corpus path enables the relative position bias (EEGPT's setting),
    # whose tables are not in the released checkpoint. Upstream loads with
    # strict=False and lets them start random, so the same is allowed here --
    # but only for these tables, not for anything else.
    BENIGN_MISSING_SUBSTR = ("relative_position_bias_table",)
    bad_missing = [k for k in missing
                   if not k.startswith(BENIGN_MISSING)
                   and not any(s in k for s in BENIGN_MISSING_SUBSTR)]
    bad_unexpected = [k for k in unexpected if not k.startswith(BENIGN_UNEXPECTED)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"labram-base.pth does not match the model: "
            f"missing={bad_missing[:5]} unexpected={bad_unexpected[:5]}")

    # Positive check: the transformer blocks must actually have been populated,
    # otherwise a checkpoint with a different key layout could load "cleanly"
    # by matching nothing at all.
    loaded_blocks = sum(1 for k in state if k.startswith("blocks."))
    if loaded_blocks == 0:
        raise RuntimeError("no transformer block weights were loaded from the checkpoint")
    print(f"  loaded {len(state)} tensors ({loaded_blocks} block tensors); "
          f"random-init: {sorted(set(k.split('.')[0] for k in missing))}")


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
