"""Layer-wise learning-rate decay, ported from LaBraM's optim_factory.py.

LaBraM's fine-tune recipe is not just "AdamW at 5e-4": it scales the LR per
depth, so early blocks move far less than late ones. Its TUAB command passes
``--layer_decay 0.65``, which over 12 blocks makes the first block's LR about
0.65^12 ~ 0.006 of the head's. Running it without that is a different recipe and
systematically under-fits the pretrained features -- which is what we saw:
TUEV kappa 0.4130 against a published 0.5067.

``get_num_layer_for_vit`` and the group construction below follow
``vendor/labram/optim_factory.py`` verbatim.
"""

from __future__ import annotations

import torch.nn as nn


def get_num_layer_for_vit(var_name: str, num_max_layer: int) -> int:
    """Depth index of a parameter. From LaBraM optim_factory.py."""
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    if var_name.startswith("patch_embed"):
        return 0
    if var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    if var_name.startswith("blocks"):
        return int(var_name.split(".")[1]) + 1
    return num_max_layer - 1


def layer_decay_param_groups(model: nn.Module, base_lr: float,
                             weight_decay: float, layer_decay: float,
                             n_layers: int | None = None,
                             skip_list: tuple[str, ...] = ("pos_embed", "cls_token")):
    """Build AdamW param groups with per-depth LR scaling and no-decay handling.

    Bias and 1-D parameters (norms) get weight_decay=0, as upstream does: decaying
    a LayerNorm gain pulls it toward zero, which is not what weight decay is for.

    ``model`` may be our LaBraMWrapper, whose parameters are named
    ``model.blocks.0...``; the prefix is stripped before the depth lookup so the
    names match what upstream's function expects.
    """
    inner = getattr(model, "model", model)          # unwrap LaBraMWrapper
    if n_layers is None:
        n_layers = (inner.get_num_layers() if hasattr(inner, "get_num_layers")
                    else len(getattr(inner, "blocks", [])))
    num_max_layer = n_layers + 2
    scales = [layer_decay ** (num_max_layer - 1 - i) for i in range(num_max_layer)]

    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        bare = name[len("model."):] if name.startswith("model.") else name
        if param.ndim == 1 or name.endswith(".bias") or bare in skip_list:
            gname, wd = "no_decay", 0.0
        else:
            gname, wd = "decay", weight_decay

        layer_id = get_num_layer_for_vit(bare, num_max_layer)
        key = f"layer_{layer_id}_{gname}"
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": wd,
                "lr": base_lr * scales[layer_id],
                "layer_id": layer_id,
            }
        groups[key]["params"].append(param)

    return [g for g in groups.values() if g["params"]]
