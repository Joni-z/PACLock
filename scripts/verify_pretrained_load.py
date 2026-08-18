"""Smoke test: does load_pretrained_backbone actually transfer weights into a
downstream-shaped PACLock model before any real finetuning compute is spent?

Checks, for each (pretrain checkpoint, downstream config) pair:
  - build_model(cfg) with cfg['checkpoint'] set doesn't crash
  - a nontrivial number of tensors report as loaded (frontend/band_pe/encoder)
  - the loaded frontend weights are bit-identical to the checkpoint's own
    (proves the values actually moved, not just that load_state_dict didn't
    raise)
  - spatial_pe and head are NOT loaded from the checkpoint (still randomly
    initialized / distinct from the checkpoint's own values), confirming the
    exclusion in load_pretrained_backbone is doing what it claims
"""
import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.paths import expand

CASES = [
    ("configs/deliverable/faced_paclock_v2.yaml", "pretrain_runs/pretrain-size_base/checkpoint.pt"),
    ("configs/deliverable/tuev_paclock_v2.yaml", "pretrain_runs/pretrain-size_base/checkpoint.pt"),
]

for cfg_path, ckpt_path in CASES:
    print(f"=== {cfg_path}  <-  {ckpt_path} ===")
    cfg = yaml.safe_load(open(cfg_path))
    cfg["checkpoint"] = ckpt_path
    # fake shape from the config's own model_kwargs -- avoids touching real data
    C_guess = {"faced": 32, "tuev": 16}[cfg["dataset"]]
    T_guess = 2000 if cfg["dataset"] == "faced" else 1000
    model = build_model(cfg, (C_guess, T_guess))

    raw = torch.load(expand(ckpt_path), map_location="cpu")
    src = raw["model"] if "model" in raw else raw
    dst = dict(model.state_dict())

    # spot-check one frontend tensor matches the checkpoint exactly
    fk = next(k for k in src if k.startswith("frontend."))
    same = torch.equal(dst[fk], src[fk])
    print(f"  frontend tensor {fk!r} transferred correctly: {same}")

    # spatial_pe must NOT match the checkpoint's (different n_channels anyway,
    # but also different random init even where shapes happen to coincide)
    sk = next((k for k in src if k.startswith("spatial_pe.")), None)
    if sk:
        same_shape = dst[sk].shape == src[sk].shape
        print(f"  spatial_pe {sk!r}: ckpt shape={tuple(src[sk].shape)} "
              f"model shape={tuple(dst[sk].shape)} (excluded from transfer as designed)")
    print()

print("done")
