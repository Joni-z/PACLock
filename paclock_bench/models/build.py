"""Model builder. One entry point for every group in the matrix."""

from __future__ import annotations

import torch.nn as nn

from .baselines.light_supervised import REGISTRY as LIGHT_REGISTRY
from ..paths import expand


def build_model(cfg: dict, input_shape: tuple[int, ...]) -> nn.Module:
    """``input_shape`` is (C, T) taken from the data, not from the config, so a
    config/data mismatch surfaces as a shape error at build time rather than as
    a silently wrong model."""
    name = cfg["model"]
    C, T = input_shape[-2], input_shape[-1]
    mk = dict(cfg.get("model_kwargs", {}))
    # model_kwargs overrides rather than duplicates: splatting it alongside the
    # derived defaults raises TypeError on any key that appears in both.
    kwargs = dict(
        in_channels=C,
        seq_len=T,
        num_classes=cfg["num_classes"],
        sample_rate=cfg.get("sample_rate", 200),
    )
    kwargs.update(mk)

    if name in LIGHT_REGISTRY:
        return LIGHT_REGISTRY[name](**kwargs)

    if name == "biot":
        # Group B/C: official BIOT code and, for group B, official weights.
        # checkpoint=None is the from-scratch (group C / "Vanilla BIOT") row.
        from .foundation.biot_adapter import build_biot

        return build_biot(
            n_classes=cfg["num_classes"],
            n_channels=C,
            checkpoint=cfg.get("checkpoint"),
            token_size=mk.get("token_size", 200),
            hop_length=mk.get("hop_length", 100),
            # cross-corpus rows: length fed to temporal_interpolation before the
            # 1x1 channel projection (see channel_adapt). Absent for the TUH /
            # CHB-MIT rows, whose channel count already matches the checkpoint.
            target_len=mk.get("target_len"),
        )

    if name == "labram":
        # Group B/C: official LaBraM code + labram-base.pth.
        # pretrained=False is the from-scratch (group C) row.
        from .foundation.labram_adapter import build_labram

        return build_labram(
            n_classes=cfg["num_classes"],
            n_channels=C,
            pretrained=bool(cfg.get("pretrained", True)),
            # 'electrode' for the TUH corpora, where LaBraM ships a dataset
            # maker defining the montage; 'positional' elsewhere, following
            # EEGPT's cross-corpus scripts.
            montage_mode=mk.get("montage_mode", "electrode"),
            target_len=mk.get("target_len"),
        )

    if name == "eegpt":
        # Group B/C: official EEGPT code + eegpt_mcae_58chs_4s_large4E.ckpt.
        # The channel list and window length come from the dataset, since
        # upstream sizes img_size per corpus.
        from .foundation.eegpt_adapter import build_eegpt

        return build_eegpt(
            n_classes=cfg["num_classes"],
            n_channels=C,
            dataset=cfg["dataset"],
            pretrained=bool(cfg.get("pretrained", True)),
        )

    if name == "cbramod":
        # Group B/C: official CBraMod code + pretrained_weights.pth.
        # Reads our own preprocessed data -- CBraMod's preprocessing IS the
        # frozen protocol (see cbramod_adapter docstring).
        from .foundation.cbramod_adapter import build_cbramod

        return build_cbramod(
            n_classes=cfg["num_classes"],
            n_channels=C,
            seq_len=T,
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=cfg.get("model_kwargs", {}).get("dropout", 0.1),
            # a 3-D sample (seq, channels, time) means the corpus is stored as
            # epoch sequences -- ISRUC -- where upstream uses model_for_isruc.py
            # rather than the flatten-and-classify head
            sequence=len(input_shape) == 3,
        )

    if name == "cbramod_paclockfe":
        # Ablation, not a matrix row: CBraMod's architecture (encoder,
        # positional encoding, classifier head, all vendor code, all
        # unmodified) with its own tokenizer replaced by PACLock's frontend.
        # See foundation/cbramod_paclockfe_adapter.py for exactly what is and
        # is not swapped.
        from .foundation.cbramod_paclockfe_adapter import build_cbramod_paclockfe

        return build_cbramod_paclockfe(
            n_classes=cfg["num_classes"],
            n_channels=C,
            seq_len=T,
            dropout=cfg.get("model_kwargs", {}).get("dropout", 0.1),
            sequence=len(input_shape) == 3,
        )

    if name == "tfm":
        # Group B/C: official TFM-Tokenizer code + its shipped weights.
        from .foundation.tfm_adapter import build_tfm

        return build_tfm(
            n_classes=cfg["num_classes"],
            dataset=cfg["dataset"],
            pretrained=bool(cfg.get("pretrained", True)),
            setting=cfg.get("tfm_setting", "single"),
            n_channels=C,
        )

    if name == "paclock":
        from .paclock.build import build_model as build_paclock, load_pretrained_backbone

        # the vendored builder is config-driven; hand it the resolved shape
        pac_cfg = {
            **cfg.get("model_kwargs", {}),
            "n_channels": C,
            "seq_len": T,
            "num_classes": cfg["num_classes"],
            "sample_rate": cfg.get("sample_rate", 200),
            "sampling_rate": cfg.get("sample_rate", 200),
            "dataset": cfg["dataset"],
        }
        model = build_paclock(pac_cfg)
        # cfg['checkpoint'] points at a training/pretrain.py checkpoint.pt --
        # transfers frontend/band_pe/encoder only (see load_pretrained_backbone's
        # docstring for why spatial_pe and the head are excluded).
        ckpt = cfg.get("checkpoint")
        if ckpt:
            report = load_pretrained_backbone(model, expand(ckpt))
            print(f"[paclock] loaded {len(report['loaded'])} pretrained backbone "
                  f"tensors from {ckpt}"
                  + (f"; skipped {len(report['skipped_shape'])} shape mismatches: "
                     f"{report['skipped_shape']}" if report["skipped_shape"] else ""))
        return model

    raise KeyError(
        f"unknown model {name!r}. Registered: "
        f"{sorted(LIGHT_REGISTRY)} + ['paclock']. "
        f"Group B/C foundation models are added under models/foundation/."
    )


def count_params(model: nn.Module) -> float:
    """Millions of parameters -- the matrix reports this per row."""
    return sum(p.numel() for p in model.parameters()) / 1e6
