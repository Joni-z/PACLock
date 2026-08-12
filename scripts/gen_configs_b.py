"""Generate group-B experiment configs from each repo's own settings.

    python -m scripts.gen_configs_b [--out configs/experiments]

Group B's rule is that every model runs its *own* repo's preprocessing,
normalisation and fine-tune recipe. So none of the numbers below are ours --
each is traceable to a line in the upstream repository, and the provenance is
recorded next to it.

BIOT (ycq091044/BIOT), from run_binary_supervised.py / run_multiclass_supervised.py
and the reference commands in its README:

    --token_size 200 --hop_length 100 --sampling_rate 200
    lr 1e-3, weight_decay 1e-5, epochs 100, plain Adam
    TUAB: --batch_size 512 --sample_length 10 --in_channels 16
    TUEV: --batch_size 128 --sample_length 5  --in_channels 16 --n_classes 6
    EarlyStopping(monitor="val_auroc", patience=5, mode="max")

Checkpoints and the xlsx rows they correspond to:

    EEG-PREST-16-channels        -> "BIOT★ (单语料预训练)"
    EEG-six-datasets-18-channels -> "BIOT (4 语料)"; BIOT reports TUAB AUROC
                                    0.8815, the exact value the xlsx lists

The 18-channel checkpoints need C3-A2 and C4-A1 on top of the 16 bipolar
montages, which our TUH preprocessing does not produce, so only the 16-channel
checkpoint is generated here. Adding the other two is a preprocessing change,
not a config change, and is tracked separately.
"""

from __future__ import annotations

import argparse
import os

import yaml

PROC_BIOT = "/work1/chenyuyou/yifanwang/Zhizhe/processed_biot"
PROC_OURS = "/work1/chenyuyou/yifanwang/Zhizhe/processed"

# BIOT argparse defaults, identical across both run scripts
BIOT_COMMON = {
    "model": "biot",
    "loader": "biot",              # its own Dataset + q95 normalisation
    "optimizer": "adam",           # BIOT uses plain Adam
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 100,
    "scheduler": None,
    "grad_clip": None,
    "patience": 5,                 # EarlyStopping(patience=5)
    "num_workers": 16,
    "device": "cuda",
    "model_kwargs": {"token_size": 200, "hop_length": 100},
}

# per-dataset values taken from the README reference commands
BIOT_DATASETS = {
    "tuab": {
        "num_classes": 2, "sample_rate": 200, "batch_size": 512,
        "loss": "bce_with_logits", "eval_every_steps": 0,
    },
    "tuev": {
        "num_classes": 6, "sample_rate": 200, "batch_size": 128,
        "loss": "cross_entropy", "label_smoothing": 0.0,   # BIOT uses plain CE
        "eval_every_steps": 0,
    },
    # CHB-MIT: BIOT's CHBMITLoader takes 256 Hz pickles, resamples to 200 in the
    # loader and applies the same q95 normalisation, over the same 16 bipolar
    # montage. Our group-A copy is already at 200 Hz with that montage, so it is
    # what BIOT would see -- no separate preprocessing pass is needed.
    "chbmit": {
        "num_classes": 2, "sample_rate": 200, "batch_size": 512,
        "loss": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0,
        "eval_every_steps": 0,
    },
    # TUSZ is the same 16 bipolar montage at 200 Hz; BIOT needs its own
    # unfiltered copy, produced by preprocessing.biot_native.
    "tusz": {
        "num_classes": 2, "sample_rate": 200, "batch_size": 512,
        "loss": "bce_with_logits", "eval_every_steps": 0,
    },
}

# group B rows: one config per (dataset, checkpoint)
BIOT_ROWS = {
    "prest16": "BIOT (pretrained)",     # the xlsx's "BIOT★" row
}


# LaBraM, from its README's TUAB fine-tune command and run_class_finetuning.py
# defaults. Its recipe shares almost nothing with BIOT's: AdamW with layer-wise
# LR decay and warmup, vs BIOT's plain Adam at a flat 1e-3.
LABRAM_COMMON = {
    "model": "labram",
    "loader": "labram",
    "pretrained": True,
    "optimizer": "adamw",
    "lr": 5e-4,               # --lr 5e-4
    "weight_decay": 0.05,     # --weight_decay 0.05
    "epochs": 50,             # --epochs 50
    "batch_size": 64,         # --batch_size 64
    "warmup_epochs": 5,       # --warmup_epochs 5
    "layer_decay": 0.65,      # --layer_decay 0.65
    "scheduler": "cosine",    # LaBraM uses cosine after warmup
    "grad_clip": None,
    "patience": 10,
    "num_workers": 16,
    "device": "cuda",
    "eval_every_steps": 0,
}

# LaBraM indexes its positional embedding by electrode identity and needs the
# 23 unipolar -REF channels, so it is limited to the TUH corpora. CHB-MIT and
# the rest ship bipolar-only or entirely different montages, and a bipolar pair
# cannot be inverted back into two unipolar signals.
LABRAM_DATASETS = {
    "tuab": {"num_classes": 2, "sample_rate": 200, "eval_every_steps": 100, "loss": "bce_with_logits"},
    "tuev": {"num_classes": 6, "sample_rate": 200, "eval_every_steps": 100, "loss": "cross_entropy",
             "label_smoothing": 0.0},
    "tusz": {"num_classes": 2, "sample_rate": 200, "eval_every_steps": 100,
             "val_subsample": 20000, "loss": "bce_with_logits"},
}

PROC_LABRAM = "/work1/chenyuyou/yifanwang/Zhizhe/processed_labram"


# ---------------------------------------------------------------------------
# Cross-corpus BIOT / LaBraM rows
# ---------------------------------------------------------------------------
# Neither repo ships a dataset maker for Sleep-EDF, ISRUC, PhysioNet-MI,
# BCI-IV-2a or FACED, so neither defines a recipe there. EEGPT's repo does:
# vendor/eegpt/downstream/{finetune,linear_probe}_{BIOT,LaBraM}_{SleepEDF,BCIC2A,
# BCIC2B,KaggleERN,PhysioP300}.py are the published provenance of the BIOT and
# LaBraM baseline numbers in the EEGPT paper (NeurIPS 2024). Those scripts are
# copied here, which is what makes these rows comparable to the literature.
#
# Two consequences that look wrong but are upstream's, see channel_adapt.py:
#   * BIOT does not match montages -- a trained 1x1 conv projects the dataset's
#     channels onto the checkpoint's 16.
#   * LaBraM does not consult standard_1020 -- input_chans is range(C+1),
#     i.e. positional. So bipolar-only corpora need no unipolar reconstruction.
#
# Shared schedule, identical in all four scripts:
#   AdamW(weight_decay=0.01); OneCycleLR(max_lr=4e-4, pct_start=0.2)
XCORPUS_SCHEDULE = {
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "lr": 4e-4,
    "max_lr": 4e-4,
    "scheduler": "onecycle",
    "pct_start": 0.2,
    "batch_size": 64,           # finetune_*: 8*8; linear_probe_*: 64
    "grad_clip": None,
    "patience": 10,
    "num_workers": 16,
    "device": "cuda",
    "eval_every_steps": 0,
}

# target_len is the length fed to temporal_interpolation, in samples at 200 Hz.
# finetune_*_SleepEDF.py uses 200*15 for a 30 s epoch rather than the native
# length; the other corpora are already short enough to pass through natively.
XCORPUS_DATASETS = {
    "sleepedf": {"num_classes": 5, "target_len": 15 * 200, "epochs": 40,
                 "loss": "cross_entropy"},
    # ISRUC is stored as 20-epoch sequences; every group-B model here consumes
    # one epoch at a time, so the loader flattens (n, 20, C, T) -> (n*20, C, T).
    "isruc": {"num_classes": 5, "target_len": 15 * 200, "epochs": 40,
              "loss": "cross_entropy", "flatten_sequences": True},
    "physionet_mi": {"num_classes": 4, "target_len": 4 * 200, "epochs": 100,
                     "loss": "cross_entropy"},
    "bci_iv_2a": {"num_classes": 4, "target_len": 4 * 200, "epochs": 100,
                  "loss": "cross_entropy"},
    "faced": {"num_classes": 9, "target_len": 10 * 200, "epochs": 100,
              "loss": "cross_entropy"},
    # CHB-MIT already matches BIOT's 16-channel montage so it keeps BIOT's own
    # recipe above; LaBraM has no maker for it and takes the positional path.
    "chbmit": {"num_classes": 2, "target_len": 10 * 200, "epochs": 100,
               "loss": "cross_entropy", "labram_only": True},
}


# EEGPT (BINE0000/EEGPT), from downstream/finetune_EEGPT_SleepEDF.py. Same
# AdamW + OneCycleLR schedule as the cross-corpus BIOT/LaBraM scripts in that
# repo; the per-dataset channel list and window length live in the adapter,
# because upstream sizes img_size per corpus.
# On the large corpora one epoch is thousands of steps and a pretrained model
# converges inside epoch 0, so a once-per-epoch curve cannot tell "converged
# fast" from "never learned" -- which is exactly what hard rule 3 asks. These
# therefore validate mid-epoch, against a capped subset of val so the cost stays
# bounded (TUSZ's dev split alone is 156,395 windows). Test metrics always use
# the full test split; only checkpoint selection sees the subset.
EEGPT_DATASETS = {
    "tuab": {"num_classes": 2, "loss": "bce_with_logits", "epochs": 40,
             "eval_every_steps": 500, "val_subsample": 20000},
    "tuev": {"num_classes": 6, "loss": "cross_entropy", "epochs": 40,
             "eval_every_steps": 200},
    "tusz": {"num_classes": 2, "loss": "bce_with_logits", "epochs": 40,
             "eval_every_steps": 500, "val_subsample": 20000},
    "chbmit": {"num_classes": 2, "loss": "focal", "focal_alpha": 0.25,
               "focal_gamma": 2.0, "epochs": 40,
               "eval_every_steps": 500, "val_subsample": 20000},
    "sleepedf": {"num_classes": 5, "loss": "cross_entropy", "epochs": 40},
    "isruc": {"num_classes": 5, "loss": "cross_entropy", "epochs": 40,
              "flatten_sequences": True},
    "physionet_mi": {"num_classes": 4, "loss": "cross_entropy", "epochs": 100},
    "bci_iv_2a": {"num_classes": 4, "loss": "cross_entropy", "epochs": 100},
    "faced": {"num_classes": 9, "loss": "cross_entropy", "epochs": 100},
}


def gen_eegpt(out_dir: str) -> list[str]:
    """EEGPT on all nine corpora, pretrained and from-scratch."""
    written = []
    for ds, dcfg in EEGPT_DATASETS.items():
        for tag, pre, grp in (("pretrained", True, "B"), ("scratch", False, "C")):
            cfg = {
                "name": f"{ds}-eegpt_{tag}",
                "group": grp,
                "dataset": ds,
                "data_root": os.path.join(PROC_OURS, ds),
                "model": "eegpt",
                "loader": "default",
                "pretrained": pre,
                "sample_rate": 200,
                # batch 32 (8*4) in finetune_EEGPT_SleepEDF.py, unlike the
                # 64 used by the BIOT/LaBraM scripts in the same repo
                **dict(XCORPUS_SCHEDULE, batch_size=32),
                **dcfg,
                "seed": 0,
            }
            path = os.path.join(out_dir, f"{ds}_eegpt_{tag}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)
    return written


def gen_xcorpus(out_dir: str) -> list[str]:
    """BIOT and LaBraM on the corpora their own repos do not cover."""
    written = []
    for ds, dcfg in XCORPUS_DATASETS.items():
        d = {k: v for k, v in dcfg.items() if k not in ("target_len", "labram_only")}
        target_len = dcfg["target_len"]

        specs = []
        if not dcfg.get("labram_only"):
            specs.append(("biot", {
                "model": "biot",
                "loader": "biot",
                "checkpoint": "prest16",
                "model_kwargs": {"token_size": 200, "hop_length": 100,
                                 "target_len": target_len},
            }, "prest16"))
        # LaBraM keeps its *own* fine-tuning recipe here, not EEGPT's OneCycle.
        # Protocol rule 2 is "each model's own recipe", and only the montage
        # handling has to be borrowed from EEGPT, because LaBraM's repo defines
        # no montage for these corpora. The difference is not cosmetic: layer-wise
        # LR decay is what stops a flat 4e-4 from washing out the pretrained
        # blocks -- adding it to TUEV moved kappa 0.4130 -> 0.6169, and without
        # it PhysioNet-MI sat at train_loss = ln(4) for every epoch.
        labram_cfg = {k: v for k, v in LABRAM_COMMON.items()
                      if k not in ("model", "loader", "pretrained")}
        specs.append(("labram", {
            **labram_cfg,
            "model": "labram",
            "loader": "labram",
            "pretrained": True,
            # The group-A arrays already carry the protocol's normalisation, so
            # LaBraM's own /100 must not be applied a second time -- doing so
            # drove the input to std ~1e-3 and collapsed the model to a constant
            # class on every div100 corpus. See labram_dataset.py.
            "loader_divisor": 1.0,
            "model_kwargs": {"montage_mode": "positional", "target_len": target_len},
        }, "pretrained"))

        for model, mcfg, tag in specs:
            cfg = {
                "name": f"{ds}-{model}_{tag}",
                "group": "B",
                "dataset": ds,
                # both read the group-A processed copy: it is already 200 Hz,
                # and the cross-corpus path resamples/re-references on top of it
                # exactly as EEGPT's scripts do
                "data_root": os.path.join(PROC_OURS, ds),
                "sample_rate": 200,
                **XCORPUS_SCHEDULE,
                **d,
                # mcfg last: the per-model recipe wins over the corpus defaults
                # taken from EEGPT's scripts, so LaBraM keeps its own optimiser,
                # schedule and epoch count (protocol rule 2).
                **mcfg,
                "seed": 0,
            }
            path = os.path.join(out_dir, f"{ds}_{model}_{tag}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)
    return written


def gen_labram(out_dir: str) -> list[str]:
    written = []
    for ds, dcfg in LABRAM_DATASETS.items():
        for tag, pre, grp in (("pretrained", True, "B"), ("scratch", False, "C")):
            cfg = {
                "name": f"{ds}-labram_{tag}",
                "group": grp,
                "dataset": ds,
                "data_root": os.path.join(PROC_LABRAM, ds),
                **LABRAM_COMMON,
                "pretrained": pre,
                **dcfg,
                "seed": 0,
            }
            path = os.path.join(out_dir, f"{ds}_labram_{tag}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)
    return written


# CBraMod, from finetune_main.py's argparse and finetune_trainer.py.
# Its preprocessing IS our frozen protocol (0.3-75 Hz, 60 Hz notch, 16 bipolar,
# 200 Hz), so these rows read the group-A processed data directly.
CBRAMOD_COMMON = {
    "model": "cbramod",
    "loader": None,           # our standard loader; the data already matches
    "pretrained": True,
    "optimizer": "adamw",     # --optimizer AdamW
    "lr": 1e-4,               # --lr 1e-4
    "weight_decay": 5e-2,     # --weight_decay 5e-2
    "epochs": 50,             # --epochs 50
    "batch_size": 64,         # --batch_size 64
    "multi_lr": True,         # --multi_lr True
    "scheduler": "cosine",
    "grad_clip": None,
    "patience": 10,
    "num_workers": 16,
    "device": "cuda",
    "eval_every_steps": 0,
    "model_kwargs": {"dropout": 0.1},   # --dropout 0.1
}

# Every dataset in the matrix. CBraMod's own preprocessing is the frozen
# protocol, so it can read the group-A copy everywhere -- there is no dataset it
# is shape-incompatible with. Datasets without a published anchor still get a
# cell; the anchor only decides whether hard rule 1 can be *evaluated*, not
# whether the row should be run.
CBRAMOD_DATASETS = {
    "tuab": {"num_classes": 2, "sample_rate": 200, "eval_every_steps": 100, "loss": "bce_with_logits"},
    "tuev": {"num_classes": 6, "sample_rate": 200, "eval_every_steps": 100, "loss": "cross_entropy",
             "label_smoothing": 0.0},
    "tusz": {"num_classes": 2, "sample_rate": 200, "eval_every_steps": 100,
             "val_subsample": 20000, "loss": "bce_with_logits"},
    "chbmit": {"num_classes": 2, "sample_rate": 200, "eval_every_steps": 100,
               "loss": "focal",
               "focal_alpha": 0.25, "focal_gamma": 2.0},
    "sleepedf": {"num_classes": 5, "sample_rate": 100, "loss": "cross_entropy",
                 "label_smoothing": 0.1},
    # NOT flattened: model_for_isruc.py consumes the whole 20-epoch sequence
    # and classifies each epoch with cross-epoch context. Flattening here would
    # silently select the TUAB-style head instead.
    "isruc": {"num_classes": 5, "sample_rate": 200, "loss": "cross_entropy",
              "label_smoothing": 0.1},
    "physionet_mi": {"num_classes": 4, "sample_rate": 200,
                     "loss": "cross_entropy", "label_smoothing": 0.1},
    "bci_iv_2a": {"num_classes": 4, "sample_rate": 200,
                  "loss": "cross_entropy", "label_smoothing": 0.1},
    "faced": {"num_classes": 9, "sample_rate": 200, "loss": "cross_entropy",
              "label_smoothing": 0.1},
}


def gen_cbramod(out_dir: str) -> list[str]:
    written = []
    for ds, dcfg in CBRAMOD_DATASETS.items():
        for tag, pre, grp in (("pretrained", True, "B"), ("scratch", False, "C")):
            cfg = {
                "name": f"{ds}-cbramod_{tag}",
                "group": grp,
                "dataset": ds,
                "data_root": os.path.join(PROC_OURS, ds),
                **CBRAMOD_COMMON,
                "pretrained": pre,
                **dcfg,
                "seed": 0,
            }
            path = os.path.join(out_dir, f"{ds}_cbramod_{tag}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)
    return written


# TFM-Tokenizer, from downstream_transformer_finetuning.py's argparse.
TFM_COMMON = {
    "model": "tfm",
    "loader": "tfm",           # q95 in the loader, like BIOT
    "pretrained": True,
    "tfm_setting": "multiple",   # the xlsx's "TFM-Tokenizer" row
    # configs/tfm_tokenizer_training_configs.yaml, the `finetuning` block:
    #   batch_size 512, AdamW, lr 1e-3, weight_decay 1e-5,
    #   num_pretrain_epochs 50, warmup_epochs 5, label_smoothing 0.1
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 50,
    "batch_size": 512,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "scheduler": "cosine",
    "grad_clip": None,
    "patience": 10,
    "num_workers": 16,
    "device": "cuda",
    "eval_every_steps": 0,
}

PROC_TFM = "/work1/chenyuyou/yifanwang/Zhizhe/processed_tfm"

# TFM-Tokenizer runs on all nine corpora with the *multi-corpus* released
# weights (multiple_dataset_settings). Two reasons, both measured:
#   * the multi-corpus tokenizer matches the current code on all 191 keys, while
#     the single-corpus TUEV tokenizer ships freq_pos_embed / temporal_pos_embed
#     from an older model version the repo no longer builds;
#   * only the multi-corpus pair exists for corpora outside TUAB/TUEV/CHB-MIT.
# So this row corresponds to the workbook's "TFM-Tokenizer† (4 语料)" pretraining
# rather than the single-corpus one -- recorded here because the B-group row is
# labelled just "TFM-Tokenizer".
TFM_DATASETS = {
    "tuab": {"num_classes": 2, "loss": "bce_with_logits", "epochs": 8,
             "eval_every_steps": 100, "val_subsample": 20000},
    "tuev": {"num_classes": 6, "loss": "cross_entropy", "eval_every_steps": 100},
    # 8/10 epochs rather than upstream's 50: measured 5h07m for 500 steps plus
    # 5 full val passes on TUSZ, so 50 epochs cannot fit the 24h wall. This is a
    # deviation and is recorded as one; the validation curve is stored with each
    # run so a reader can check the curve had flattened.
    "tusz": {"num_classes": 2, "loss": "bce_with_logits", "epochs": 5,
             "eval_every_steps": 100, "val_subsample": 20000},
    "chbmit": {"num_classes": 2, "loss": "focal", "focal_alpha": 0.25,
               "focal_gamma": 2.0, "epochs": 7, "eval_every_steps": 100,
               "val_subsample": 20000},
    "sleepedf": {"num_classes": 5, "loss": "cross_entropy"},
    "isruc": {"num_classes": 5, "loss": "cross_entropy", "flatten_sequences": True},
    "physionet_mi": {"num_classes": 4, "loss": "cross_entropy"},
    "bci_iv_2a": {"num_classes": 4, "loss": "cross_entropy"},
    "faced": {"num_classes": 9, "loss": "cross_entropy"},
}


# Upstream's batch 512 is sized for its own corpora (TUAB, TUEV, CHB-MIT,
# IIIC), all of which are large. On BCI-IV-2a it would give 4 steps per epoch,
# which is the starvation failure group A already measured (0.4996 -> 0.6478
# after fixing it). So 512 is kept where it yields >= 100 steps/epoch and
# halved until it does otherwise -- the same rule, and the same floor, as
# scripts/gen_configs.py.
TFM_TRAIN_SIZE = {
    "tuab": 297103, "tuev": 68445, "tusz": 326668, "chbmit": 316205,
    "sleepedf": 122430, "isruc": 69420, "physionet_mi": 6300,
    "bci_iv_2a": 2160, "faced": 6720,
}


def tfm_batch_for(ds: str, default: int = 512) -> int:
    n = TFM_TRAIN_SIZE.get(ds)
    if n is None:
        return default
    bs = default
    while bs > 16 and n / bs < 100:
        bs //= 2
    return bs


def gen_tfm(out_dir: str) -> list[str]:
    written = []
    for ds, dcfg in TFM_DATASETS.items():
        for tag, pre, grp in (("pretrained", True, "B"), ("scratch", False, "C")):
            # All nine TFM rows read the group-A copy, deliberately and
            # consistently. TFM's repo ships a maker for only two of the nine
            # corpora, so sourcing per corpus would make a single row mix two
            # pipelines -- worse for interpretation than one stated choice. The
            # gap is small and bounded: TFM conditions at 0.1-75 Hz + 50 Hz notch
            # / 200 Hz against our 0.3-75 Hz + 60 Hz notch / 200 Hz, and its q95
            # normalisation is scale-invariant, so only the filter edges differ.
            root = PROC_OURS
            cfg = {
                "name": f"{ds}-tfm_{tag}",
                "group": grp,
                "dataset": ds,
                "data_root": os.path.join(root, ds),
                **TFM_COMMON,
                "batch_size": tfm_batch_for(ds),
                "pretrained": pre,
                **dcfg,
                "seed": 0,
            }
            path = os.path.join(out_dir, f"{ds}_tfm_{tag}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/experiments")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = []
    for ds, dcfg in BIOT_DATASETS.items():
        for ckpt in BIOT_ROWS:
            root = PROC_OURS if ds == "chbmit" else PROC_BIOT
            cfg = {
                "name": f"{ds}-biot_{ckpt}",
                "group": "B",
                "dataset": ds,
                "data_root": os.path.join(root, ds),
                "checkpoint": ckpt,
                **BIOT_COMMON,
                **dcfg,
                "seed": 0,
            }
            path = os.path.join(args.out, f"{ds}_biot_{ckpt}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)

        # the matching from-scratch run ("Vanilla BIOT"), same recipe, no weights.
        # It belongs to group C but is generated here so the only difference from
        # the group-B row is the checkpoint -- which is the whole point of the
        # comparison.
        cfg = {
            "name": f"{ds}-biot_scratch",
            "group": "C",
            "dataset": ds,
            "data_root": os.path.join(PROC_OURS if ds == "chbmit" else PROC_BIOT, ds),
            "checkpoint": None,
            **BIOT_COMMON,
            **dcfg,
            "seed": 0,
        }
        path = os.path.join(args.out, f"{ds}_biot_scratch.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        written.append(path)

    written += gen_labram(args.out)
    written += gen_cbramod(args.out)
    written += gen_tfm(args.out)
    written += gen_xcorpus(args.out)
    written += gen_eegpt(args.out)

    print(f"wrote {len(written)} configs")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
