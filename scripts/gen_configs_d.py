"""Generate PACLock configs: the model's own row, plus the paper's control arms.

    python -m scripts.gen_configs_d [--out configs/experiments] [--arms]

PACLock is ours, so unlike groups A/B there is no upstream recipe to copy. The
architecture is vendored bit-identical from ``Joni-z/PACLock`` (see
models/paclock/PROVENANCE.md); what is fixed here is the size and the training
recipe.

Size: ``d_model=128, depth=6, n_bands=8`` measures **1.636M** parameters, which
is the 1.64 the workbook lists for "PACLock (from scratch, full)". That match is
the check that we are building the same model the row refers to -- the other
grid points (0.29M .. 3.37M) are all far off.

Operator configuration follows the paper, §4.4: the token is
``h_j = a_j ⊙ Σ_{i<j} α_ij e^{−i∠Z_ij} p_i``, and "the frequency-axis mixer is
plain attention: PAC enters the model in exactly one place". So the full model is

    arch=triaxial, tokenizer_mode=pac_interaction,
    pac_token_mode=measured, interaction_mode=product, freq_mixer=attention

Training recipe matches group A's, deliberately: §6.1 requires that every arm
holds "data, splits, preprocessing, optimiser, schedule and seed fixed and
varies only the tokenizer". Using group A's schedule also keeps PACLock
comparable to the supervised baselines it sits beside in the table.
"""

from __future__ import annotations

import argparse
import os

import yaml

PROC = "/work1/chenyuyou/yifanwang/Zhizhe/processed"

# measured 1.636M against the workbook's 1.64
ARCH = {
    "model": "paclock",
    "model_kwargs": {
        "arch": "triaxial",
        "d_model": 128,
        "depth": 6,
        "n_bands": 8,
        "n_heads": 4,
        "dropout": 0.2,
        "kernel_size": 201,
        "patch_len": 200,
        "augmentations": [],
        "freq_mixer": "attention",
        # AGENT.md:2974 names these as the architecture of record
        # ("backbone: tri-axial, band_pe: index, spatial_pe: xyz"), and :2490
        # records that omitting band_pe silently yields `hz`. They are
        # architectural, not tuning knobs -- leaving them at the defaults builds
        # a different model than the one the reference results came from.
        "band_pe": "index",
        "tokenizer_mode": "pac_interaction",
        "pac_token_mode": "measured",
        "interaction_mode": "product",
    },
}

# Table 1 of the paper. Each arm removes exactly one ingredient of equation 4 and
# thereby targets one alternative explanation; all are parameter-matched to the
# full operator except `concat`, which upstream already notes carries slightly
# more parameters -- a bias *against* our own operator, so it is kept.
ARMS = {
    "raw":       {"tokenizer_mode": "raw"},
    "uniform":   {"pac_token_mode": "uniform"},
    "magnitude": {"pac_token_mode": "magnitude"},
    "concat":    {"interaction_mode": "concat"},
    "scramble":  {"pac_token_mode": "scramble"},
    "measured":  {},                                  # the full operator
}

# group A's schedule, per dataset -- see scripts/gen_configs.py for the
# batch-size floor these came from (100 steps/epoch minimum).
SCHEDULE = {
    "tuab":         dict(num_classes=2, epochs=20, batch_size=512, eval_every_steps=200,
                         patience=8, loss="bce_with_logits", sample_rate=200),
    "tuev":         dict(num_classes=6, epochs=100, batch_size=512, eval_every_steps=0,
                         patience=15, loss="cross_entropy", label_smoothing=0.1,
                         sample_rate=200),
    "tusz":         dict(num_classes=2, epochs=20, batch_size=512, eval_every_steps=200,
                         patience=8, loss="bce_with_logits", sample_rate=200,
                         val_subsample=20000),
    "chbmit":       dict(num_classes=2, epochs=20, batch_size=512, eval_every_steps=200,
                         patience=8, loss="focal", focal_alpha=0.25, focal_gamma=2.0,
                         sample_rate=200),
    "sleepedf":     dict(num_classes=5, epochs=60, batch_size=512, eval_every_steps=0,
                         patience=15, loss="cross_entropy", label_smoothing=0.1,
                         sample_rate=100),
    "isruc":        dict(num_classes=5, epochs=60, batch_size=512, eval_every_steps=0,
                         patience=15, loss="cross_entropy", label_smoothing=0.1,
                         sample_rate=200, flatten_sequences=True),
    "physionet_mi": dict(num_classes=4, epochs=100, batch_size=64, eval_every_steps=0,
                         patience=20, loss="cross_entropy", label_smoothing=0.1,
                         sample_rate=200),
    "bci_iv_2a":    dict(num_classes=4, epochs=100, batch_size=16, eval_every_steps=0,
                         patience=20, loss="cross_entropy", label_smoothing=0.1,
                         sample_rate=200),
    "faced":        dict(num_classes=9, epochs=100, batch_size=64, eval_every_steps=0,
                         patience=20, loss="cross_entropy", label_smoothing=0.1,
                         sample_rate=200),
}

# Verbatim from the validated reference configs (configs/pacint_tuev_measured.yaml
# and pacint_tuab_measured.yaml in Joni-z/PACLock), which are the configs that
# produced TUEV kappa 0.5493 and Sleep-EDF 0.5732. Deviating from them is how the
# first AMD attempt ended up 10x over on the learning rate.
COMMON = {
    "optimizer": "adamw",
    "lr": 1e-4,               # NOT 1e-3
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "grad_clip": 1.0,
    "num_workers": 8,
    "device": "cuda",
}

# The reference runs use batch 32 and 20 epochs on TUEV/TUAB, 60 epochs on
# Sleep-EDF (which also carries augmentations). Patience is generous (20) because
# §13.43-J2 records that the PAC tokenizer "lengthens useful training" -- it
# early-stopped at epoch 46 where the raw baseline stopped at 22.
REFERENCE_SCHEDULE = {
    "batch_size": 32,
    "epochs": 20,
    "patience": 20,
}

# Windows in each corpus's training split, needed to turn the reference schedule
# into a step count. Read off the class_counts stored in the finished runs.
TRAIN_WINDOWS = {
    "tuab": 297103, "tuev": 68445, "tusz": 326668, "chbmit": 316205,
    "sleepedf": 122430, "isruc": 69420, "physionet_mi": 6300,
    "bci_iv_2a": 2160, "faced": 6720,
}

# The reference schedule is 20 epochs, but "20 epochs" is not the quantity that
# was validated -- the number of optimiser steps is. On TUEV at batch 32 that is
# 68445/32*20 = 42760 steps. Copying the epoch count to a small corpus silently
# copies a tenth of the training: BCI-IV-2a has 2160 windows, so 20 epochs is
# 1340 steps, and PhysioNet-MI and FACED are little better.
#
# The consequence was not subtle. On FACED the training loss went 2.2207 ->
# 2.1858 across the whole run against a ln(9) = 2.1972 floor -- the model never
# fit the training set, and all three seeds landed on Cohen's Kappa ~0.018 while
# CBraMod from scratch reached 0.40 on the identical data.
#
# So the budget is held in steps and the epoch count is derived per corpus. This
# is the fair reading rather than a generous one: every model in the matrix runs
# its own repo's recipe, and PACLock's recipe says how long to train, not how
# many times to sweep whatever corpus happens to be in front of it.
REFERENCE_STEPS = TRAIN_WINDOWS["tuev"] // REFERENCE_SCHEDULE["batch_size"] \
    * REFERENCE_SCHEDULE["epochs"]

# Wall-clock ceiling. FACED runs ~7 min/epoch (32 channels x 8 bands x 10
# patches = 2560 tokens per window), so the full step budget would need 25h
# against a 24h partition limit. Corpora not listed here are unconstrained.
EPOCH_CAP = {"faced": 120}


def epochs_for(ds: str, batch_size: int) -> int:
    """Epoch count giving this corpus *at least* the reference step budget.

    A floor, not a target. Matching the budget exactly would cut TUAB, TUSZ and
    CHB-MIT from 20 epochs to 4, since one epoch of TUAB is already 9284 steps --
    a real reduction on the corpora that currently work, to fix corpora that do
    not. The floor only ever lengthens training.

    Lengthening is close to free here because it does not decide when training
    ends: ``patience`` does. The large corpora early-stop long before their
    twentieth epoch (TUAB stopped after 24 evaluations at eval_every_steps=200,
    about half an epoch), so raising their ceiling changes nothing about what
    they actually run. On the small corpora the ceiling was the binding
    constraint and the model was still descending when it hit it.
    """
    steps_per_epoch = max(1, TRAIN_WINDOWS[ds] // batch_size)
    epochs = max(1, REFERENCE_STEPS // steps_per_epoch)
    cap = EPOCH_CAP.get(ds)
    if cap:
        epochs = min(epochs, cap)
    return max(REFERENCE_SCHEDULE["epochs"], epochs)


# (channels, samples) each corpus feeds the model, so the token count can be
# computed before anything is allocated.
SHAPE = {
    "tuab": (16, 2000), "tuev": (16, 1000), "tusz": (16, 2000),
    "chbmit": (16, 2000), "sleepedf": (2, 3000), "isruc": (6, 6000),
    "physionet_mi": (64, 800), "bci_iv_2a": (22, 800), "faced": (32, 2000),
}
TOKEN_BUDGET = 100_000      # tokens per batch that fit in one MI210


def batch_for(ds: str, default: int) -> int:
    """Cap the batch so the tri-axial token tensor fits in memory.

    Unlike the flat baselines, PACLock's frontend expands each window into
    ``channels x bands x patches`` tokens: ISRUC is 6 x 8 x 30 = 1440 per
    sample, so group A's batch of 512 asks for 737k tokens of width 128 and the
    run dies with a HIP OOM (observed). The cap is on tokens, not windows, so it
    tracks the corpus that actually causes the blow-up rather than guessing per
    dataset.
    """
    C, T = SHAPE[ds]
    per_sample = C * ARCH["model_kwargs"]["n_bands"] * (T // ARCH["model_kwargs"]["patch_len"])
    bs = default
    while bs > 8 and bs * per_sample > TOKEN_BUDGET:
        bs //= 2
    return bs


# models/paclock/montage.py carries 6-D endpoint coordinates for the bipolar
# montages only. `spatial_pe: xyz` on a corpus without them silently falls back
# to the learned index embedding, so it is set per dataset and recorded in the
# config rather than left to that fallback -- otherwise two runs with the same
# written config would have different architectures.
HAS_COORDS = {"tuab", "tuev", "tusz", "chbmit", "sleepedf"}


def make(ds: str, arm: str | None) -> dict:
    mk = dict(ARCH["model_kwargs"])
    mk["spatial_pe"] = "xyz" if ds in HAS_COORDS else "index"
    if arm is not None:
        mk.update(ARMS[arm])
    name = f"{ds}-paclock" + (f"_{arm}" if arm else "_full")
    batch_size = batch_for(ds, REFERENCE_SCHEDULE["batch_size"])
    return {
        "name": name,
        # the workbook's "PACLock (from scratch, full)" sits in the C block
        "group": "C" if arm in (None, "measured") else "D",
        "dataset": ds,
        "data_root": os.path.join(PROC, ds),
        "model": "paclock",
        **COMMON,
        **SCHEDULE[ds],
        **REFERENCE_SCHEDULE,
        "batch_size": batch_size,
        # after REFERENCE_SCHEDULE, which carries the epoch count that was only
        # ever right for TUEV/TUAB
        "epochs": epochs_for(ds, batch_size),
        "model_kwargs": mk,
        "seed": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/experiments")
    ap.add_argument("--arms", action="store_true",
                    help="also emit the five control arms of the paper's Table 1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = []
    for ds in SCHEDULE:
        specs = [None] if not args.arms else [None] + sorted(ARMS)
        for arm in specs:
            cfg = make(ds, arm)
            path = os.path.join(args.out, cfg["name"].replace("-", "_") + ".yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)

    print(f"wrote {len(written)} configs")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
