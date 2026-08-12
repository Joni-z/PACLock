"""Generate the experiment configs from the frozen dataset protocols.

    python -m scripts.gen_configs [--group A]

Every training knob that the protocol pins (loss, label smoothing, focal
parameters, class weighting, number of classes) is read out of
``configs/datasets/<ds>.yaml`` rather than retyped here, so a config can never
drift from the protocol it claims to implement.

Only the optimisation schedule is chosen here. For group A it is deliberately
identical across the five models -- they are a calibration set, and giving each
one its own tuned schedule would make a failure to reproduce ambiguous between
"our pipeline is wrong" and "we tuned it differently".
"""

from __future__ import annotations

import argparse
from paclock_bench.paths import processed
import os

import yaml

DATASETS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
            "physionet_mi", "bci_iv_2a", "faced"]
GROUP_A = ["sparcnet", "contrawr", "cnn_transformer", "ffcl", "st_transformer"]

PROC_ROOT = processed("processed")

# BIOT's own training recipe, from run_binary_supervised.py /
# run_multiclass_supervised.py. Group A is the calibration set, so it trains the
# way the published numbers were trained -- an earlier pass used AdamW at
# lr=1e-4 with a cosine schedule and left SPaRCNet badly under-trained on TUEV
# (kappa 0.276 against a published 0.423).
OPTIMIZER = "adam"          # plain Adam, not AdamW
LR = 1e-3                   # 10x what the first pass used
WEIGHT_DECAY = 1e-5
SCHEDULER = None            # BIOT builds one and then does not return it

# Epoch budget is the one place we deviate: BIOT runs 100 epochs, which on
# TUAB/TUSZ/CHB-MIT (300k+ windows) is far past where these models peak and past
# our node time limit. Early stopping on the primary metric decides the actual
# stopping point; the cap only bounds the job. eval_every_steps samples the val
# curve inside an epoch on the large corpora so best-checkpoint selection is not
# quantised to epoch boundaries.
# Batch size follows dataset size rather than BIOT's fixed 512.
#
# BIOT chose 512 for corpora with 100k+ training windows, where that still gives
# 500+ optimiser steps per epoch. Applied unchanged to the small datasets it
# starves them: FACED (6,720 train windows) got 13 steps/epoch and never left
# chance level (BAcc 0.1111 = 1/9, flat for 25 epochs), and BCI-IV-2a (2,160)
# got 4 steps/epoch and scored 0.148 BAcc below what batch 64 reaches.
#
# Measured, same seed and schedule, only batch changed:
#   BCI-IV-2a  sparcnet  512 -> 0.4996   64 -> 0.6478
#   PhysioNet  sparcnet  512 -> 0.5733   64 -> 0.6000
#   FACED      contrawr  512 -> 0.1111   64 -> 0.1758
#
# Batch size is not a frozen protocol value -- the protocols pin data handling
# (rate, filtering, windows, splits, normalisation), not optimiser settings --
# and BIOT never ran these three corpora, so there is no official value to keep.
# The rule below targets >= ~100 steps per epoch, which is where the measured
# curves stop being starved. The large corpora keep BIOT's 512 exactly.
MIN_STEPS_PER_EPOCH = 100

TRAIN_SIZE = {          # train windows, from each manifest
    "tuab": 297103, "tuev": 68445, "tusz": 326668, "chbmit": 316205,
    "sleepedf": 122430, "isruc": 69420, "physionet_mi": 10400,
    "bci_iv_2a": 2160, "faced": 6720,
}


def batch_for(ds: str, default: int = 512) -> int:
    """Largest power-of-two batch <= default that still gives ~100 steps/epoch."""
    n = TRAIN_SIZE.get(ds)
    if n is None:
        return default
    bs = default
    while bs > 16 and n / bs < MIN_STEPS_PER_EPOCH:
        bs //= 2
    return bs


SCHEDULE = {
    "tuab":         dict(epochs=20, batch_size=512, eval_every_steps=200, patience=8),
    "tuev":         dict(epochs=100, batch_size=512, eval_every_steps=0, patience=15),
    "tusz":         dict(epochs=20, batch_size=512, eval_every_steps=200, patience=8),
    "chbmit":       dict(epochs=20, batch_size=512, eval_every_steps=200, patience=8),
    "sleepedf":     dict(epochs=60, batch_size=512, eval_every_steps=0, patience=15),
    # ISRUC is stored as sequences of 20 epochs; flatten_sequences turns ~5k
    # sequences into ~100k single epochs, so the step count is moderate.
    "isruc":        dict(epochs=60, batch_size=512, eval_every_steps=0, patience=15),
    "physionet_mi": dict(epochs=100, batch_size=512, eval_every_steps=0, patience=20),
    "bci_iv_2a":    dict(epochs=100, batch_size=512, eval_every_steps=0, patience=20),
    # FACED: 123 subjects x 28 videos x 3 windows ~= 10k windows total, small.
    "faced":        dict(epochs=100, batch_size=512, eval_every_steps=0, patience=20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A")
    ap.add_argument("--out", default="configs/experiments")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    written = []

    for ds in DATASETS:
        proto = yaml.safe_load(open(f"configs/datasets/{ds}.yaml"))
        sched = SCHEDULE[ds]

        for model in GROUP_A:
            cfg = {
                "name": f"{ds}-{model}",
                "group": args.group,
                "model": model,
                "dataset": ds,
                "data_root": os.path.join(PROC_ROOT, ds),
                # straight from the frozen protocol
                "num_classes": proto["num_classes"],
                "sample_rate": proto["sample_rate"],
                "loss": proto["loss"],
                # schedule
                "batch_size": batch_for(ds, sched["batch_size"]),
                "num_workers": 16,
                "epochs": sched["epochs"],
                "optimizer": OPTIMIZER,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": SCHEDULER,
                "grad_clip": None,          # BIOT does not clip
                "patience": sched["patience"],
                "eval_every_steps": sched["eval_every_steps"],
                "seed": 0,
                "device": "cuda",
            }
            if proto.get("label_smoothing") is not None:
                cfg["label_smoothing"] = proto["label_smoothing"]
            for k in ("focal_alpha", "focal_gamma"):
                if proto.get(k) is not None:
                    cfg[k] = proto[k]

            # ISRUC stores (n_seq, 20, C, T); these models take one epoch at a time
            if proto.get("sequence", {}).get("enabled"):
                cfg["flatten_sequences"] = True

            # STFT models (ContraWR / CNN-Transformer / FFCL): BIOT uses
            # token_size=200 with steps=20, i.e. a 1 s FFT window and a 10-sample
            # hop, on 5-10 s windows at 200 Hz.
            #
            # Two things must follow the data rather than stay fixed:
            #
            #  1. token_size = one second of signal, so it tracks the sample rate
            #     (Sleep-EDF is 100 Hz).
            #  2. These three models pool their spectrogram by 4x four times and
            #     assume it collapses to 1x1, so the frame count must not exceed
            #     4^4 = 256. At the default hop a 30 s epoch (Sleep-EDF, ISRUC)
            #     produces ~601 frames and the classifier's matmul fails on the
            #     3 frames left over. The hop is widened just enough to get back
            #     under the limit, and left alone whenever the default already
            #     fits -- so every 5-10 s dataset keeps BIOT's exact settings.
            #
            # Model-side only: the data is untouched and all five models on a
            # given dataset still see identical input.
            if model in ("contrawr", "ffcl", "cnn_transformer"):
                token_size = proto["sample_rate"]           # 1 s window
                seq_len = int(proto["sample_rate"] * proto.get(
                    "window_sec", proto.get("trial_sec", 10)))
                steps = 20                                   # BIOT default
                max_frames = 256                             # 4 pooling stages
                while steps > 1 and seq_len / max(token_size // steps, 1) > max_frames:
                    steps -= 1
                kwargs = {}
                if token_size != 200:
                    kwargs["token_size"] = token_size
                if steps != 20:
                    kwargs["steps"] = steps
                if kwargs:
                    cfg["model_kwargs"] = kwargs

            path = os.path.join(args.out, f"{ds}_{model}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)

    print(f"wrote {len(written)} configs for group {args.group}")
    for ds in DATASETS:
        n = sum(1 for p in written if f"/{ds}_" in p)
        print(f"  {ds:<14} {n}")


if __name__ == "__main__":
    main()
