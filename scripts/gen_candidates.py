"""Architecture candidates for PACLock, round 1.

    python -m scripts.gen_candidates [--out configs/_cand] [--amp] [--batch N]

Why these candidates and not others
-----------------------------------
The upstream log already swept the backbone on TUEV and nothing beat the base
(AGENT.md 13.40-A): hz BandPE 0.4544, n_bands=16 0.4435, d_model=192 0.4181,
freq_mixer=coupling 0.4175, depth=8 0.3632, against base 0.4679. 13.41 rejected
the MI-guided frequency mixer, 13.44a retracted the objective and BandPE
directions as unstable, and 13.46 measured a 5.23x scale-up as a *loss*
(-0.011 measured, -0.018 raw). Width, depth, band count, mixer and both
positional encodings are therefore spent.

Two things follow from that pattern rather than being contradicted by it.

**Every capacity increase hurt.** Wider, deeper, more bands, 5x params — all
negative. That is the signature of a model limited by data, not by capacity. So
the first direction is regularisation and augmentation, which have never been
varied: `augmentations: []` is empty in every config. It was emptied
deliberately, to match BIOT for an architecture-vs-architecture comparison
(13.32 "BIOT trains with NO augmentation"), which was correct for that question
and is not correct for this one -- under hard rule 2 each model runs its own
recipe, and the upstream reference configs do carry
`jitter0.1 / mask0.2 / channel0.1 / frequency0.2`.

**The readout was never varied.** Every result to date mean-pools uniformly over
all C*n_bands*P tokens -- 1280 on TUEV, 2560 on FACED -- into a single linear
layer. For a model whose claim is that particular band pairs at particular
electrodes carry the signal, averaging every band, electrode and time patch with
equal weight discards precisely the structure the PAC tokenizer builds. The
baselines that beat us carry far richer readouts: CBraMod's `all_patch_reps`
concatenates every patch representation, which is why its head grows to 9.85M on
Sleep-EDF and 56.25M on FACED.

Round 1 is deliberately config-only. Every candidate below is reachable through
existing knobs, so nothing here can break the model in a way that costs a sweep.
`head=band` and `head=attn` add 5.4K and 33K parameters against a 1.64M model,
which makes them parameter-matched controls rather than capacity changes.
"""

from __future__ import annotations

import argparse
import copy
import os

import yaml

# The upstream reference augmentation set (configs/pacint_sleepedf_*.yaml).
REFERENCE_AUG = ["jitter0.1", "mask0.2", "channel0.1", "frequency0.2"]

# name -> (top-level overrides, model_kwargs overrides)
CANDIDATES: dict[str, tuple[dict, dict]] = {
    # control: the current cell, so the sweep carries its own baseline under
    # identical batch/precision rather than being compared to an older run
    "base":        ({}, {}),

    # --- readout: the axis never varied -------------------------------------
    "head_band":   ({}, {"head": "band"}),
    "head_attn":   ({}, {"head": "attn"}),

    # --- regularisation: every capacity increase hurt, so try the other way --
    "aug":         ({}, {"augmentations": REFERENCE_AUG}),
    "drop03":      ({}, {"dropout": 0.3}),
    "aug_drop03":  ({}, {"augmentations": REFERENCE_AUG, "dropout": 0.3}),

    # --- temporal resolution of the patch grid ------------------------------
    # patch_len is the window the PAC statistic is estimated over AND the token
    # stride. 100 doubles the token count and halves the estimation window (at
    # 200 Hz, 0.5 s is only two cycles of a 4 Hz phase carrier, so this may cost
    # PAC quality to buy temporal detail); 400 is the opposite trade. BIOT, which
    # beats us on several corpora, tokenises at 200 with hop 100 -- 50% overlap,
    # which our tokenizer cannot express without changing how the PAC vector is
    # patched, so these two bracket it from either side instead.
    "patch100":    ({}, {"patch_len": 100}),
    "patch400":    ({}, {"patch_len": 400}),

    # --- heads: 13.40-A varied d_model and depth but never the head count ----
    "heads8":      ({}, {"n_heads": 8}),

    # --- the two most promising directions together -------------------------
    "band_aug":    ({}, {"head": "band", "augmentations": REFERENCE_AUG}),
}

DATASETS = ["tuev", "isruc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="configs/experiments")
    ap.add_argument("--out", default="configs/_cand")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast (frontend stays fp32)")
    ap.add_argument("--batch", type=int, default=0,
                    help="override batch size; 0 keeps the reference 32")
    ap.add_argument("--epochs", type=int, default=0,
                    help="override epochs; 0 keeps the cell's own")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = []
    for ds in args.datasets:
        src = os.path.join(args.src, f"{ds}_paclock_full.yaml")
        if not os.path.exists(src):
            print(f"  {ds}: no source config at {src}, skipped")
            continue
        base = yaml.safe_load(open(src))
        for tag, (top, mk) in CANDIDATES.items():
            cfg = copy.deepcopy(base)
            cfg.update(top)
            cfg["model_kwargs"] = {**cfg.get("model_kwargs", {}), **mk}
            if args.amp:
                cfg["amp"] = True
            if args.batch:
                cfg["batch_size"] = args.batch
            if args.epochs:
                cfg["epochs"] = args.epochs
            cfg["name"] = f"{ds}-cand_{tag}"
            cfg["group"] = "cand"
            cfg["seed"] = 0                     # single seed for screening
            path = os.path.join(args.out, f"{ds}_{tag}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)

    print(f"wrote {len(written)} candidate configs "
          f"({len(CANDIDATES)} arms x {len(args.datasets)} datasets)")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
