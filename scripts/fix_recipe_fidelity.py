"""Bring the configs back onto each model's own published recipe (hard rule 2).

    python -m scripts.fix_recipe_fidelity [--apply]

Hard rule 2 says every model runs the recipe from its own repository. Three
settings had drifted off it. None was a judgement call -- each is a default
sitting in the vendored source that our config contradicts.

CBraMod -- gradient clipping
    ``vendor/cbramod/finetune_main.py`` declares ``--clip_value default=1`` and
    ``finetune_trainer.py`` calls ``clip_grad_norm_(model.parameters(),
    clip_value)`` on every step. Every one of our cbramod configs carries
    ``grad_clip: null``, so no clipping happened at all.

    This is not cosmetic. Pretrained CBraMod starts in a good region and
    survives without it; CBraMod from scratch on CHB-MIT does not -- all three
    seeds collapse onto the majority class of a 0.9%-positive corpus and finish
    at test AUROC 0.5000, which is the one genuinely dead cell in group C.

CBraMod -- label smoothing on TUEV
    ``--label_smoothing default=0.1``. Our TUEV configs set it to 0.0.

LaBraM -- label smoothing on the multi-class corpora
    ``run_class_finetuning.py`` declares ``--smoothing default=0.1`` and
    selects ``LabelSmoothingCrossEntropy(0.1)`` whenever ``nb_classes != 1``.
    Our TUEV configs set 0.0 and the other multi-class corpora omit the key,
    which ``losses.py`` reads as 0.0.

    The binary corpora are deliberately left alone: LaBraM takes the
    ``nb_classes == 1 -> BCEWithLogitsLoss`` branch there, which never consults
    smoothing, so absent is correct for TUAB, TUSZ and CHB-MIT.

What is deliberately NOT changed
    BIOT and EEGPT both construct a plain ``CrossEntropyLoss()`` with no
    smoothing, so their 0.0 is faithful. LaBraM's ``layer_decay: 0.65`` looks
    wrong against the argparse default of 0.9, but 0.65 is what LaBraM's own
    downstream script passes, and the script wins over the default. CHB-MIT's
    focal loss is a protocol-level choice applied to every model alike
    (PROTOCOLS.md appendix A), not a per-model recipe, so it stays.

Every one of these corrections helps a *baseline*, not us. They are applied
because the comparison has to be against each model at its best.
"""

from __future__ import annotations

import argparse
import glob
import os

import yaml

# corpora where LaBraM/CBraMod take a multi-class cross-entropy path
MULTICLASS = {"tuev", "sleepedf", "isruc", "physionet_mi", "faced", "bci_iv_2a"}

OFFICIAL_SMOOTHING = 0.1
CBRAMOD_CLIP = 1.0


def dataset_of(cfg: dict, path: str) -> str:
    return cfg.get("dataset") or os.path.basename(path).split("_")[0]


def fixes_for(cfg: dict, path: str) -> dict:
    """Settings this config should carry but does not."""
    model = cfg.get("model")
    ds = dataset_of(cfg, path)
    out: dict = {}

    if model == "cbramod":
        if cfg.get("grad_clip") != CBRAMOD_CLIP:
            out["grad_clip"] = CBRAMOD_CLIP
        if ds in MULTICLASS and cfg.get("label_smoothing") != OFFICIAL_SMOOTHING:
            out["label_smoothing"] = OFFICIAL_SMOOTHING

    if model == "labram":
        if ds in MULTICLASS and cfg.get("label_smoothing") != OFFICIAL_SMOOTHING:
            out["label_smoothing"] = OFFICIAL_SMOOTHING

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs/experiments")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    changed = []
    for path in sorted(glob.glob(os.path.join(args.configs, "*.yaml"))):
        cfg = yaml.safe_load(open(path))
        if not isinstance(cfg, dict):
            continue
        # probe configs are throwaway diagnostics, not matrix cells
        if "_probe" in os.path.basename(path):
            continue
        fix = fixes_for(cfg, path)
        if not fix:
            continue
        changed.append((os.path.basename(path), cfg.get("group"), fix))
        if args.apply:
            cfg.update(fix)
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    by_group: dict[str, int] = {}
    for _, group, _ in changed:
        by_group[str(group)] = by_group.get(str(group), 0) + 1

    print(f"{len(changed)} configs off-recipe "
          + ", ".join(f"group {g}: {n}" for g, n in sorted(by_group.items())))
    for name, group, fix in changed:
        print(f"  [{group}] {name:38s} {fix}")
    if not args.apply:
        print("\ndry run -- pass --apply to write")


if __name__ == "__main__":
    main()
