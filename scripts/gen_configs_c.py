"""Derive the group-C from-scratch configs from their group-B counterparts.

    python -m scripts.gen_configs_c [--out configs/experiments]

Group C is labelled "FM · 同 pipeline scratch" in the workbook, and that label is
the specification: a scratch row must differ from the pretrained row of the same
model in the weights and in nothing else. So rather than write a second recipe,
each config is read back from the group-B file and exactly one key is flipped:

    biot    checkpoint: prest16  -> null
    labram  pretrained: true     -> false

Everything else -- data_root, montage, window, optimiser, schedule, batch,
patience, loss, loader_divisor -- is carried through untouched. This is checked
against the three cells that already exist: diffing tuab_biot_prest16.yaml
against the hand-written tuab_biot_scratch.yaml shows only name, group and
checkpoint, and the same for tuev_labram_*. The generator therefore reproduces
the existing configs rather than introducing a new convention alongside them.

Deriving instead of re-specifying also settles the preprocessing question by
construction. Hard rule 2 says every model reads its own repo's preprocessing;
whatever the group-B config resolved to is by definition that, so the scratch row
inherits it and cannot drift from the pretrained row it is meant to isolate.
"""

from __future__ import annotations

import argparse
import os

import yaml

DATASETS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
            "physionet_mi", "faced", "bci_iv_2a"]

# out_suffix -> (src_suffix, key, scratch_value)
MODELS = {
    "biot_scratch":   ("biot_prest16",      "checkpoint", None),
    "labram_scratch": ("labram_pretrained", "pretrained", False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="configs/experiments")
    ap.add_argument("--out", default="configs/experiments")
    ap.add_argument("--only", nargs="*", help="limit to these datasets")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite scratch configs that already exist")
    args = ap.parse_args()

    datasets = args.only or DATASETS
    written, skipped = [], []
    for ds in datasets:
        for out_suffix, (src_suffix, key, val) in MODELS.items():
            src = os.path.join(args.src, f"{ds}_{src_suffix}.yaml")
            dst = os.path.join(args.out, f"{ds}_{out_suffix}.yaml")
            if not os.path.exists(src):
                skipped.append(f"{ds}/{out_suffix}: no group-B source {src}")
                continue
            if os.path.exists(dst) and not args.overwrite:
                skipped.append(f"{ds}/{out_suffix}: exists")
                continue

            cfg = yaml.safe_load(open(src))
            # data_root is inherited, so a corpus whose native preprocessing was
            # deleted must be caught here rather than at 3am inside the job.
            root = cfg.get("data_root")
            if root and not os.path.isdir(root):
                skipped.append(f"{ds}/{out_suffix}: data_root missing ({root})")
                continue

            cfg["name"] = f"{ds}-{out_suffix}"
            cfg["group"] = "C"
            cfg[key] = val

            with open(dst, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(dst)

    print(f"wrote {len(written)} configs")
    for p in written:
        print("  +", os.path.basename(p))
    if skipped:
        print(f"skipped {len(skipped)}")
        for s in skipped:
            print("  -", s)


if __name__ == "__main__":
    main()
