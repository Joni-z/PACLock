"""Training configs on the PAC-methodology protocol: PACLock, plus its control.

    python -m scripts.gen_configs_pac [--out configs/experiments]

Derived from the frozen-protocol configs by redirecting ``data_root`` and
nothing else, so a PAC-protocol run differs from its frozen counterpart in the
filtering of the input and in no other respect -- same architecture, same
recipe, same seeds, same splits.

Two models are emitted, and the second is the point of the exercise.

``paclock``
    The model the protocol is motivated by. Its sinc filterbank spans 1-98 Hz;
    the frozen protocol's 75 Hz ceiling and 60 Hz notch leave three of its eight
    learnable bands dead or punctured.

``cbramod`` -- **the control**
    The strongest baseline in the frozen-protocol table, run on exactly the same
    PAC-protocol data. Without it, "PACLock improves when we change the
    preprocessing" is indistinguishable from "this preprocessing is simply
    better data". With it the two are separable:

      * CBraMod gains little  -> the notch cost is specific to the model that
        has a learnable filterbank, which is the claim
      * CBraMod also gains    -> the notch hurts everything; our relative
        standing is unchanged and the finding is still worth reporting
      * CBraMod loses         -> the protocol helps only filterbank models

    Every branch is reportable, which is what makes it a control rather than a
    gamble. This is the same parameter-matched-control logic the paper already
    applies inside the model (§6.2), moved onto the preprocessing axis.

FACED is carried on the frozen data: PROTOCOLS.md sec.8 forbids re-filtering the
officially pre-processed release, so no PAC variant of it exists. Its configs are
still emitted, pointing at the frozen copy, and flagged -- silently dropping the
corpus would leave a hole in the table that looks like a failure.
"""

from __future__ import annotations

from paclock_bench.paths import processed
import argparse
import os

import yaml

PROC_PAC = processed("processed_pac")
PROC_FROZEN = processed("processed")

DATASETS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
            "physionet_mi", "bci_iv_2a", "faced"]

# corpora with no PAC variant -- see module docstring
FROZEN_ONLY = {"faced"}

# (source config suffix, output suffix)
MODELS = [("paclock_full", "paclock_pac"),
          ("cbramod_pretrained", "cbramod_pac")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="configs/experiments")
    ap.add_argument("--out", default="configs/experiments")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = []
    for ds in DATASETS:
        for src_suffix, out_suffix in MODELS:
            src = os.path.join(args.src, f"{ds}_{src_suffix}.yaml")
            if not os.path.exists(src):
                print(f"  {ds}/{src_suffix}: missing {src}, skipped")
                continue
            cfg = yaml.safe_load(open(src))

            if ds in FROZEN_ONLY:
                cfg["data_root"] = os.path.join(PROC_FROZEN, ds)
                cfg["_pac_note"] = ("no PAC variant: officially pre-processed "
                                    "upstream, PROTOCOLS.md sec.8 forbids "
                                    "re-filtering. Reads the frozen copy.")
            else:
                cfg["data_root"] = os.path.join(PROC_PAC, ds)
                cfg["_pac_note"] = ("PAC-methodology protocol: 0.5 Hz high-pass, "
                                    "no notch. Identical to the frozen config in "
                                    "every other respect.")
            cfg["name"] = f"{ds}-{out_suffix}"
            cfg["group"] = "D"          # protocol-variant rows, not the main table

            path = os.path.join(args.out, f"{ds}_{out_suffix}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            written.append(path)

    print(f"wrote {len(written)} configs")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
