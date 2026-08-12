"""Derive the PAC-methodology preprocessing configs from the frozen ones.

    python -m scripts.make_pac_protocol [--out configs/datasets_pac]

Why a second protocol exists at all
-----------------------------------
Hard rule 2 already says every model runs its *own* repo's preprocessing, and
five of them do: BIOT reads unfiltered 200 Hz, LaBraM reads 0.1-75 Hz with a
**50 Hz** notch on 23 unipolar channels, CBraMod reads the frozen 0.3-75 Hz with
a **60 Hz** notch. Two of the baselines already see spectral holes in different
places. PACLock getting a protocol of its own is the sixth application of that
rule, not an exception to it.

Why *this* protocol
-------------------
The frozen protocol's 60 Hz notch is not merely unhelpful to PACLock, it is
methodologically wrong for any paper that measures phase-amplitude coupling.
Kramer, Tort & Kopell (2008), "Sharp edge artifacts and spurious coupling in EEG
frequency comodulation measures" -- already cited in the paper's Limitations --
shows that sharp spectral edges *generate* spurious PAC. A notch filter is such
an edge, sitting inside the gamma band that carries the amplitude term.

The mechanism is measurable here rather than merely argued. PACLock's sinc
filterbank initialises eight bands linearly across 1-98 Hz at 200 Hz:

    [1-13] [13-25] [25-37] [37-49] [49-61] [61-74] [74-86] [86-98]
                                      ^60 Hz notch^   ^--- above 75 Hz cutoff ---^

so the frozen protocol delivers dead or punctured signal to three of the eight
learnable bands, and they are the fast bands that carry the amplitude envelope.
No baseline has a learnable filterbank, so none of them pays this cost.

What changes, and what deliberately does not
--------------------------------------------
Exactly one axis moves: the filtering.

    band  [0.3, 75.0] -> hp 0.5      high-pass only; removes DC drift, keeps
                                     everything up to Nyquist
    notch 60.0 (or 50.0) -> None     the Kramer et al. objection

Montage, window length, stride, label rule, split, and the /100 normalisation
are copied unchanged. That is what makes "frozen vs PAC" a clean single-variable
contrast rather than two unrelated pipelines, and it is why the sensitivity
analysis it feeds can be read as an effect of the notch specifically.

Datasets whose frozen protocol already has no notch (Sleep-EDF) still get the
low-pass removed, because a 35 Hz ceiling also truncates the filterbank -- but
for Sleep-EDF that band is empty of PAC anyway, which is precisely why its
frozen-protocol result already matched the reference.
"""

from __future__ import annotations

import argparse
from paclock_bench.paths import processed
import os
import shutil

import yaml

SRC = "configs/datasets"
PROC_PAC = processed("processed_pac")

DATASETS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
            "physionet_mi", "bci_iv_2a", "faced"]

# FACED ships officially pre-processed data (0.05-47 Hz, ICA, average reference)
# and PROTOCOLS.md sec.8 states we must not filter it again. Re-filtering would
# also be pointless: the 47 Hz ceiling is baked into the released .pkl and we
# cannot undo it. So FACED carries the frozen settings through unchanged and is
# flagged, rather than being silently given a protocol it cannot honour.
NO_REFILTER = {"faced"}

HIGH_PASS_HZ = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="configs/datasets_pac")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for ds in DATASETS:
        src = os.path.join(args.src, f"{ds}.yaml")
        if not os.path.exists(src):
            print(f"  {ds}: no frozen config at {src}, skipped")
            continue
        cfg = yaml.safe_load(open(src))

        cfg["out_dir"] = os.path.join(PROC_PAC, ds)
        if ds in NO_REFILTER:
            cfg["_pac_note"] = ("officially pre-processed upstream (0.05-47 Hz, "
                                "ICA, average reference); PROTOCOLS.md forbids "
                                "re-filtering, so the frozen settings stand")
        else:
            old_band, old_notch = cfg.get("band"), cfg.get("notch")
            cfg.pop("band", None)
            cfg["hp"] = HIGH_PASS_HZ
            cfg["notch"] = None
            cfg["_pac_note"] = (
                f"PAC-methodology variant: band {old_band} -> high-pass "
                f"{HIGH_PASS_HZ} Hz, notch {old_notch} -> none "
                f"(Kramer et al. 2008: sharp spectral edges generate spurious "
                f"phase-amplitude coupling). Everything else identical to the "
                f"frozen protocol, so the contrast isolates the filtering.")

        dst = os.path.join(args.out, f"{ds}.yaml")
        with open(dst, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"  {ds}: hp={cfg.get('hp')} notch={cfg.get('notch')} -> {dst}")


if __name__ == "__main__":
    main()
