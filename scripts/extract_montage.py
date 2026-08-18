"""Extract electrode coordinates for the corpora that currently fall back to
SpatialPE's learned index embedding.

Why this matters: in this benchmark every corpus PACLock is competitive on
feeds `spatial_pe: xyz` (real electrode geometry), and every corpus it loses
badly on feeds `spatial_pe: index` (a learned embedding per electrode slot).
The space axis is self-attention over electrodes, which is permutation-
equivariant -- without coordinates the only way it can learn that C3 and C4
straddle the sensorimotor strip is from the training data itself, and these
corpora have 2k-7k training windows to teach it 22-64 electrode identities.

The existing montage table is 6-D because the TUH corpora are bipolar (a
channel is an electrode *pair*). These four are referential, so a channel is
one electrode and its coordinate is 3-D. SpatialPE builds its MLP from
`coords.shape[1]`, so a 3-D table needs no code change -- verified by reading
SpatialPE.__init__.

Emits a Python literal block to paste into models/paclock/montage.py rather
than writing it directly, so the values are reviewable in the diff and the
runtime keeps its "no mne dependency" property.
"""
import sys

import mne
import numpy as np

mne.set_log_level("ERROR")

# Channel order must match what preprocessing/*.py actually produced, so it is
# read back from the raw files rather than assumed.
SOURCES = {
    "physionet_mi": ("$PACLOCK_DATA/physionet-mi/S001/S001R04.edf", "edf", None),
    "bci_iv_2a": (None, "moabb", None),   # filled in below
    "isruc": (None, "manual", ["F3", "C3", "O1", "F4", "C4", "O2"]),
}

STD = mne.channels.make_standard_montage("standard_1005")
POS = {k.lower(): v for k, v in STD.get_positions()["ch_pos"].items()}
RADIUS = 0.1     # same normalisation the existing 6-D table used


def lookup(name):
    """Map a dataset's channel label to standard_1005, tolerating the dots and
    case variations EDF headers carry (BCI2000 writes 'Fc5.', 'Af7.')."""
    n = name.strip().rstrip(".").lower()
    n = n.replace("fp", "fp")            # keep, standard_1005 uses Fp1 not FP1
    aliases = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8",
               "fpz": "fpz", "iz": "iz"}
    n = aliases.get(n, n)
    if n in POS:
        return POS[n]
    return None


def emit(tag, names):
    rows, missing = [], []
    for nm in names:
        p = lookup(nm)
        if p is None:
            missing.append(nm)
            rows.append(None)
        else:
            rows.append([round(float(x) / RADIUS, 4) for x in p])
    if missing:
        print("  !! unmapped channels for %s: %s" % (tag, missing), file=sys.stderr)
        return None
    print("_%s = [" % tag.upper())
    for nm, r in zip(names, rows):
        print("    [%s],   # %s" % (", ".join("%.4f" % v for v in r), nm))
    print("]")
    print()
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "physionet_mi":
        import glob
        from paclock_bench.paths import expand
        cands = sorted(glob.glob(expand("$PACLOCK_DATA/physionet-mi/S001/S001R*.edf")))
        raw = mne.io.read_raw_edf(cands[0], preload=False)
        emit("pmi_64", raw.ch_names)
    elif which == "isruc":
        emit("isruc_6", ["F3", "C3", "O1", "F4", "C4", "O2"])
    elif which == "bci_iv_2a":
        # MOABB/GDF order for BCI-IV-2a's 22 EEG electrodes
        names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                 "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                 "CP3", "CP1", "CPz", "CP2", "CP4",
                 "P1", "Pz", "P2", "POz"]
        emit("bci22", names)
    elif which == "faced":
        # FACED official 32-channel order (10-20)
        names = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
                 "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
                 "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8",
                 "PO3", "PO4", "Oz", "O1", "O2", "A2", "A1"]
        emit("faced32", names)
