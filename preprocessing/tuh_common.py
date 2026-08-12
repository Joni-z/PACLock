"""TUH-family shared logic: bipolar montage construction from referential EDF.

The three TUH corpora in this benchmark (TUAB, TUEV, TUSZ) all ship referential
recordings and all three protocols ask for the same 16 bipolar derivations. The
reference suffix varies by montage family:

    01_tcp_ar     "EEG FP1-REF"   averaged reference
    02_tcp_le     "EEG FP1-LE"    linked-ear reference
    03_tcp_ar_a   "EEG FP1-REF"   averaged reference, reduced channel set

Rather than hard-coding one suffix, the suffix is detected per file. A recording
that cannot supply every one of the 16 pairs is excluded whole and logged --
never partially filled or zero-padded, which the TUSZ protocol requires
explicitly and which is the right behaviour for the others too.
"""

from __future__ import annotations

import os

import mne
import numpy as np

# The 16 bipolar pairs, in protocol order. TUH uses the old T3/T4/T5/T6 labels.
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]


class MissingChannels(Exception):
    """Raised when a recording cannot supply all 16 bipolar derivations."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"missing channels: {sorted(missing)}")


def detect_suffix(ch_names: list[str]) -> str:
    """Return the reference suffix used by this recording ('-REF' or '-LE')."""
    for suffix in ("-REF", "-LE"):
        if any(c.upper().endswith(suffix) for c in ch_names):
            return suffix
    raise MissingChannels(["<no -REF or -LE channels found>"])


def build_bipolar(raw: mne.io.BaseRaw) -> np.ndarray:
    """Referential EDF -> (16, T) bipolar array in volts.

    Raises MissingChannels if any electrode the montage needs is absent, so the
    caller can exclude and log the whole recording.
    """
    ch_names = raw.ch_names
    suffix = detect_suffix(ch_names)
    # map bare electrode name -> channel index, case-insensitively
    index: dict[str, int] = {}
    for i, name in enumerate(ch_names):
        up = name.upper()
        if not up.endswith(suffix):
            continue
        bare = up[: -len(suffix)].replace("EEG ", "").strip()
        index.setdefault(bare, i)

    needed = {e for pair in BIPOLAR_PAIRS for e in pair}
    missing = sorted(e for e in needed if e not in index)
    if missing:
        raise MissingChannels(missing)

    data = raw.get_data()
    out = np.empty((len(BIPOLAR_PAIRS), data.shape[1]), dtype=np.float64)
    for k, (a, b) in enumerate(BIPOLAR_PAIRS):
        out[k] = data[index[a]] - data[index[b]]
    return out


def load_bipolar_uv(edf_path: str) -> tuple[np.ndarray, float]:
    """Read a TUH EDF and return (16, T) in microvolts plus the sample rate.

    MNE returns volts; the protocols specify microvolt input before the
    divide-by-100 normalisation, so the conversion happens here once.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    sig = build_bipolar(raw) * 1e6
    del raw
    return sig, fs


def subject_of(path: str) -> str:
    """TUH filenames are ``<subject>_s###_t###.edf``; the subject is field 0."""
    return os.path.basename(path).split("_")[0]


def list_edfs(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".edf"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def sort_subject_split(subjects: list[str], ratio: float) -> tuple[list[str], list[str]]:
    """Sort subject IDs, take the first ``ratio`` as train and the rest as val.

    Deterministic by construction -- the protocol replaced BIOT's seeded
    ``np.random.shuffle`` with a sort precisely so the split does not depend on
    a RNG implementation.
    """
    ordered = sorted(set(subjects))
    cut = int(len(ordered) * ratio)
    return ordered[:cut], ordered[cut:]
