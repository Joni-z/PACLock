"""Electrode coordinates for montage-agnostic spatial positional encoding
(AGENT.md sec. 13.23 A / 13.4).

Every dataset in this project feeds a fixed **bipolar** montage (a channel is an
electrode *pair*, e.g. FP1-F7), so a channel is encoded by the concatenated xyz
of its two 10-20 endpoints -> a 6-D coordinate. Coordinates were extracted once
from `mne.channels.make_standard_montage("standard_1020")` (TUH's legacy T3/T4/
T5/T6 mapped to modern T7/T8/P7/P8) and normalised by a ~0.1 m head radius so
they sit in ~[-1.2, 1.2]; hardcoded here so runtime has no mne dependency.

`coords_for(dataset)` returns an (n_channels, 6) float array, or None for an
unknown montage (SpatialPE then falls back to its learned index embedding, so
existing behaviour is unchanged when coordinates are absent).
"""

from __future__ import annotations

import numpy as np

# 16-channel bipolar montage shared by TUAB/TUEV/TUSZ/CHB-MIT (BIOT recipe,
# scripts/preprocess_tuab.py). Order matches the preprocessing channel order.
_BIPOLAR_16 = [
    [-0.2944, 0.8392, -0.0699, -0.7026, 0.4247, -0.1142],   # FP1-F7
    [-0.7026, 0.4247, -0.1142, -0.8416, -0.1602, -0.0935],  # F7-T3
    [-0.8416, -0.1602, -0.0935, -0.7243, -0.7345, -0.0249], # T3-T5
    [-0.7243, -0.7345, -0.0249, -0.2941, -1.1245, 0.0884],  # T5-O1
    [0.2987, 0.849, -0.0708, 0.7304, 0.4442, -0.12],        # FP2-F8
    [0.7304, 0.4442, -0.12, 0.8508, -0.1502, -0.0949],      # F8-T4
    [0.8508, -0.1502, -0.0949, 0.7306, -0.7307, -0.0254],   # T4-T6
    [0.7306, -0.7307, -0.0254, 0.2984, -1.1216, 0.088],     # T6-O2
    [-0.2944, 0.8392, -0.0699, -0.5024, 0.5311, 0.4219],    # FP1-F3
    [-0.5024, 0.5311, 0.4219, -0.6536, -0.1163, 0.6436],    # F3-C3
    [-0.6536, -0.1163, 0.6436, -0.5301, -0.7879, 0.5594],   # C3-P3
    [-0.5301, -0.7879, 0.5594, -0.2941, -1.1245, 0.0884],   # P3-O1
    [0.2987, 0.849, -0.0708, 0.5184, 0.543, 0.4081],        # FP2-F4
    [0.5184, 0.543, 0.4081, 0.6712, -0.109, 0.6358],        # F4-C4
    [0.6712, -0.109, 0.6358, 0.5567, -0.7856, 0.5656],      # C4-P4
    [0.5567, -0.7856, 0.5656, 0.2984, -1.1216, 0.088],      # P4-O2
]

# Sleep-EDF Cassette 2-channel montage (Fpz-Cz, Pz-Oz).
_SLEEP_2 = [
    [0.0011, 0.8825, -0.0171, 0.004, -0.0917, 1.0024],      # Fpz-Cz
    [0.0032, -0.8111, 0.8261, 0.0011, -1.1489, 0.1466],     # Pz-Oz
]


# ---- referential (non-bipolar) montages ---------------------------------- #
# The tables above are 6-D because the TUH corpora are bipolar: a channel is an
# electrode PAIR, so it is encoded by both endpoints. The corpora below are
# referential -- one electrode per channel -- so their coordinates are 3-D.
# SpatialPE builds its projection from coords.shape[1], so both widths coexist
# without a code change; what must not happen is mixing widths inside one
# dataset.
#
# These exist because of a near-perfect split in this benchmark: every corpus
# PACLock is competitive on fed real coordinates, and every corpus it lost
# badly on was falling back to SpatialPE's learned per-electrode embedding --
# asking 2k-7k training windows to teach the model 22-64 electrode identities
# from scratch, on exactly the paradigms (motor imagery, emotion) where the
# discriminative signal is spatial. See docs/TIER3.md.

# PhysioNet-MI: 64-electrode BCI2000 referential montage (EDF header order).
_PMI_64 = [
    [-0.7721, 0.1864, 0.2446],   # Fc5
    [-0.6018, 0.2272, 0.5554],   # Fc3
    [-0.3406, 0.2601, 0.7999],   # Fc1
    [0.0038, 0.2739, 0.8867],   # Fcz
    [0.3478, 0.2644, 0.7881],   # Fc2
    [0.6229, 0.2372, 0.5563],   # Fc4
    [0.7953, 0.1994, 0.2444],   # Fc6
    [-0.8028, -0.1376, 0.2916],   # C5
    [-0.6536, -0.1163, 0.6436],   # C3
    [-0.3616, -0.0998, 0.8975],   # C1
    [0.0040, -0.0917, 1.0024],   # Cz
    [0.3767, -0.0962, 0.8841],   # C2
    [0.6712, -0.1090, 0.6358],   # C4
    [0.8346, -0.1278, 0.2921],   # C6
    [-0.7959, -0.4655, 0.3095],   # Cp5
    [-0.6356, -0.4701, 0.6562],   # Cp3
    [-0.3551, -0.4729, 0.9131],   # Cp1
    [0.0039, -0.4732, 0.9943],   # Cpz
    [0.3838, -0.4707, 0.9069],   # Cp2
    [0.6661, -0.4664, 0.6558],   # Cp4
    [0.8332, -0.4610, 0.3121],   # Cp6
    [-0.2944, 0.8392, -0.0699],   # Fp1
    [0.0011, 0.8825, -0.0171],   # Fpz
    [0.2987, 0.8490, -0.0708],   # Fp2
    [-0.5484, 0.6857, -0.1059],   # Af7
    [-0.3370, 0.7684, 0.2123],   # Af3
    [0.0023, 0.8077, 0.3542],   # Afz
    [0.3571, 0.7773, 0.2196],   # Af4
    [0.5574, 0.6966, -0.1076],   # Af8
    [-0.7026, 0.4247, -0.1142],   # F7
    [-0.6447, 0.4804, 0.1692],   # F5
    [-0.5024, 0.5311, 0.4219],   # F3
    [-0.2750, 0.5693, 0.6034],   # F1
    [0.0031, 0.5851, 0.6646],   # Fz
    [0.2951, 0.5760, 0.5954],   # F2
    [0.5184, 0.5430, 0.4081],   # F4
    [0.6791, 0.4983, 0.1637],   # F6
    [0.7304, 0.4442, -0.1200],   # F8
    [-0.8077, 0.1412, -0.1113],   # Ft7
    [0.8182, 0.1542, -0.1133],   # Ft8
    [-0.8416, -0.1602, -0.0935],   # T7
    [0.8508, -0.1502, -0.0949],   # T8
    [-0.8589, -0.1583, -0.4828],   # T9
    [0.8556, -0.1636, -0.4827],   # T10
    [-0.8483, -0.4602, -0.0706],   # Tp7
    [0.8555, -0.4555, -0.0713],   # Tp8
    [-0.7243, -0.7345, -0.0249],   # P7
    [-0.6727, -0.7629, 0.2838],   # P5
    [-0.5301, -0.7879, 0.5594],   # P3
    [-0.2862, -0.8052, 0.7544],   # P1
    [0.0032, -0.8111, 0.8261],   # Pz
    [0.3192, -0.8049, 0.7672],   # P2
    [0.5567, -0.7856, 0.5656],   # P4
    [0.6789, -0.7590, 0.2809],   # P6
    [0.7306, -0.7307, -0.0254],   # P8
    [-0.5484, -0.9753, 0.0279],   # Po7
    [-0.3651, -1.0085, 0.3717],   # Po3
    [0.0022, -1.0218, 0.5061],   # Poz
    [0.3678, -1.0085, 0.3640],   # Po4
    [0.5567, -0.9763, 0.0273],   # Po8
    [-0.2941, -1.1245, 0.0884],   # O1
    [0.0011, -1.1489, 0.1466],   # Oz
    [0.2984, -1.1216, 0.0880],   # O2
    [0.0000, -1.1856, -0.2308],   # Iz
]

# BCI-IV-2a: 22 EEG electrodes in GDF order (3 EOG dropped upstream).
_BCI22 = [
    [0.0031, 0.5851, 0.6646],   # Fz
    [-0.6018, 0.2272, 0.5554],   # FC3
    [-0.3406, 0.2601, 0.7999],   # FC1
    [0.0038, 0.2739, 0.8867],   # FCz
    [0.3478, 0.2644, 0.7881],   # FC2
    [0.6229, 0.2372, 0.5563],   # FC4
    [-0.8028, -0.1376, 0.2916],   # C5
    [-0.6536, -0.1163, 0.6436],   # C3
    [-0.3616, -0.0998, 0.8975],   # C1
    [0.0040, -0.0917, 1.0024],   # Cz
    [0.3767, -0.0962, 0.8841],   # C2
    [0.6712, -0.1090, 0.6358],   # C4
    [0.8346, -0.1278, 0.2921],   # C6
    [-0.6356, -0.4701, 0.6562],   # CP3
    [-0.3551, -0.4729, 0.9131],   # CP1
    [0.0039, -0.4732, 0.9943],   # CPz
    [0.3838, -0.4707, 0.9069],   # CP2
    [0.6661, -0.4664, 0.6558],   # CP4
    [-0.2862, -0.8052, 0.7544],   # P1
    [0.0032, -0.8111, 0.8261],   # Pz
    [0.3192, -0.8049, 0.7672],   # P2
    [0.0022, -1.0218, 0.5061],   # POz
]

# ISRUC: 6 channels, each referenced to the contralateral mastoid; the
# mastoid is a near-common reference so the active electrode locates the channel.
_ISRUC_6 = [
    [-0.5024, 0.5311, 0.4219],   # F3
    [-0.6536, -0.1163, 0.6436],   # C3
    [-0.2941, -1.1245, 0.0884],   # O1
    [0.5184, 0.5430, 0.4081],   # F4
    [0.6712, -0.1090, 0.6358],   # C4
    [0.2984, -1.1216, 0.0880],   # O2
]


# Mumtaz2016 / EEGMat: the classic 19-electrode referential 10-20 set, shared
# canonical order (configs/datasets/{mumtaz,eegmat}.yaml reorder to match).
_MONO_19 = [
    [-0.2944, 0.8392, -0.0699],   # Fp1
    [0.2987, 0.8490, -0.0708],   # Fp2
    [-0.5024, 0.5311, 0.4219],   # F3
    [0.5184, 0.5430, 0.4081],   # F4
    [-0.6536, -0.1163, 0.6436],   # C3
    [0.6712, -0.1090, 0.6358],   # C4
    [-0.5301, -0.7879, 0.5594],   # P3
    [0.5567, -0.7856, 0.5656],   # P4
    [-0.2941, -1.1245, 0.0884],   # O1
    [0.2984, -1.1216, 0.0880],   # O2
    [-0.7026, 0.4247, -0.1142],   # F7
    [0.7304, 0.4442, -0.1200],   # F8
    [-0.8416, -0.1602, -0.0935],   # T3 (T7)
    [0.8508, -0.1502, -0.0949],   # T4 (T8)
    [-0.7243, -0.7345, -0.0249],   # T5 (P7)
    [0.7306, -0.7307, -0.0254],   # T6 (P8)
    [0.0031, 0.5851, 0.6646],   # Fz
    [0.0040, -0.0917, 1.0024],   # Cz
    [0.0032, -0.8111, 0.8261],   # Pz
]


_BY_DATASET = {
    "tuab": _BIPOLAR_16,
    "tuev": _BIPOLAR_16,
    "tusz": _BIPOLAR_16,
    "chbmit": _BIPOLAR_16,
    "tuep": _BIPOLAR_16,
    "sleepedf": _SLEEP_2,
    "sleep": _SLEEP_2,
    "physionet_mi": _PMI_64,
    "bci_iv_2a": _BCI22,
    "isruc": _ISRUC_6,
    "mumtaz": _MONO_19,
    "eegmat": _MONO_19,
    # IIIC: order verified against runSPaRCNet.py = the double-banana order
    "iiic": _BIPOLAR_16,
    "caueeg": _MONO_19,
    "siena": _MONO_19,
}

COORD_DIM = 6      # bipolar montages; referential ones are 3 (see above)


def coords_for(dataset: str | None):
    """(n_channels, 6) coordinate array for a known montage, else None."""
    if dataset is None:
        return None
    table = _BY_DATASET.get(str(dataset).lower())
    if table is None:
        return None
    return np.asarray(table, dtype=np.float32)
