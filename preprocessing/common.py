"""Shared preprocessing primitives.

Every dataset script in this package builds on these. The protocol values
themselves live in ``configs/datasets/*.yaml`` and are transcribed from
``docs/PROTOCOLS.md``; nothing here hard-codes a frozen parameter.

Two invariants this module enforces:

1. **Filter then resample.** Anti-alias order matters: the band-pass low cut
   must be applied at the original rate before decimation, otherwise content
   above the new Nyquist folds back in. All datasets in this benchmark specify
   a low-pass at or below the target Nyquist, so this ordering is always safe.
2. **Manifest or it did not happen.** Every script emits a manifest recording
   split membership, per-class counts, discarded-tail counts and raw-file
   SHA256. The protocol requires the same manifest to be shared by every model,
   so the manifest -- not the npy -- is the unit of reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, resample_poly


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def bandpass(x: np.ndarray, fs: float, low: float, high: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis.

    ``sosfiltfilt`` rather than ``filtfilt``: at fs=250 with a 0.3 Hz low cut the
    normalised frequency is 2.4e-3, where a transfer-function-form filter loses
    numerical conditioning and can ring or blow up. SOS is stable there.
    """
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)          # guard: 75 Hz cut at fs=100 would be >= nyq
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def highpass(x: np.ndarray, fs: float, cut: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass (PhysioNet-MI has no low cut)."""
    sos = butter(order, cut / (fs / 2.0), btype="highpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, fs: float, freq: float, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch. Skipped when the line frequency is unreachable."""
    nyq = fs / 2.0
    if freq >= nyq:
        return x
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, x, axis=-1)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample along the last axis. No-op when rates already match."""
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from math import gcd
    fi, fo = int(round(fs_in)), int(round(fs_out))
    g = gcd(fi, fo)
    return resample_poly(x, fo // g, fi // g, axis=-1)


def preprocess_signal(x: np.ndarray, fs_in: float, *, fs_out: float,
                      band: tuple[float, float] | None = None,
                      hp: float | None = None,
                      notch_freq: float | None = None) -> np.ndarray:
    """Filter at the native rate, then resample. See module docstring, invariant 1."""
    x = np.asarray(x, dtype=np.float64)
    if band is not None:
        x = bandpass(x, fs_in, band[0], band[1])
    elif hp is not None:
        x = highpass(x, fs_in, hp)
    if notch_freq is not None:
        x = notch(x, fs_in, notch_freq)
    x = resample_to(x, fs_in, fs_out)
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def window_signal(x: np.ndarray, win: int, stride: int) -> tuple[np.ndarray, int]:
    """Cut ``x`` (C, T) into (N, C, win). Returns the windows and the number of
    tail samples discarded -- the protocols require the discarded-tail count in
    the manifest, so it is returned rather than silently dropped."""
    C, T = x.shape
    if T < win:
        return np.empty((0, C, win), dtype=x.dtype), T
    n = (T - win) // stride + 1
    idx = np.arange(n) * stride
    out = np.stack([x[:, i:i + win] for i in idx], axis=0)
    tail = T - (idx[-1] + win)
    return out, int(tail)


def intervals_overlap_labels(n_windows: int, win: int, stride: int,
                             intervals: list[tuple[float, float]],
                             fs: float) -> np.ndarray:
    """Label windows positive iff intersection with any interval is > 0 samples.

    Half-open ``[start, end)`` on both the window and the interval, as the TUSZ
    and CHB-MIT protocols specify. Written as an explicit overlap test rather
    than a midpoint or containment test: the CHB-MIT protocol calls out that
    TFM's original implementation missed seizures that fully covered a window,
    which is exactly what a containment test does wrong.
    """
    y = np.zeros(n_windows, dtype=np.int64)
    for k in range(n_windows):
        w0 = k * stride
        w1 = w0 + win
        for (s, e) in intervals:
            s_smp, e_smp = s * fs, e * fs
            if min(w1, e_smp) - max(w0, s_smp) > 0:
                y[k] = 1
                break
    return y


# --------------------------------------------------------------------------- #
# Normalisation (the three schemes in PROTOCOLS.md appendix B)
# --------------------------------------------------------------------------- #
def norm_div100(x: np.ndarray) -> np.ndarray:
    """TUAB/TUEV/TUSZ/ISRUC/PhysioNet-MI/SEED-V/BCI-IV-2a. Input must be in uV."""
    return x / 100.0


def norm_q95(x: np.ndarray) -> np.ndarray:
    """CHB-MIT: per-window per-channel 95th percentile of |x|."""
    q = np.quantile(np.abs(x), 0.95, axis=-1, keepdims=True)
    return x / (q + 1e-8)


def compute_train_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sleep-EDF: per-channel mean/std over all retained train epochs.

    ``x`` is (N, C, T); reduces over N and T so the statistic is per-channel.
    """
    mean = x.mean(axis=(0, 2), keepdims=True)[0]      # (C, 1)
    std = x.std(axis=(0, 2), keepdims=True)[0]        # (C, 1)
    return mean, np.maximum(std, 1e-8)


def norm_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class Manifest:
    """The reproducibility unit. One manifest per dataset, shared by all models.

    ``protocol`` is the frozen config dict as loaded from configs/datasets/*.yaml,
    embedded verbatim so a manifest is self-describing even if the config later
    changes -- a mismatch between a run's manifest and the current config is a
    signal to re-run preprocessing, not to silently reuse.
    """

    dataset: str
    protocol: dict[str, Any]
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_sha256: dict[str, str] = field(default_factory=dict)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_split(self, name: str, *, subjects: list, n_windows: int,
                  class_counts: dict[int, int], discarded_tail: int = 0,
                  **extra) -> None:
        self.splits[name] = {
            "subjects": sorted(map(str, subjects)),
            "n_subjects": len(subjects),
            "n_windows": int(n_windows),
            "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "discarded_tail_samples": int(discarded_tail),
            **extra,
        }

    def exclude(self, item: str, reason: str, **extra) -> None:
        """Protocols require exclusions to be logged, not silently dropped."""
        self.excluded.append({"item": item, "reason": reason, **extra})

    def check_disjoint(self, strict: bool = True) -> list[dict]:
        """Check for subject leakage across splits.

        ``strict=False`` records the overlap in ``qc`` and returns it instead of
        raising. Used only where a frozen protocol's split rule is known to
        permit overlap and we are deliberately matching published practice --
        currently TUAB, see preprocessing/tuab.py. Everywhere else leakage is a
        bug and must be fatal.
        """
        names = list(self.splits)
        found = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = set(self.splits[names[i]]["subjects"])
                b = set(self.splits[names[j]]["subjects"])
                overlap = a & b
                if overlap:
                    found.append({
                        "splits": [names[i], names[j]],
                        "n_subjects": len(overlap),
                        "subjects": sorted(overlap),
                    })
        if found and strict:
            f = found[0]
            raise ValueError(
                f"subject leakage between {f['splits'][0]} and {f['splits'][1]}: "
                f"{f['subjects'][:10]}"
            )
        if found:
            self.qc["subject_overlap"] = found
            for f in found:
                print(f"[warn] {f['n_subjects']} subjects appear in both "
                      f"{f['splits'][0]} and {f['splits'][1]}: {f['subjects']}",
                      flush=True)
        return found

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        print(f"[manifest] {path}")


def save_split(out_dir: str, split: str, X: np.ndarray, y: np.ndarray) -> None:
    """Write one split as a consolidated npy pair.

    Consolidated rather than per-window pickles: the reference repo hit a
    ~15 it/s dataloader ceiling on TUAB regardless of GPU speed because each
    __getitem__ was a separate pickle.load of a small file. mmap-backed npy
    removes that bottleneck.
    """
    os.makedirs(out_dir, exist_ok=True)
    assert len(X) == len(y), f"{split}: {len(X)} windows vs {len(y)} labels"
    np.save(os.path.join(out_dir, f"{split}_signals.npy"), X)
    np.save(os.path.join(out_dir, f"{split}_labels.npy"), y)
    # ISRUC labels are 2-D (n_seq, epochs_per_seq); bincount needs 1-D.
    counts = np.bincount(np.asarray(y).ravel()).tolist()
    print(f"[{split}] {X.shape} -> {out_dir}  labels={counts}")


def assert_finite(x: np.ndarray, ctx: str) -> None:
    if not np.isfinite(x).all():
        n = int((~np.isfinite(x)).sum())
        raise ValueError(f"{ctx}: {n} non-finite values")
