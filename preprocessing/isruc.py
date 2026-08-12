"""ISRUC-Sleep Subgroup I -> sequence npy + manifest. Protocol: docs/PROTOCOLS.md sec.6.

    python -m preprocessing.isruc --config configs/datasets/isruc.yaml

Already 200 Hz, so no resampling. Output is grouped into sequences of 20
consecutive 30 s epochs: shape (N, 20, 6, 6000). Per-subject tails shorter than
20 epochs are dropped and counted -- the protocol refuses to pre-commit to a
sample count, so the manifest reports what was actually produced.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import mne
import numpy as np
import yaml

from .common import (
    highpass,
    Manifest,
    assert_finite,
    bandpass,
    norm_div100,
    notch,
    save_split,
    sha256_file,
)


def read_labels(path: str) -> np.ndarray:
    """Expert-1 scoring: one integer stage code per 30 s epoch."""
    vals = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(float(line)))
    return np.array(vals, dtype=np.int64)


def read_rec_as_edf(path: str) -> mne.io.BaseRaw:
    """ISRUC ships EDF content under a ``.rec`` extension.

    MNE dispatches on the file extension and refuses ``.rec`` outright, so the
    file is exposed to it through a temporary ``.edf`` symlink. The bytes are
    genuine EDF -- verified with ``file(1)`` -- so no conversion is needed.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        link = os.path.join(tmp, os.path.basename(path)[:-4] + ".edf")
        os.symlink(os.path.abspath(path), link)
        return mne.io.read_raw_edf(link, preload=True, verbose="ERROR")


def pick_channels(raw: mne.io.BaseRaw, wanted: list[str]) -> np.ndarray:
    """Select the wanted derivations, tolerating ISRUC's two naming conventions.

    Subjects are split between an ear-reference spelling (``F3-A2``) and a
    mastoid one (``F3-M2``). A1/A2 and M1/M2 are the same two reference
    electrodes -- left and right ear/mastoid -- so ``F3-A2`` and ``F3-M2`` are
    the same derivation, and the protocol's channel list matches either. Only
    subject 2 uses the A spelling among the first few, which is why a smoke test
    on one subject did not catch this.
    """
    def key(s: str) -> str:
        k = s.upper().replace(" ", "").replace("-", "")
        return k.replace("A1", "M1").replace("A2", "M2")       # A_n == M_n

    index: dict[str, int] = {}
    for i, name in enumerate(raw.ch_names):
        index.setdefault(key(name), i)
    missing = [c for c in wanted if key(c) not in index]
    if missing:
        raise KeyError(f"missing channels {missing} (have {raw.ch_names})")
    data = raw.get_data()
    return np.stack([data[index[key(c)]] for c in wanted])


def process_subject(subject_dir: str, cfg: dict):
    """One subject -> (n_seq, 20, 6, 6000) sequences and (n_seq, 20) labels."""
    sub = os.path.basename(subject_dir.rstrip("/"))
    rec = os.path.join(subject_dir, f"{sub}.rec")
    lab = os.path.join(subject_dir, f"{sub}_1.txt")          # expert 1, per CBraMod
    if not os.path.exists(rec):
        raise FileNotFoundError(rec)
    if not os.path.exists(lab):
        raise FileNotFoundError(lab)

    raw = read_rec_as_edf(rec)
    fs = float(raw.info["sfreq"])
    sig = pick_channels(raw, cfg["channels"]) * 1e6          # volts -> uV
    del raw

    # PAC protocol drops the band-pass for a high-pass; see
    # scripts/make_pac_protocol.py. One implementation serves both so
    # only the filter settings can differ between the two protocols.
    _b = cfg.get("band")
    sig = (bandpass(sig.astype(np.float64), fs, _b[0], _b[1]) if _b
           else highpass(sig.astype(np.float64), fs, cfg["hp"]))
    if cfg.get("notch"):
        sig = notch(sig, fs, cfg["notch"])
    if abs(fs - cfg["sample_rate"]) > 1e-6:
        raise ValueError(f"expected {cfg['sample_rate']} Hz, got {fs}")

    y_raw = read_labels(lab)
    label_map = {int(k): int(v) for k, v in cfg["label_map"].items()}
    epoch = int(cfg["window_sec"] * fs)
    n_epochs = min(len(y_raw), sig.shape[1] // epoch)

    X, y = [], []
    for k in range(n_epochs):
        code = int(y_raw[k])
        if code not in label_map:
            continue                                          # unscored/artefact
        X.append(sig[:, k * epoch:(k + 1) * epoch])
        y.append(label_map[code])
    if not X:
        raise ValueError("no scored epochs")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    seq_len = cfg["sequence"]["length"]
    n_seq = len(X) // seq_len
    dropped = len(X) - n_seq * seq_len
    if n_seq == 0:
        raise ValueError(f"fewer than {seq_len} epochs")
    X = X[:n_seq * seq_len].reshape(n_seq, seq_len, *X.shape[1:])
    y = y[:n_seq * seq_len].reshape(n_seq, seq_len)
    return X, y, dropped


def _worker(subject_dir: str, cfg: dict):
    """Pool entry point: returns (subject, X, y, dropped, sha, error)."""
    sub = os.path.basename(subject_dir.rstrip("/"))
    try:
        X, y, dropped = process_subject(subject_dir, cfg)
        assert_finite(X, f"subject {sub}")
        rec = os.path.join(subject_dir, f"{sub}.rec")
        return sub, X, y, dropped, sha256_file(rec), None
    except Exception as e:                                     # noqa: BLE001
        return sub, None, None, 0, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # Each subject is an independent EDF read + filter, so this parallelises
    # cleanly. The flag also has to exist because slurm/preprocess.slurm passes
    # it to every dataset uniformly.
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out_dir = cfg["raw_root"], cfg["out_dir"]
    sp = cfg["split"]
    man = Manifest(dataset=cfg["dataset"], protocol=cfg)

    def split_of(n: int) -> str | None:
        for name in ("train", "val", "test"):
            lo, hi = sp[name]
            if lo <= n <= hi:
                return name
        return None

    buckets = {k: {"X": [], "y": [], "subs": [], "dropped": 0}
               for k in ("train", "val", "test")}

    subs = sorted((d for d in os.listdir(root) if d.isdigit()), key=int)
    todo = []
    for s in subs:
        split = split_of(int(s))
        if split is None:
            man.exclude(s, f"subject {int(s)} outside all split ranges")
            continue
        todo.append((os.path.join(root, s), split))

    split_by_sub = {os.path.basename(p.rstrip("/")): sp for p, sp in todo}
    with Pool(args.jobs) as pool:
        for sub, X, y, dropped, sha, err in pool.imap_unordered(
            partial(_worker, cfg=cfg), [p for p, _ in todo], chunksize=1
        ):
            split = split_by_sub[sub]
            if err is not None:
                man.exclude(sub, err, split=split)
                print(f"  sub {sub:>3} EXCLUDED ({err})", flush=True)
                continue
            buckets[split]["X"].append(X)
            buckets[split]["y"].append(y)
            buckets[split]["subs"].append(sub)
            buckets[split]["dropped"] += dropped
            man.raw_sha256[f"{sub}.rec"] = sha
            print(f"  sub {sub:>3} [{split}] {X.shape} dropped_tail={dropped}",
                  flush=True)

    # Pool returns out of order; keep the arrays deterministic across runs.
    for b in buckets.values():
        if b["subs"]:
            order = np.argsort([int(s) for s in b["subs"]])
            b["X"] = [b["X"][i] for i in order]
            b["y"] = [b["y"][i] for i in order]
            b["subs"] = [b["subs"][i] for i in order]

    for split, b in buckets.items():
        if not b["X"]:
            raise RuntimeError(f"{split}: no subjects processed")
        X = norm_div100(np.concatenate(b["X"])).astype(np.float32)
        y = np.concatenate(b["y"]).astype(np.int64)
        save_split(out_dir, split, X, y)
        man.add_split(split, subjects=b["subs"], n_windows=len(X),
                      class_counts=Counter(y.ravel().tolist()),
                      discarded_tail=b["dropped"], shape=list(X.shape[1:]),
                      n_epochs=int(y.size), sequence_length=cfg["sequence"]["length"])

    man.check_disjoint()
    man.qc = {
        "n_excluded": len(man.excluded),
        "n_subjects_found": len(subs),
        "expected_n_samples": cfg.get("expected", {}).get("n_samples"),
        "actual_n_epochs": sum(v["n_epochs"] for v in man.splits.values()),
        "note": "89,240 deliberately not asserted; counts are outputs",
    }
    man.save(os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
