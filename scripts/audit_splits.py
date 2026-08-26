"""Split-pathology audit across every processed corpus.

    python -m scripts.audit_splits

Born from the IIIC incident (2026-08-26): a sorted-by-ID patient split put
44% class-1 / 0.9% class-2 in train against 8% / 18% in test, and every
model's number measured the shift instead of the model. This audit checks
each corpus for the whole family of such defects BEFORE any more compute is
spent on them:

  A  subject overlap across splits (train/val/test must be disjoint)
  B  class-mix shift: max |p_split - p_global| per class, plus the relative
     shift of the rarest class (rare-positive corpora fail relatively long
     before they fail absolutely)
  C  window-count ratios: a "70/15/15 by subjects" cut can be 34/33/33 by
     windows when subject sizes correlate with ID order (IIIC's second bug)
  D  test-set statistical power: classes with <30 test windows, or <5 test
     subjects, cannot support a comparison
  E  cross-split duplicate windows: identical signal content appearing in
     two splits is leakage regardless of subject bookkeeping. Fingerprint =
     exact float values of channel 0 subsampled (::64), matched exactly.

Official-split corpora are audited too but their findings are REPORTED, not
fixed -- the published protocol is the protocol.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = "/work1/chenyuyou/yifanwang/Zhizhe/processed"
OFFICIAL = {"tuab", "tuev", "sleepedf", "isruc", "caueeg", "faced",
            "bci_iv_2a", "physionet_mi"}   # split published/official
SPLITS = ("train", "val", "test")


def fingerprints(path: str) -> set:
    X = np.load(path, mmap_mode="r")
    n = X.shape[0]
    idx = np.arange(n)
    out = set()
    for i in idx:
        v = np.asarray(X[i, 0, ::64], dtype=np.float32)
        out.add(v.tobytes())
    return out


def audit(ds: str) -> list[str]:
    d = os.path.join(ROOT, ds)
    man = json.load(open(os.path.join(d, "manifest.json")))
    flags = []
    subs, mixes, wins = {}, {}, {}
    n_classes = 0
    for sp in SPLITS:
        v = man["splits"].get(sp)
        if v is None:
            flags.append(f"MISSING split {sp}")
            continue
        subs[sp] = set(map(str, v["subjects"]))
        cc = {int(k): int(c) for k, c in v["class_counts"].items()}
        n_classes = max(n_classes, max(cc) + 1)
        wins[sp] = sum(cc.values())
        mixes[sp] = cc

    # A. subject overlap
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in subs and b in subs:
            ov = subs[a] & subs[b]
            if ov:
                flags.append(f"A-LEAK {a}∩{b}: {len(ov)} subjects e.g. {sorted(ov)[:3]}")

    # B. class-mix shift
    tot = np.zeros(n_classes)
    for sp in mixes:
        for k, c in mixes[sp].items():
            tot[k] += c
    p_glob = tot / tot.sum()
    worst_abs, worst_rel = 0.0, 0.0
    for sp in mixes:
        w = sum(mixes[sp].values())
        p = np.array([mixes[sp].get(k, 0) for k in range(n_classes)]) / max(w, 1)
        worst_abs = max(worst_abs, float(np.abs(p - p_glob).max()))
        rare = int(np.argmin(np.where(p_glob > 0, p_glob, np.inf)))
        if p_glob[rare] > 0:
            worst_rel = max(worst_rel, float(abs(p[rare] - p_glob[rare]) / p_glob[rare]))
    if worst_abs > 0.05:
        flags.append(f"B-MIXSHIFT max|Δp|={worst_abs:.3f}")
    if worst_rel > 0.5:
        flags.append(f"B-RARESHIFT rarest-class relΔ={worst_rel:.0%}")

    # C. window ratios (unofficial splits should be ~70/15/15 by windows)
    if ds not in OFFICIAL and all(sp in wins for sp in SPLITS):
        total = sum(wins.values())
        r = {sp: wins[sp] / total for sp in SPLITS}
        if abs(r["train"] - 0.70) > 0.10:
            flags.append(f"C-RATIO train={r['train']:.0%} (windows), "
                         f"val={r['val']:.0%}, test={r['test']:.0%}")
        if wins["train"] < wins["test"]:
            flags.append("C-INVERTED train smaller than test")

    # D. test power
    if "test" in mixes:
        weak = {k: c for k, c in mixes["test"].items() if c < 30}
        if weak:
            flags.append(f"D-POWER test classes with <30 windows: {weak}")
        if len(subs.get("test", ())) < 5:
            flags.append(f"D-SUBJ only {len(subs['test'])} test subjects")

    # E. cross-split duplicates (signal content)
    try:
        fp = {sp: fingerprints(os.path.join(d, f"{sp}_signals.npy"))
              for sp in SPLITS if sp in wins}
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            if a in fp and b in fp:
                dup = len(fp[a] & fp[b])
                if dup:
                    flags.append(f"E-DUP {a}∩{b}: {dup} identical windows")
    except Exception as e:                                    # noqa: BLE001
        flags.append(f"E-SKIP fingerprinting failed: {type(e).__name__}")

    return flags


def main():
    corpora = sorted(d for d in os.listdir(ROOT)
                     if os.path.isfile(os.path.join(ROOT, d, "manifest.json"))
                     and not d.startswith("_"))
    print(f"auditing {len(corpora)} corpora under {ROOT}\n")
    clean, dirty = [], []
    for ds in corpora:
        try:
            flags = audit(ds)
        except Exception as e:                                # noqa: BLE001
            flags = [f"AUDIT-ERROR {type(e).__name__}: {e}"]
        tag = "OFFICIAL" if ds in OFFICIAL else "ours"
        if flags:
            dirty.append(ds)
            print(f"[{ds}] ({tag})")
            for f in flags:
                print(f"    {f}")
        else:
            clean.append(ds)
    print(f"\nCLEAN: {', '.join(clean)}")
    print(f"FLAGGED: {', '.join(dirty) if dirty else '(none)'}")


if __name__ == "__main__":
    main()
