"""Smoke-test every dataset reader against one real file.

    python -m tests.test_readers

Unit tests cover the maths; this covers the file-format reality (channel naming
variants, annotation quirks, unit scaling). Run it before launching a full
preprocessing job -- a 300 GB corpus is an expensive place to discover that a
channel name has a different suffix.
"""

from __future__ import annotations

import os
import traceback

import numpy as np
import yaml

CFG = "configs/datasets"
results: list[tuple[str, bool, str]] = []


def report(name: str, fn) -> None:
    try:
        detail = fn()
        results.append((name, True, detail))
    except Exception as e:                                    # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))
        if os.environ.get("VERBOSE"):
            traceback.print_exc()


def cfg(name: str) -> dict:
    return yaml.safe_load(open(os.path.join(CFG, f"{name}.yaml")))


def first_file(root: str, suffix: str) -> str:
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(suffix):
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"no {suffix} under {root}")


# --------------------------------------------------------------------------- #
def t_tuab():
    from preprocessing.common import preprocess_signal, window_signal, norm_div100
    from preprocessing.tuh_common import load_bipolar_uv

    c = cfg("tuab")
    f = first_file(os.path.join(c["raw_root"], "train", "normal"), ".edf")
    sig, fs = load_bipolar_uv(f)
    assert sig.shape[0] == 16, f"got {sig.shape[0]} channels"
    # EEG in microvolts is O(10-100); catches a missing or doubled 1e6 scaling
    amp = float(np.percentile(np.abs(sig), 95))
    assert 1 < amp < 5000, f"suspicious amplitude {amp:.1f} uV"
    z = preprocess_signal(sig, fs, fs_out=c["sample_rate"],
                          band=tuple(c["band"]), notch_freq=c["notch"])
    X, tail = window_signal(z, 2000, 2000)
    X = norm_div100(X)
    return (f"{os.path.basename(f)} fs={fs:g} -> {X.shape} "
            f"amp_p95={amp:.1f}uV tail={tail}")


def t_tuev():
    from preprocessing.tuev import read_rec
    from preprocessing.tuh_common import load_bipolar_uv

    c = cfg("tuev")
    f = first_file(os.path.join(c["raw_root"], "train"), ".edf")
    rec = f[:-4] + ".rec"
    ev = read_rec(rec)
    assert len(ev) > 0, "no events parsed"
    labs = sorted({int(r[3]) for r in ev})
    assert all(1 <= v <= 6 for v in labs), f"labels out of range: {labs}"
    durs = {round(float(r[2] - r[1]), 2) for r in ev}
    sig, fs = load_bipolar_uv(f)
    assert sig.shape[0] == 16
    return (f"{os.path.basename(f)} fs={fs:g} events={len(ev)} "
            f"labels={labs} durations={sorted(durs)[:3]}")


def t_tusz():
    from preprocessing.tusz import read_csv_bi
    from preprocessing.tuh_common import load_bipolar_uv, MissingChannels

    c = cfg("tusz")
    # cover all three montage families present in the corpus
    seen = {}
    for fam in ("01_tcp_ar", "02_tcp_le", "03_tcp_ar_a"):
        hit = None
        for dirpath, _, files in os.walk(os.path.join(c["raw_root"], "train")):
            if not dirpath.endswith(fam):
                continue
            for f in sorted(files):
                if f.endswith(".edf"):
                    hit = os.path.join(dirpath, f)
                    break
            if hit:
                break
        if hit is None:
            seen[fam] = "none found"
            continue
        try:
            sig, fs = load_bipolar_uv(hit)
            n_seiz = len(read_csv_bi(hit[:-4] + ".csv_bi"))
            seen[fam] = f"ok fs={fs:g} ch={sig.shape[0]} seiz={n_seiz}"
        except MissingChannels as e:
            seen[fam] = f"EXCLUDED ({len(e.missing)} missing)"
    return "; ".join(f"{k}: {v}" for k, v in seen.items())


def t_chbmit():
    from preprocessing.chbmit import parse_summary, pick_channels
    import mne

    c = cfg("chbmit")
    case = c["split"]["train"][0]
    d = os.path.join(c["raw_root"], case)
    ann = parse_summary(os.path.join(d, f"{case}-summary.txt"))
    n_seiz = sum(len(v) for v in ann.values())
    f = os.path.join(d, sorted(x for x in os.listdir(d) if x.endswith(".edf"))[0])
    raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    sig = pick_channels(raw, c["channels"]) * 1e6
    amp = float(np.percentile(np.abs(sig), 95))
    assert sig.shape[0] == 16
    assert 1 < amp < 5000, f"suspicious amplitude {amp:.1f} uV"
    return (f"{case} fs={fs:g} ch={sig.shape[0]} files_with_ann={len(ann)} "
            f"seizures={n_seiz} amp_p95={amp:.1f}uV")


def t_sleepedf():
    from preprocessing.sleepedf import pair_files, epochs_from_recording

    c = cfg("sleepedf")
    pairs = pair_files(c["raw_root"])
    assert len(pairs) > 100, f"only {len(pairs)} PSG/hypnogram pairs"
    sub, psg, hyp = pairs[0]
    X, y = epochs_from_recording(psg, hyp, c)
    assert X.shape[1] == 2 and X.shape[2] == 3000, str(X.shape)
    return (f"{len(pairs)} pairs; sub{sub:02d} -> {X.shape} "
            f"classes={sorted(set(y.tolist()))}")


def t_isruc():
    """Covers BOTH channel spellings.

    ISRUC subjects are split between an ear-reference spelling (F3-A2) and a
    mastoid one (F3-M2) for the same derivations. Testing a single subject
    passed while 60+ others failed in the real run, so this walks subjects until
    it has exercised one of each spelling.
    """
    import mne

    from preprocessing.isruc import process_subject, read_rec_as_edf

    c = cfg("isruc")
    root = c["raw_root"]
    subs = sorted((d for d in os.listdir(root) if d.isdigit()), key=int)

    seen = {}
    detail = []
    for s in subs:
        d = os.path.join(root, s)
        if not (os.path.exists(os.path.join(d, f"{s}.rec"))
                and os.path.exists(os.path.join(d, f"{s}_1.txt"))):
            continue
        raw = read_rec_as_edf(os.path.join(d, f"{s}.rec"))
        names = " ".join(raw.ch_names).upper()
        del raw
        spelling = "M" if "-M2" in names else ("A" if "-A2" in names else "?")
        if spelling in seen:
            continue
        X, y, dropped = process_subject(d, c)
        assert X.shape[1] == 20 and X.shape[2] == 6, str(X.shape)
        seen[spelling] = s
        detail.append(f"sub{s}({spelling}-ref) {X.shape} dropped={dropped}")
        if len(seen) >= 2:
            break
    if not seen:
        raise FileNotFoundError("no complete ISRUC subject found")
    return f"{len(subs)} subject dirs; " + "; ".join(detail)


def t_physionet_mi():
    from preprocessing.physionet_mi import process_run

    c = cfg("physionet_mi")
    f = os.path.join(c["raw_root"], "S001", "S001R04.edf")
    X, y = process_run(f, 4, c)
    assert X.shape[1] == 64 and X.shape[2] == 800, str(X.shape)
    assert set(y.tolist()) <= {0, 1}, f"run 4 should give classes 0/1, got {set(y)}"
    return f"S001R04 -> {X.shape} classes={sorted(set(y.tolist()))}"


def t_bci_iv_2a():
    from preprocessing.bci_iv_2a import epochs_from_run

    c = cfg("bci_iv_2a")
    os.environ.setdefault("MNE_DATA", c["raw_root"])
    from moabb.datasets import BNCI2014_001

    ds = BNCI2014_001()
    data = ds.get_data(subjects=[1])[1]
    session = sorted(data)[0]
    run_name = sorted(data[session])[0]
    X, y = epochs_from_run(data[session][run_name], c)
    assert X.shape[1] == 22 and X.shape[2] == 800, str(X.shape)
    return (f"sessions={sorted(data)} runs={sorted(data[session])} "
            f"S1/{session}/{run_name} -> {X.shape} classes={sorted(set(y.tolist()))}")


for name, fn in [
    ("TUAB", t_tuab),
    ("TUEV", t_tuev),
    ("TUSZ", t_tusz),
    ("CHB-MIT", t_chbmit),
    ("Sleep-EDF", t_sleepedf),
    ("ISRUC", t_isruc),
    ("PhysioNet-MI", t_physionet_mi),
    ("BCI-IV-2a", t_bci_iv_2a),
]:
    report(name, fn)

print("=" * 78)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<14} {detail}")
print("=" * 78)
nfail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - nfail}/{len(results)} readers OK")
