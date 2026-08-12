"""Report the two waves that decide which PACLock goes into pretraining.

    python3 -m scripts.collect_waves [--runs runs]

Wave 1  base vs patch100, TUEV + ISRUC, 3 seeds, 20 epochs.
        Settles patch100, and -- more importantly -- gives the first honest
        seed spread under the GEMM tokeniser. Every threshold used so far
        (0.029) came from comparing two runs across a code change, which is a
        different quantity.

Wave 2  base / size_base / size_large / patch100 at 60 epochs, 1 seed.
        Three separate sweeps concluded "scaling hurts", all of them at 20
        epochs, which was all the old speed allowed. A 8.5M model and a 1.6M
        model do not need the same budget, so "capacity does not help" and
        "capacity was never trained" have never been distinguished. For a
        model that is supposed to go into PRETRAINING this is the load-bearing
        question: a backbone that cannot use capacity is not a backbone.

Decision rule, fixed here before the numbers land so it cannot be fitted to
them:

  * a wave-1 delta counts only if it exceeds the measured 3-seed spread of the
    control on that dataset, and has the same sign on both datasets;
  * wave 2 says "scale" only if size_large at 60 epochs beats base at 60 epochs
    by more than that same spread, on both datasets;
  * if 60-epoch base beats 20-epoch base by more than the spread, then every
    20-epoch conclusion in runs_conv_tokenizer/ was budget-limited and none of
    the architecture rankings from it can be carried over.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def load(runs, pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs, pattern, "seed*", "result.json"))):
        cell = os.path.basename(os.path.dirname(os.path.dirname(p)))
        r = json.load(open(p))
        out.setdefault(cell, []).append(r)
    return out


def fmt(rs, key):
    vals = sorted(r["test"][key] for r in rs)
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else None
    return m, sd, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    cells = load(args.runs, "*-cand_*")
    if not cells:
        print("no cells finished yet")
        return

    for wave, keep in [("WAVE 1  (20 epochs, 3 seeds)", lambda a: "e60" not in a),
                       ("WAVE 2  (60 epochs, 1 seed)", lambda a: "e60" in a)]:
        rows = {}
        for cell, rs in cells.items():
            ds, arm = cell.split("-cand_", 1)
            if not keep(arm):
                continue
            rows.setdefault(ds, {})[arm] = rs
        if not rows:
            continue
        print("=" * 74)
        print(wave)
        print("=" * 74)
        for ds, arms in sorted(rows.items()):
            key = next(iter(arms.values()))[0]["primary_metric"]
            ctrl = arms.get("base") or arms.get("e60_base")
            base_m, base_sd, _ = fmt(ctrl, key) if ctrl else (None, None, None)
            spread = base_sd
            print("\n  %s  (%s)" % (ds, key))
            if spread is not None:
                print("    control spread over %d seeds: sd = %.4f  "
                      "-> a delta must clear ~%.4f to mean anything"
                      % (len(ctrl), spread, 2 * spread))
            for arm, rs in sorted(arms.items(),
                                  key=lambda kv: -fmt(kv[1], key)[0]):
                m, sd, vals = fmt(rs, key)
                d = "" if base_m is None else "%+.4f" % (m - base_m)
                verdict = ""
                if base_m is not None and spread:
                    verdict = "  SIGNIFICANT" if abs(m - base_m) > 2 * spread \
                        else "  (within noise)"
                ep = rs[0].get("epochs_run")
                print("    %-16s %.4f %s  %-9s %dseed ep=%-3s%s"
                      % (arm, m, ("±%.4f" % sd) if sd else "       ", d,
                         len(rs), ep, verdict))
        print()


if __name__ == "__main__":
    main()
