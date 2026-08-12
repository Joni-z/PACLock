"""Recompute hard rule 3 over runs that finished under the literal reading.

    python -m scripts.reverdict --runs runs [--apply]

``epoch0_peak_check`` used to fail any run whose validation curve peaked at
index 0. That condemned runs which had converged inside the first epoch rather
than runs which never trained (see the docstring on the check itself). The rule
now asks whether the model ever cleared chance, which needs no information the
finished runs did not already store: ``val_curve``, ``primary_metric`` and
``config.num_classes`` are all in ``result.json``.

So the verdict is recomputed from the stored curve instead of re-running the
job. The previous verdict is kept as ``verdict_v1`` and the new one records
``rule_version: 2``, so a cell that changed status can be traced to this script
rather than looking like it silently moved.

Runs a report by default; ``--apply`` is required to write.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paclock_bench.training.metrics import epoch0_peak_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    changed, unchanged, skipped = [], 0, []
    for path in sorted(glob.glob(os.path.join(args.runs, "*", "*", "result.json"))):
        with open(path) as f:
            res = json.load(f)
        curve = res.get("val_curve")
        if not curve:
            skipped.append((path, "no val_curve"))
            continue
        # Already migrated: recomputing is harmless but reporting it as a change
        # every time is not, so leave it alone.
        if res.get("verdict", {}).get("rule_version") == 2:
            unchanged += 1
            continue

        old = res.get("verdict", {})
        # Same inputs the live check gets: PR-AUC's chance level is the val
        # positive rate, and the dead-run test reads the test scores.
        val_counts = (res.get("class_counts") or {}).get("val")
        prevalence = None
        if val_counts and len(val_counts) == 2 and sum(val_counts):
            prevalence = val_counts[1] / sum(val_counts)
        new = epoch0_peak_check(curve, res.get("primary_metric"),
                                (res.get("config") or {}).get("num_classes"),
                                prevalence, res.get("test"))
        new["rule_version"] = 2

        if old.get("ok") != new["ok"] or old.get("status") != new["status"]:
            changed.append((res["name"], res["seed"], old.get("status"),
                            new["status"], new["reason"]))
        else:
            unchanged += 1

        if args.apply:
            if old:
                res["verdict_v1"] = old
            res["verdict"] = new
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f, indent=1)
            os.replace(tmp, path)   # atomic, so a reader never sees a half file

    print(f"{len(changed)} verdicts change, {unchanged} unchanged, "
          f"{len(skipped)} skipped")
    for name, seed, o, n, why in changed:
        print(f"  {name:34s} seed{seed}  {o} -> {n}")
        print(f"      {why}")
    for path, why in skipped:
        print(f"  skip {path}: {why}")
    if not args.apply:
        print("\ndry run -- pass --apply to write")


if __name__ == "__main__":
    main()
