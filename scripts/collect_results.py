"""Aggregate runs/ into the matrix shape, with the published anchors alongside.

    python -m scripts.collect_results [--runs runs] [--dataset tuev]

Applies the hard rules rather than just printing means:

* rule 4 -- fewer than 3 seeds is reported as ``n<3``, not as a result
* rule 3 -- any seed flagged mis-configured (val peak at epoch 0, or a flat val
  curve) is counted and the cell is marked, because such a cell may not be
  written to the matrix
* rule 1 -- where the xlsx lists a published value for a row, the reproduction
  gate (published inside mean +- 2*std) is evaluated

The published anchors come from ``configs/published_reference.json``, extracted
from the xlsx's own "外部参考值" blocks. They are calibration anchors only and
never go into the paper table beside our numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# our model key -> the row label used in the xlsx reference block
MODEL_TO_PUBLISHED = {
    # Group B/C foundation models -> the xlsx row for the checkpoint we load.
    # The sheets label the same model differently (with/without a "(单语料)"
    # suffix), so the lookup tries each spelling in order.
    "biot": ["BIOT\u2605", "BIOT\u2605 (单语料预训练)"],
    "labram": ["LaBraM-Base\u2605", "LaBraM-Base\u2605 (单语料)"],
    "cbramod": ["CBraMod\u2020 (4 语料)"],
    "tfm": ["TFM-Tokenizer"],
    "sparcnet": "SPaRCNet",
    "contrawr": "ContraWR",
    "cnn_transformer": "CNN-Transformer",
    "ffcl": "FFCL",
    "st_transformer": "ST-Transformer",
}

DATASET_TO_SHEET = {
    "tuab": "TUAB", "tuev": "TUEV", "tusz": "TUSZ", "chbmit": "CHB-MIT",
    "sleepedf": "Sleep-EDF", "isruc": "ISRUC", "physionet_mi": "PhysioNet-MI",
    "faced": "FACED", "bci_iv_2a": "BCI-IV-2a",
}

# Datasets whose protocol forbids point-comparison with the published anchors.
# CHB-MIT: "不与 TFM CHB-MIT 数字逐点比较;论文中必须注明 split 与标签规则不同"
# -- we use a strict subject-disjoint split and corrected overlap labelling, so
# a difference against those anchors is the intended outcome, not a failure.
NO_POINT_COMPARISON = {
    "chbmit": "protocol: strict subject-disjoint split + corrected labelling; "
              "TFM anchors are not point-comparable",
}

# Individual published cells the protocol itself rejects. Hard rule 3 names this
# one explicitly: "TFM-Tokenizer 论文里 CBraMod 的 TUAB 行就是这种
# (BAcc 0.5000±0.0000)". Its AUROC of 0.5281 with a balanced accuracy of exactly
# 0.5 is a model that never learned, so comparing against it would be comparing
# against a run the workbook already ruled inadmissible.
REJECTED_ANCHORS = {
    ("tuab", "cbramod"): "published row is mis-configured (BAcc 0.5000, "
                         "AUROC 0.5281) -- hard rule 3 names it explicitly",
}

# our metric key -> the column label in the xlsx reference block
METRIC_TO_PUBLISHED = {
    "balanced_acc": "Balanced Acc",
    "cohen_kappa": "Cohen's Kappa",
    "weighted_f1": "Weighted F1",
    "auroc": "AUROC",
    "pr_auc": "AUC-PR",
}


def load_published(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--reference", default="configs/published_reference.json")
    args = ap.parse_args()

    published = load_published(args.reference)
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.runs, "*", "seed*", "result.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if args.dataset and r["dataset"] != args.dataset:
            continue
        # "tuab-biot_prest16" -> variant "biot_prest16"; without this every
        # pretrained/scratch pair collapses into one ambiguous "biot" row
        variant = r["name"].split("-", 1)[1] if "-" in r["name"] else r["model"]
        groups[(r["dataset"], variant, r.get("group"))].append(r)

    if not groups:
        print("no results found")
        return

    by_dataset = defaultdict(list)
    for (ds, model, grp), rs in groups.items():
        by_dataset[ds].append((model, grp, rs))

    for ds in sorted(by_dataset):
        sheet = DATASET_TO_SHEET.get(ds, ds)
        ref = published.get(sheet, {})
        ref_rows = ref.get("rows", {})
        primary = groups[[k for k in groups if k[0] == ds][0]][0]["primary_metric"]
        metrics = sorted(groups[[k for k in groups if k[0] == ds][0]][0]["test"])

        print(f"\n{'=' * 96}")
        print(f"{sheet}   primary = {primary}"
              + (f"   [{len(ref_rows)} published anchors]" if ref_rows else
                 "   [no comparable published numbers]"))
        if ds in NO_POINT_COMPARISON:
            print(f"  NOTE: {NO_POINT_COMPARISON[ds]}")
        print("=" * 96)
        head = "%-18s %-3s %-7s " % ("model", "n", "params")
        head += " ".join("%-17s" % m for m in metrics)
        head += " status"
        print(head)
        print("-" * 96)

        for model, grp, rs in sorted(by_dataset[ds], key=lambda x: x[0]):
            n = len(rs)
            misconf = sum(1 for r in rs if not r["verdict"]["ok"])
            cells = []
            for m in metrics:
                vals = [r["test"][m] for r in rs]
                mean, std = float(np.mean(vals)), float(np.std(vals, ddof=0))
                cells.append("%.4f±%.4f" % (mean, std))

            status = []
            if n < 3:
                status.append(f"n<3 (rule 4)")
            if misconf:
                status.append(f"{misconf} mis-configured (rule 3)")

            # Comparison against the published anchor.
            #
            # Hard rule 1 (the reproduction gate, published value inside
            # mean+-2std) is written for **group B** -- "B 组任一模型进表前".
            # Group A's role is different: "对不上 TFM-Tokenizer 的已发表值,
            # 问题在 pipeline 不在模型". So for group A the anchor is a
            # diagnostic signal about the pipeline, not a pass/fail gate, and
            # reporting it as "NOT reproduced" overstates it.
            base = rs[0]["model"]
            labels = MODEL_TO_PUBLISHED.get(base, [])
            if isinstance(labels, str):
                labels = [labels]
            pub_row, pub_label = {}, None
            for lab in labels:
                if lab in ref_rows:
                    pub_row, pub_label = ref_rows[lab], lab
                    break
            pub_col = METRIC_TO_PUBLISHED.get(primary)
            if ds in NO_POINT_COMPARISON:
                status.append("anchors not point-comparable")
            elif (ds, base) in REJECTED_ANCHORS:
                status.append(f"anchor rejected: {REJECTED_ANCHORS[(ds, base)]}")
            elif pub_row and pub_col in pub_row:
                p = pub_row[pub_col]
                vals = [r["test"][primary] for r in rs]
                mean, std = float(np.mean(vals)), float(np.std(vals, ddof=0))
                delta = mean - p
                if grp == "B":
                    lo, hi = mean - 2 * std, mean + 2 * std
                    ok = n >= 3 and std > 0 and lo <= p <= hi
                    status.append(
                        f"{'reproduced' if ok else 'NOT reproduced'} (pub {p:.4f})")
                else:
                    # calibration signal: flag only deviations big enough to
                    # suspect the pipeline rather than run-to-run variation
                    flag = "  <-- CHECK PIPELINE" if abs(delta) > 0.10 else ""
                    status.append(f"vs pub {p:.4f}: {delta:+.4f}{flag}")

            print("%-18s %-3d %-7.2f " % (model, n, rs[0]["n_params_M"])
                  + " ".join("%-17s" % c for c in cells)
                  + "  " + "; ".join(status or ["ok"]))

        if ref_rows:
            print("\n  published anchors (calibration only, never beside ours in the paper):")
            for label, vals in ref_rows.items():
                if label in MODEL_TO_PUBLISHED.values():
                    s = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
                    print(f"    {label:<22} {s}")


if __name__ == "__main__":
    main()
