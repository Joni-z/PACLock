"""Promote the surviving CNN-Transformer LR variant into the canonical cell.

`cnn_transformer` collapsed to chance on FACED and PhysioNet-MI at the group-A
recipe's lr=1e-3 (0/3 seeds admissible on both). A sweep found lr=1e-4 and
lr=3e-4 both fix it outright; lr=1e-4 scored higher on both corpora and is
adopted. See docs/PRETRAIN.md sec 14 for why deviating here is correct
rather than reporting an empty cell.

The collapsed lr=1e-3 runs are archived, not deleted -- they are the evidence
that the deviation was necessary.
"""
import glob
import json
import os
import shutil

WINNER = "lr1e4"
CORPORA = ["faced", "physionet_mi"]
ARCHIVE = "archive/runs_cnnT_lr1e3_collapsed"

for ds in CORPORA:
    canon = "runs/%s-cnn_transformer" % ds
    src = "runs/%s-cnn_transformer_%s" % (ds, WINNER)

    if not os.path.isdir(src):
        print("  %-14s no %s runs -- skipped" % (ds, WINNER))
        continue

    # archive the collapsed originals
    if os.path.isdir(canon):
        dst = os.path.join(ARCHIVE, os.path.basename(canon))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(canon, dst)
        print("  %-14s archived collapsed lr=1e-3 runs -> %s" % (ds, dst))

    # promote the winner, rewriting `name` so the result self-identifies as
    # the canonical cell (fill_xlsx keys off the directory, but a result.json
    # whose name disagreed with its path would be a trap for the next reader)
    shutil.copytree(src, canon)
    for p in sorted(glob.glob(os.path.join(canon, "seed*/result.json"))):
        r = json.load(open(p))
        r["name"] = "%s-cnn_transformer" % ds
        r.setdefault("config", {})["name"] = r["name"]
        r["recipe_deviation"] = (
            "lr %s (group-A recipe lr=1e-3 collapses to chance on this corpus, "
            "0/3 seeds admissible; see docs/PRETRAIN.md sec 14)"
            % r.get("config", {}).get("lr")
        )
        json.dump(r, open(p, "w"), indent=2, ensure_ascii=False)
    n = len(glob.glob(os.path.join(canon, "seed*/result.json")))
    print("  %-14s promoted %s -> %s (%d seeds)" % (ds, WINNER, canon, n))

print("done")
