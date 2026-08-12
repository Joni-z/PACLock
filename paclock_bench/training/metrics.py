"""Metrics, per PROTOCOLS.md appendix A.

The PR-AUC definition is protocol-specified and is *not* sklearn's
``average_precision_score``. CBraMod's evaluator takes the trapezoidal integral
of the precision-recall curve, ``auc(recall, precision)``. The two differ:
average precision is a step-wise sum that does not interpolate between
thresholds, while the trapezoidal integral does. On CHB-MIT (~1% positives) the
gap is large enough to change rankings, so the protocol pins the trapezoidal
form and this module implements only that. Reporting AP under the name PR-AUC
would silently make our numbers incomparable to the published anchors.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal PR-AUC -- CBraMod ``finetune_evaluator.py`` definition."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray,
                    num_classes: int) -> dict[str, float]:
    """Return every metric the protocol asks for, for any dataset.

    ``logits`` is (N, num_classes) for multiclass, or (N,) / (N, 1) raw logits
    for the binary BCEWithLogits datasets (TUAB, TUSZ, CHB-MIT).
    """
    y_true = np.asarray(y_true).ravel()
    logits = np.asarray(logits)

    if num_classes == 2:
        if logits.ndim == 2 and logits.shape[1] == 2:
            score = _softmax(logits)[:, 1]
        else:                                   # single-logit BCE head
            score = 1.0 / (1.0 + np.exp(-logits.ravel()))
        y_pred = (score >= 0.5).astype(np.int64)
        return {
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "pr_auc": pr_auc(y_true, score),
            "auroc": float(roc_auc_score(y_true, score)),
        }

    y_pred = logits.argmax(axis=-1)
    return {
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


# Primary metric per dataset -- best-checkpoint selection uses this key.
PRIMARY_METRIC = {
    "tuab": "auroc",
    "tuev": "cohen_kappa",
    "tusz": "pr_auc",
    "chbmit": "pr_auc",
    "sleepedf": "cohen_kappa",
    "isruc": "cohen_kappa",
    "physionet_mi": "balanced_acc",
    "faced": "balanced_acc",
    "bci_iv_2a": "balanced_acc",
}


def primary_metric(dataset: str) -> str:
    if dataset not in PRIMARY_METRIC:
        raise KeyError(
            f"unknown dataset {dataset!r}; add its primary metric to "
            f"PRIMARY_METRIC (see docs/PROTOCOLS.md appendix A)"
        )
    return PRIMARY_METRIC[dataset]


# --------------------------------------------------------------------------- #
# Multi-seed aggregation (hard rule 4) and the reproduction gate (hard rule 1)
# --------------------------------------------------------------------------- #
def aggregate_seeds(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean/std over seeds. Rule 4: single-seed numbers cannot enter the table,
    so ``n_seeds`` is carried through for the table writer to check."""
    if not runs:
        return {}
    keys = sorted(set().union(*(r.keys() for r in runs)))
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "n_seeds": len(vals),
            "values": [float(v) for v in vals],
        }
    return out


def reproduction_gate(published: float, agg: dict[str, float]) -> dict[str, object]:
    """Hard rule 1: a group-B model enters the table only if the published value
    falls inside mean +- 2*std of our reproduction, over >= 3 seeds.

    A zero-std reproduction is treated as failing rather than as an exact match:
    identical values across seeds means the run is not actually varying with the
    seed, which is the same pathology rule 3 rejects.
    """
    mean, std, n = agg["mean"], agg["std"], agg["n_seeds"]
    lo, hi = mean - 2 * std, mean + 2 * std
    passed = bool(n >= 3 and std > 0 and lo <= published <= hi)
    return {
        "published": float(published),
        "mean": mean,
        "std": std,
        "n_seeds": n,
        "interval": [lo, hi],
        "passed": passed,
        "status": "reproduced" if passed else "not reproduced",
        "reason": (
            "ok" if passed
            else f"n_seeds={n} < 3" if n < 3
            else "std == 0 (run not varying with seed)" if std == 0
            else f"published {published:.4f} outside [{lo:.4f}, {hi:.4f}]"
        ),
    }


# Value a metric takes when the model has learned nothing. PR-AUC is absent
# because its chance level is the positive prevalence, not a constant: CHB-MIT
# is 150 positives in 21184 val windows (0.71%), and healthy runs there score
# 0.35-0.55, so any fixed floor near 0.5 would condemn them. It is passed in.
CHANCE = {"auroc": 0.5, "cohen_kappa": 0.0}

# How far above chance the best validation score must sit before an early peak
# is read as "converged fast" rather than "never started". Set from the observed
# seed-to-seed spread on the corpora in the matrix, which runs to ~0.03; 0.05
# clears that without being so wide that a genuinely dead run slips through
# (the CHB-MIT/CBraMod failure sits at exactly chance).
LEARNED_MARGIN = 0.05


def chance_level(metric: str | None, num_classes: int | None = None,
                 prevalence: float | None = None) -> float | None:
    """Score a model gets on ``metric`` by not learning, or None if unknowable.

    ``prevalence`` is the positive rate of the split the curve was measured on,
    needed only for PR-AUC. Returning None rather than a guess matters: the
    caller treats an unknown floor as "cannot say the model failed", so a
    missing prevalence can never manufacture a mis-configured verdict.
    """
    if metric in CHANCE:
        return CHANCE[metric]
    if metric == "balanced_acc" and num_classes:
        return 1.0 / num_classes
    if metric == "pr_auc":
        return prevalence
    return None


def epoch0_peak_check(val_curve: list[float], metric: str | None = None,
                      num_classes: int | None = None,
                      prevalence: float | None = None,
                      test_metrics: dict[str, float] | None = None,
                      ) -> dict[str, object]:
    """Hard rule 3: the run never learned => mis-configured, refuse to write.

    ``val_curve`` is the primary-metric value at each validation point in order.

    The rule is written in PROTOCOLS.md as "val peak at epoch 0", and taking that
    literally is wrong. Index 0 is not "before training" -- it is the first
    *evaluation*, and with per-epoch validation on TUAB at batch 512 that is
    ~2700 optimiser steps in, by which point these models have converged. So the
    literal test condemned healthy runs: BIOT-scratch on TUAB seed 1 peaks at
    index 0 (0.8783, with the rest of the curve inside 0.015 of it) and goes on
    to score test AUROC 0.8761 -- the best of its three seeds, and in line with
    BIOT's own published 0.870. Nothing about that run is mis-configured; it
    simply converged inside the first epoch and then drifted down.

    What the rule is *for* is the other case, which is real and is in this
    matrix: CBraMod from scratch on CHB-MIT sits at AUROC 0.5000 for its one or
    two epochs and stops. There the peak at index 0 means the model never left
    its initialisation.

    The two are separated by whether the model ever got above chance, so that is
    what is tested. An early peak on a curve that clears chance by more than the
    seed-to-seed noise is reported (``peaked-first-eval``) but not withheld; an
    early peak at chance, or a curve with no variance at all, still fails.

    This is a narrowing of *when* the rule fires, not of what it rejects: every
    run the old test would have withheld for genuinely not learning is still
    withheld.
    """
    if not val_curve:
        return {"ok": False, "status": "mis-configured", "reason": "empty val curve"}
    peak = int(np.argmax(val_curve))
    best = float(np.max(val_curve))
    flat = float(np.std(val_curve)) == 0.0
    floor = chance_level(metric, num_classes, prevalence)
    # An unknown floor cannot condemn: give the run the benefit of the doubt.
    learned = True if floor is None else best > floor + LEARNED_MARGIN

    if flat:
        ok, status = False, "mis-configured"
        reason = "val curve is flat (model not learning)"
    elif peak != 0:
        ok, status, reason = True, "ok", "ok"
    elif learned:
        ok, status = True, "peaked-first-eval"
        vs = "chance unknown" if floor is None else f"chance {floor:.4f}"
        reason = (f"val peaked at the first evaluation, but reached {best:.4f} "
                  f"vs {vs} -- converged early, not untrained")
    else:
        ok, status = False, "mis-configured"
        reason = (f"val peak at first evaluation and never cleared chance "
                  f"({best:.4f} vs {floor:.4f})")

    if ok:
        dead = dead_run_check(test_metrics, num_classes)
        if dead:
            ok, status, reason = False, "mis-configured", dead

    return {
        "ok": ok,
        "status": status,
        "peak_index": peak,
        "n_evals": len(val_curve),
        "best_val": best,
        "chance": floor,
        "reason": reason,
    }


# A discriminating model can score at chance on *some* metric -- balanced
# accuracy is 0.5000 for any binary model whose threshold is wrong for an
# imbalanced corpus, which is the normal state of affairs on CHB-MIT and says
# nothing about whether it learned. Ranking metrics are threshold-free, so only
# those are consulted here.
DEAD_AUROC = 0.51
DEAD_KAPPA = 0.02


def dead_run_check(test_metrics: dict[str, float] | None,
                   num_classes: int | None) -> str | None:
    """Reason the test scores show no discrimination at all, else None.

    The peak-position test has a blind spot in the other direction from the one
    it is usually accused of: CBraMod from scratch on CHB-MIT peaks at index 37
    of 50 on a non-flat curve, so it passes, yet all three seeds land on test
    AUROC 0.5000 and the late half of the val curve is the handful of quantised
    values (0.2016, 0.3090, 0.5035) a constant predictor produces. The model
    collapsed to one class. Nothing in the shape of the validation curve says
    so; the test AUROC says so immediately.

    So the verdict also looks at what the run actually scored. This only ever
    adds rejections -- it is applied to runs the curve test already passed --
    and it is applied to every group alike.
    """
    if not test_metrics:
        return None
    auroc = test_metrics.get("auroc")
    if auroc is not None and auroc <= DEAD_AUROC:
        return (f"test AUROC {auroc:.4f} is at chance -- the model does not "
                f"discriminate, whatever the val curve did")
    kappa = test_metrics.get("cohen_kappa")
    if kappa is not None and num_classes and num_classes > 2 and kappa <= DEAD_KAPPA:
        return (f"test Cohen's Kappa {kappa:.4f} is at chance -- the model does "
                f"not discriminate, whatever the val curve did")
    return None
