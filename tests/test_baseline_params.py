"""Check group-A parameter counts against the values the xlsx lists.

    python -m tests.test_baseline_params

Group A exists to prove the pipeline reproduces published numbers. That only
works if the architectures are the published ones, and the cheapest sufficient
check that they are is the parameter count -- it is printed in the matrix and it
is sensitive to almost any structural deviation.

Measured in the TUEV configuration (16 channels, 1000 samples, 200 Hz) -- see
the note by the constants for why that is the right reference. A forward pass is run too: a model
that instantiates but cannot consume the real input shape is no use.
"""

from __future__ import annotations

import sys

import torch

from paclock_bench.models.baselines.light_supervised import (
    EXPECTED_PARAMS_M,
    KNOWN_PARAM_DISCREPANCIES,
    REGISTRY,
)

# TUEV reference configuration (16 ch, 5 s @ 200 Hz). The xlsx lists one
# parameter count per model across all sheets; FFCL pins which configuration
# they were measured in -- it is length-dependent and lands on 2.414M here
# against the listed 2.40M, versus 2.465M at TUAB's 10 s.
C, T, FS, NCLS = 16, 1000, 200, 6
TOL = 0.10                       # 10% -- absorbs head width and rounding in the xlsx

rows = []
for name, ctor in REGISTRY.items():
    model = ctor(in_channels=C, seq_len=T, num_classes=NCLS, sample_rate=FS)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    exp = EXPECTED_PARAMS_M[name]
    rel = abs(n - exp) / exp

    model.eval()
    try:
        with torch.no_grad():
            out = model(torch.randn(2, C, T))
        fwd = tuple(out.shape)
        fwd_ok = out.shape == (2, NCLS)
    except Exception as e:                                        # noqa: BLE001
        fwd, fwd_ok = f"{type(e).__name__}: {e}", False

    known = name in KNOWN_PARAM_DISCREPANCIES
    rows.append((name, n, exp, rel, rel <= TOL or known, fwd, fwd_ok, known))

print("=" * 82)
print("%-18s %10s %10s %8s  %-7s %s" % ("model", "actual(M)", "xlsx(M)", "rel", "params", "forward"))
print("-" * 82)
for name, n, exp, rel, ok, fwd, fwd_ok, known in rows:
    status = "KNOWN" if known else ("PASS" if ok else "FAIL")
    print("%-18s %10.3f %10.2f %7.1f%%  %-7s %s %s" % (
        name, n, exp, rel * 100, status,
        "PASS" if fwd_ok else "FAIL", fwd if not fwd_ok else ""))
print("=" * 82)

for name, reason in KNOWN_PARAM_DISCREPANCIES.items():
    print()
    print(f"KNOWN discrepancy -- {name}:")
    for line in reason.split(". "):
        if line.strip():
            print("  " + line.strip().rstrip(".") + ".")

nfail = sum(1 for r in rows if not r[4] or not r[6])
print(f"{len(rows) - nfail}/{len(rows)} models match the published architecture "
      f"(tolerance {TOL:.0%}, TUEV config {C}ch x {T} @ {FS}Hz)")
sys.exit(1 if nfail else 0)
