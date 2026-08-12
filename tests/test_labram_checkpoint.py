"""Verify labram-base.pth loads into the official LaBraM architecture.

    python -m tests.test_labram_checkpoint

Same reasoning as the BIOT checkpoint test: a partial load leaves layers random
while still being reported as "pretrained". load_pretrained() raises on any
non-head key mismatch, so reaching the forward pass is itself the assertion.
"""
from __future__ import annotations

import sys

import torch

from paclock_bench.models.foundation.labram_adapter import (
    CH_NAMES, build_labram, count_params,
)

results = []
for n_classes, T in [(2, 2000), (6, 1000)]:      # TUAB 10 s, TUEV 5 s
    try:
        m = build_labram(n_classes=n_classes, n_channels=len(CH_NAMES),
                         pretrained=True).eval()
        with torch.no_grad():
            out = m(torch.randn(2, len(CH_NAMES), T))
        ok = out.shape == (2, n_classes)
        results.append((f"{n_classes}-class T={T}", count_params(m),
                        tuple(out.shape), ok, ""))
    except Exception as e:                        # noqa: BLE001
        results.append((f"{n_classes}-class T={T}", 0, None, False,
                        f"{type(e).__name__}: {e}"))

print("=" * 84)
print("%-20s %11s %-12s %s" % ("config", "params(M)", "forward", "status"))
print("-" * 84)
for name, n, shape, ok, err in results:
    print("%-20s %11.2f %-12s %s" % (name, n, str(shape), "OK" if ok else f"FAIL {err}"))
print("=" * 84)
print(f"channels: {len(CH_NAMES)}  {CH_NAMES[:6]}...")
nfail = sum(1 for r in results if not r[3])
print(f"{len(results)-nfail}/{len(results)} configs load and run")
sys.exit(1 if nfail else 0)
