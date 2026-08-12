"""Sanity-check the CBraMod-architecture / PACLock-tokenizer ablation before
spending any training compute on it.

    sbatch slurm/run.slurm scripts.verify_cbramod_paclockfe
"""

from __future__ import annotations

import sys

import torch

from paclock_bench.models.build import build_model, count_params
from paclock_bench.models.foundation.cbramod_adapter import build_cbramod

fail = 0


def check(name, fn):
    global fail
    try:
        fn()
        print("  ok    %s" % name)
    except Exception as e:                                   # noqa: BLE001
        fail += 1
        print("  FAIL  %s  -- %s: %s" % (name, type(e).__name__, e))


CASES = [
    # dataset-like name, input_shape, num_classes, is_sequence
    ("tuev",  (16, 1000), 6, False),
    ("tuab",  (16, 6000), 2, False),
    ("isruc", (20, 6, 6000), 5, True),      # (seq_len, ch, T) -- ISRUC's 3-D shape
]

for name, shape, K, is_seq in CASES:
    def _shape(name=name, shape=shape, K=K, is_seq=is_seq):
        cfg = {"model": "cbramod_paclockfe", "num_classes": K, "dataset": name,
              "sample_rate": 200}
        C = shape[1] if is_seq else shape[0]
        T = shape[2] if is_seq else shape[1]
        m = build_model(cfg, shape)
        x = torch.randn(2, *shape)
        out = m(x)
        want = (2, shape[0], K) if is_seq else (2, K)
        assert tuple(out.shape) == want, "got %s want %s" % (tuple(out.shape), want)
        # gradient actually reaches the frontend's own parameters (the sinc
        # filterbank, the tokenizers), not just CBraMod's downstream weights
        out.square().sum().backward()
        fe_params = list(m.backbone.frontend.parameters())
        assert fe_params, "frontend has no parameters at all"
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in fe_params), "no gradient reached the frontend"
        p = count_params(m)
        print("       %-6s in=%-20s out=%-14s %.3fM params" % (
            name, tuple(x.shape), tuple(out.shape), p))
    check("cbramod_paclockfe[%s]" % name, _shape)

print()
print("=== compare backbone param counts: native CBraMod vs PACLock-tokenizer ===")
for name, shape, K, is_seq in CASES:
    C = shape[1] if is_seq else shape[0]
    T = shape[2] if is_seq else shape[1]
    native = build_cbramod(K, C, T, pretrained=False, sequence=is_seq)
    swapped = build_model({"model": "cbramod_paclockfe", "num_classes": K,
                          "dataset": name, "sample_rate": 200}, shape)
    p_native = sum(p.numel() for p in native.backbone.parameters()) / 1e6
    p_swapped = sum(p.numel() for p in swapped.backbone.parameters()) / 1e6
    print("  %-6s native backbone %.3fM   paclockfe backbone %.3fM   (diff %+.3fM)"
        % (name, p_native, p_swapped, p_swapped - p_native))

print()
print("%d checks failed" % fail)
sys.exit(1 if fail else 0)
