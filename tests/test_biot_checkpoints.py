"""Verify the official BIOT checkpoints load into the official model code.

    python -m tests.test_biot_checkpoints

Group B runs each foundation model with its own repo's code and weights, so the
first thing to establish is that the released checkpoint actually populates the
released architecture -- a silently partial load would leave randomly
initialised layers and produce numbers that look plausible but mean nothing.
``load_state_dict(strict=False)`` is used deliberately so the missing/unexpected
key lists can be inspected rather than hidden behind an exception.

Constructor arguments come from BIOT's own ``run_binary_supervised.py``:
n_fft = token_size = 200, hop_length = 100, and the model file's defaults
emb_size=256, heads=8, depth=4.
"""

from __future__ import annotations

import sys

import torch

VENDOR = "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/vendor/biot"
sys.path.insert(0, VENDOR)

from model.biot import BIOTClassifier  # noqa: E402

# (checkpoint, channel count the checkpoint was pretrained with)
CKPTS = [
    ("EEG-PREST-16-channels.ckpt", 16),
    ("EEG-six-datasets-18-channels.ckpt", 18),
    ("EEG-SHHS+PREST-18-channels.ckpt", 18),
]

rows = []
for name, nch in CKPTS:
    model = BIOTClassifier(n_classes=2, n_channels=nch, n_fft=200, hop_length=100)
    sd = torch.load(f"{VENDOR}/pretrained-models/{name}", map_location="cpu")
    missing, unexpected = model.biot.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    with torch.no_grad():
        out = model(torch.randn(2, nch, 2000))
    ok = out.shape == (2, 2) and not missing and not unexpected
    rows.append((name, nch, n_params, len(missing), len(unexpected),
                 tuple(out.shape), ok))
    if missing:
        print(f"  {name}: missing {missing[:5]}")
    if unexpected:
        print(f"  {name}: unexpected {unexpected[:5]}")

print("=" * 88)
print("%-40s %5s %9s %9s %11s %s" % (
    "checkpoint", "nch", "params(M)", "missing", "unexpected", "forward"))
print("-" * 88)
for name, nch, n, nm, nu, shape, ok in rows:
    print("%-40s %5d %9.2f %9d %11d %-10s %s" % (
        name, nch, n, nm, nu, str(shape), "OK" if ok else "FAIL"))
print("=" * 88)
nfail = sum(1 for r in rows if not r[6])
print(f"{len(rows) - nfail}/{len(rows)} checkpoints load cleanly into the official model")
sys.exit(1 if nfail else 0)
