"""Regression check for the PACLock frontend.

    sbatch slurm/run.slurm scripts.verify_frontend

Self-contained on purpose. The two scripts this replaces each diffed against a
frozen copy of the previous frontend, so both stopped working the moment those
copies were deleted, and neither could be run again by anyone changing the code
later. Everything below is checked against a property instead of against a
snapshot, so it keeps working.

What it guards, and why each one cost something to learn:

1. `_patch_project` is `Conv1d(1, K, k=P, stride=P)`. Replacing that convolution
   with a GEMM is what made training 9.4x faster (docs/PERF.md) -- forced
   determinism made MIOpen pick an atomics-free backward-weights kernel for the
   in_channels=1 shape. If someone "simplifies" it back, this fails.
2. A one-element `pac_patch_len` list is bit-identical to the scalar. Multi-scale
   must not perturb the single-scale path: a mathematically identical frontend
   change once moved an ISRUC result by 13.4 seed standard deviations, so every
   result in the repo would have to be re-run.
3. The pretraining outputs (`return_amp_target`, `return_pac_vector`) and the
   phase-ablation modes still build and run. These are the paths supervised
   training never touches, so nothing else notices when they break.
"""

from __future__ import annotations

import sys

import torch

from paclock_bench.models.build import build_model, count_params
from paclock_bench.models.paclock.frontend.triaxial import (
    TriAxialFrontend, _patch_project,
)

B, C, T, NB, D, PATCH = 4, 16, 1000, 8, 128, 100
torch.manual_seed(0)
x = torch.randn(B, C, T)
fail = 0


def check(name, fn):
    global fail
    try:
        fn()
        print("  ok    %s" % name)
    except Exception as e:                                   # noqa: BLE001
        fail += 1
        print("  FAIL  %s  -- %s: %s" % (name, type(e).__name__, e))


print("1. the GEMM tokeniser is the convolution it replaced")
for out_ch, bias, lab in [(D, True, "with bias"), (D // 2, False, "no bias")]:
    def _op(out_ch=out_ch, bias=bias):
        conv = torch.nn.Conv1d(1, out_ch, PATCH, stride=PATCH, bias=bias)
        sig = torch.randn(B * C * NB, T)
        want = conv(sig.unsqueeze(1)).transpose(1, 2)
        got = _patch_project(conv, sig)
        err = (want - got).abs().max().item()
        assert err < 1e-4, "max|d| %.3e exceeds fp32 rounding" % err
    check("Conv1d(1,%d,k=%d,s=%d) %s" % (out_ch, PATCH, PATCH, lab), _op)

print()
print("2. one-element pac_patch_len list == the scalar, bit for bit")
kw = dict(n_bands=NB, hidden_dim=D, sample_rate=200, kernel_size=201,
          patch_len=PATCH, tokenizer_mode="pac_interaction")
for w in [PATCH, 200]:
    def _one(w=w):
        torch.manual_seed(1); a = TriAxialFrontend(pac_patch_len=w, **kw).eval()
        torch.manual_seed(1); b = TriAxialFrontend(pac_patch_len=[w], **kw).eval()
        assert set(a.state_dict()) == set(b.state_dict()), "state_dict keys differ"
        with torch.no_grad():
            assert torch.equal(a(x)[0], b(x)[0]), "tokens are not bit-identical"
    check("pac_patch_len=%d vs [%d]" % (w, w), _one)

def _multi():
    torch.manual_seed(1)
    ms = TriAxialFrontend(pac_patch_len=[PATCH, 200, 500], **kw)
    tok = ms(x)[0]
    assert tok.shape == (B, C, NB, T // PATCH, D), tok.shape
    tok.square().sum().backward()
    g = ms.scale_proj.weight.grad
    assert torch.isfinite(g).all() and (g != 0).any(), "scale_proj gets no gradient"
check("multi-scale [%d,200,500] runs and trains" % PATCH, _multi)

def _reject():
    try:
        TriAxialFrontend(pac_patch_len=[PATCH, 300], **kw)(x)
    except ValueError:
        return
    raise AssertionError("a window that does not divide the token grid was accepted")
check("non-dividing window is refused, not silently wrong", _reject)

print()
print("3. the paths only pretraining uses")
for mode in ["raw", "pac_interaction"]:
    def _pre(mode=mode):
        fe = TriAxialFrontend(n_bands=NB, hidden_dim=D, sample_rate=200,
                              kernel_size=201, patch_len=PATCH,
                              tokenizer_mode=mode, return_pac_vector=True)
        tok, cp, hz, pac = fe(x)
        assert pac.is_complex() and pac.shape == (B, C, T // PATCH, NB, NB)
        fe.return_pac_vector = False
        tok, cp, hz, amp = fe(x, return_amp_target=True)
        assert amp.shape == (B, C, NB, T // PATCH) and torch.isfinite(amp).all()
    check("frontend[%s]: amp_target + pac_vector" % mode, _pre)


def _cfg(**over):
    mk = {"arch": "triaxial", "d_model": D, "depth": 6, "n_bands": NB,
          "n_heads": 4, "dropout": 0.2, "kernel_size": 201, "patch_len": PATCH,
          "augmentations": [], "freq_mixer": "attention", "band_pe": "index",
          "tokenizer_mode": "pac_interaction", "pac_token_mode": "measured",
          "interaction_mode": "product", "spatial_pe": "index"}
    mk.update(over)
    return {"model": "paclock", "dataset": "tuev", "num_classes": 6,
            "sample_rate": 200, "model_kwargs": mk}


for over, lab in [({"aux_recon_weight": 0.1}, "aux reconstruction loss"),
                  ({"freq_mixer": "phase"}, "freq_mixer=phase + phase_mode ablations")]:
    def _model(over=over):
        m = build_model(_cfg(**over), (C, T))
        with torch.no_grad():
            assert m(x).shape == (B, 6)
            if over.get("freq_mixer") == "phase":
                for pm in ["magnitude", "scramble"]:
                    assert m(x, phase_mode=pm).shape == (B, 6)
        if over.get("aux_recon_weight"):
            m.train()
            loss = m.crossfreq_aux_loss(x)
            assert loss.ndim == 0 and torch.isfinite(loss)
        return count_params(m)
    check("model: %s" % lab, _model)

print()
print("%d checks failed" % fail)
sys.exit(1 if fail else 0)
