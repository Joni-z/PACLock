"""Exercise the paths a pretraining run needs, not just the classifier forward.

The tokenisers were rewritten from Conv1d to a GEMM (docs/PERF.md). The
classifier path is covered by scripts/verify_patch_project.py, but the frontend
has two more outputs that only pretraining and the phase mixer ask for, and both
are downstream of the tokenisers:

  * ``return_amp_target=True`` -> the masked band-amplitude regression target
    that models/pretrain.py reconstructs, and that build.py's
    ``crossfreq_aux_loss`` uses as a supervised auxiliary;
  * ``return_pac_vector=True`` -> the complex PAC edge tensor the
    ``freq_mixer="phase"`` encoder consumes, including its magnitude/scramble
    phase-ablation modes.

A deliverable that goes into pretraining has to have these checked, and the raw
tokenizer_mode has to be checked too even though every current config uses
pac_interaction -- the raw branch is where the P index moved from shape[-1] to
shape[1].
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from paclock_bench.models.build import build_model, count_params        # noqa: E402
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend  # noqa: E402

B, C, T = 2, 16, 1000
NB, D, PATCH = 8, 128, 200
P = T // PATCH
fail = 0


def check(name, fn):
    global fail
    try:
        fn()
        print("  ok   %s" % name)
    except Exception as e:                                   # noqa: BLE001
        fail += 1
        print("  FAIL %s  %s: %s" % (name, type(e).__name__, e))


x = torch.randn(B, C, T)

for mode in ["raw", "pac_interaction"]:
    def _amp_target(mode=mode):
        fe = TriAxialFrontend(n_bands=NB, hidden_dim=D, sample_rate=200,
                              kernel_size=201, patch_len=PATCH,
                              tokenizer_mode=mode)
        tokens, coupling, hz, amp_target = fe(x, return_amp_target=True)
        assert tokens.shape == (B, C, NB, P, D), tokens.shape
        assert amp_target.shape == (B, C, NB, P), amp_target.shape
        assert torch.isfinite(amp_target).all()
    check("frontend[%s] return_amp_target -> (B,C,nb,P)" % mode, _amp_target)

    def _pac_vec(mode=mode):
        fe = TriAxialFrontend(n_bands=NB, hidden_dim=D, sample_rate=200,
                              kernel_size=201, patch_len=PATCH,
                              tokenizer_mode=mode, return_pac_vector=True)
        tokens, coupling, hz, pac = fe(x)
        assert pac.shape == (B, C, P, NB, NB), pac.shape
        assert pac.is_complex()
    check("frontend[%s] return_pac_vector -> complex (B,C,P,nb,nb)" % mode, _pac_vec)


def _cfg(**over):
    mk = {"arch": "triaxial", "d_model": D, "depth": 6, "n_bands": NB,
          "n_heads": 4, "dropout": 0.2, "kernel_size": 201, "patch_len": PATCH,
          "augmentations": [], "freq_mixer": "attention", "band_pe": "index",
          "tokenizer_mode": "pac_interaction", "pac_token_mode": "measured",
          "interaction_mode": "product", "spatial_pe": "index"}
    mk.update(over)
    return {"model": "paclock", "dataset": "tuev", "num_classes": 6,
            "sample_rate": 200, "model_kwargs": mk}


for over, label in [
    ({}, "base"),
    ({"patch_len": 100}, "patch100"),
    ({"d_model": 256, "depth": 8, "n_heads": 8}, "size_large"),
    ({"tokenizer_mode": "raw"}, "raw tokenizer"),
    ({"freq_mixer": "phase"}, "freq_mixer=phase (uses pac_vector)"),
    ({"interaction_mode": "concat"}, "interaction=concat"),
    ({"pac_token_mode": "magnitude"}, "pac_token_mode=magnitude"),
    ({"aux_recon_weight": 0.1}, "aux_recon_weight>0 (pretrain-style aux)"),
]:
    def _model(over=over, label=label):
        m = build_model(_cfg(**over), (C, T))
        with torch.no_grad():
            out = m(x)
        assert out.shape == (B, 6), out.shape
        if over.get("aux_recon_weight"):
            m.train()
            loss = m.crossfreq_aux_loss(x)
            assert loss.ndim == 0 and torch.isfinite(loss), loss
        print("       %-40s %.3fM params" % (label, count_params(m)))
    check("model[%s]" % label, _model)

# phase-ablation modes travel through the same tokens
def _phase_modes():
    m = build_model(_cfg(freq_mixer="phase"), (C, T))
    with torch.no_grad():
        for pm in ["normal", "magnitude", "scramble"]:
            assert m(x, phase_mode=pm).shape == (B, 6)
check("phase_mode normal/magnitude/scramble", _phase_modes)

print("\n%d checks failed" % fail)
sys.exit(1 if fail else 0)
