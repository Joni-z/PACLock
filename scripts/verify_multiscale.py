"""Multi-scale PAC must not disturb the single-scale path.

Every wave-1..5 number was produced by the single-scale code. A tokeniser
change that was only *mathematically* identical already moved an ISRUC result
by 13.4 seed standard deviations (docs/PERF.md), so "equivalent" is not good
enough here: with one window the new code has to be **bit-identical**, or the
whole search restarts again.

It should be, by construction -- `scale_proj` is allocated only when there is
more than one window, so single-scale RNG consumption, parameter count and
state_dict keys are untouched, and the per-scale loop runs exactly once with
the same operations in the same order. This checks that rather than trusting it.
"""

from __future__ import annotations

import torch

from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.models.paclock.frontend import _tri_pre_ms as ref

B, C, T, NB, D, PATCH = 4, 16, 1000, 8, 128, 100   # P=10, so P_pac in {10,5,2,1}
torch.manual_seed(0)
x = torch.randn(B, C, T)

print("=" * 70)
print("1. single scale must be BIT-identical to the pre-multiscale code")
print("=" * 70)
ok = True
for mode in ["raw", "pac_interaction"]:
    for pac_len in [None, 100, 200, 500]:
        kw = dict(n_bands=NB, hidden_dim=D, sample_rate=200, kernel_size=201,
                  patch_len=PATCH, tokenizer_mode=mode, pac_patch_len=pac_len)
        torch.manual_seed(1)
        old = ref.TriAxialFrontend(**kw).eval()
        torch.manual_seed(1)
        new = TriAxialFrontend(**kw).eval()

        same_keys = set(old.state_dict()) == set(new.state_dict())
        same_init = all(torch.equal(a, b) for a, b in
                        zip(old.state_dict().values(), new.state_dict().values()))
        with torch.no_grad():
            o_tok, o_cp = old(x)[:2]
            n_tok, n_cp = new(x)[:2]
        bit = torch.equal(o_tok, n_tok) and torch.equal(o_cp, n_cp)
        ok &= same_keys and same_init and bit
        print("  mode=%-16s pac_patch_len=%-5s keys=%s init=%s tokens_bit_identical=%s"
              % (mode, pac_len, same_keys, same_init, bit))

print()
print("=" * 70)
print("2. multi-scale runs, and reduces to single scale for a 1-element list")
print("=" * 70)
kw = dict(n_bands=NB, hidden_dim=D, sample_rate=200, kernel_size=201,
          patch_len=PATCH, tokenizer_mode="pac_interaction")
torch.manual_seed(1)
one = TriAxialFrontend(pac_patch_len=200, **kw).eval()
torch.manual_seed(1)
lst = TriAxialFrontend(pac_patch_len=[200], **kw).eval()
with torch.no_grad():
    a = one(x)[0]
    b = lst(x)[0]
print("  [200] == 200 : bit-identical=%s" % torch.equal(a, b))
ok &= torch.equal(a, b)

n_par_1 = sum(p.numel() for p in one.parameters())
for wins in [[100, 200], [100, 200, 500], [200, 500, 1000]]:
    torch.manual_seed(1)
    ms = TriAxialFrontend(pac_patch_len=wins, **kw).eval()
    with torch.no_grad():
        tok, cp, hz = ms(x)[:3]
    extra = sum(p.numel() for p in ms.parameters()) - n_par_1
    fin = torch.isfinite(tok).all().item()
    print("  windows=%-18s tokens=%s finite=%s  +%d params (%.3fM total)"
          % (wins, tuple(tok.shape), fin, extra,
             sum(p.numel() for p in ms.parameters()) / 1e6))
    ok &= fin and tuple(tok.shape) == (B, C, NB, T // PATCH, D)

# a window that does not divide the token grid must be refused, not silently wrong
try:
    TriAxialFrontend(pac_patch_len=[100, 300], **kw)(x)
    print("  FAIL: pac_patch_len=300 with P=10 should have raised")
    ok = False
except ValueError as e:
    print("  ok   non-dividing window refused: %s" % str(e)[:60])

# gradients flow through every scale
torch.manual_seed(1)
ms = TriAxialFrontend(pac_patch_len=[100, 200], **kw)
ms(x)[0].square().sum().backward()
g = ms.scale_proj.weight.grad
print("  scale_proj grad finite=%s  nonzero=%s" % (torch.isfinite(g).all().item(),
                                                   bool((g != 0).any())))
ok &= torch.isfinite(g).all().item() and bool((g != 0).any())

print()
print("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***")
raise SystemExit(0 if ok else 1)
