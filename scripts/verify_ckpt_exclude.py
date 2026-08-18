"""Gates for the three-arm pretrained-tokenizer test.

The measurement only means something if the three arms differ in exactly the
intended way, so that is asserted rather than assumed:

  A  scratch                      -- no checkpoint
  B  pretrained, tokenizer loaded -- at patch_len=200 the kernel matches, so the
                                     tokenizer must actually transfer (0 shape skips)
  C  pretrained, tokenizer excluded

  * B must differ from A in the tokenizer AND the encoder
  * C must differ from A in the encoder ONLY -- its tokenizer must be
    bit-identical to a fresh init with the same seed
  * B and C must be bit-identical everywhere EXCEPT the tokenizer

If any of those fail, B - C is not "the pretrained tokenizer's contribution".

    sbatch slurm/run.slurm scripts.verify_ckpt_exclude
"""
import torch

from paclock_bench.models.build import build_model
from paclock_bench.paths import expand

CKPT = "pretrain_runs_60k/pretrain-size_base/checkpoint.pt"
TOK = ("frontend.phase_tokenizer", "frontend.amplitude_tokenizer")

BASE = dict(
    model="paclock", num_classes=2, sample_rate=200, dataset="chbmit",
    model_kwargs=dict(arch="triaxial", d_model=128, depth=6, n_bands=8, n_heads=4,
                      dropout=0.2, kernel_size=201, patch_len=200, pac_patch_len=200,
                      augmentations=[], freq_mixer="attention", band_pe="index",
                      tokenizer_mode="pac_interaction", pac_token_mode="measured",
                      interaction_mode="product", spatial_pe="index"),
)
SHAPE = (16, 2000)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


def build(**over):
    torch.manual_seed(0)
    cfg = {**BASE, **over}
    return build_model(cfg, input_shape=SHAPE)


def sd(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def differing(a, b, prefix=None):
    out = []
    for k in a:
        if prefix and not k.startswith(prefix):
            continue
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            out.append(k)
    return out


print("=== building the three arms", flush=True)
A = sd(build())
B = sd(build(checkpoint=CKPT))
C = sd(build(checkpoint=CKPT, checkpoint_exclude=list(TOK)))

print("\n=== 1. B really transfers the tokenizer at patch_len=200", flush=True)
tok_ba = differing(A, B, TOK)
check("B's tokenizer differs from scratch", len(tok_ba) >= 2, str(tok_ba))

print("\n=== 2. C really does NOT", flush=True)
tok_ca = differing(A, C, TOK)
check("C's tokenizer is identical to scratch", not tok_ca, str(tok_ca))

print("\n=== 3. both load the encoder", flush=True)
enc_ba = differing(A, B, ("encoder.",))
enc_ca = differing(A, C, ("encoder.",))
check("B encoder differs from scratch", len(enc_ba) > 50, f"{len(enc_ba)} tensors")
check("C encoder differs from scratch", len(enc_ca) > 50, f"{len(enc_ca)} tensors")
check("B and C load the SAME encoder", enc_ba == enc_ca,
      f"B {len(enc_ba)} vs C {len(enc_ca)}")

print("\n=== 4. B and C differ ONLY in the tokenizer", flush=True)
bc = differing(B, C)
check("difference confined to the tokenizer",
      bc and all(k.startswith(TOK) for k in bc), str(bc))

print("\n=== 5. forward runs and the arms are not accidentally equal", flush=True)
x = torch.randn(2, *SHAPE)
outs = {}
for nm, over in (("A", {}), ("B", dict(checkpoint=CKPT)),
                 ("C", dict(checkpoint=CKPT, checkpoint_exclude=list(TOK)))):
    m = build(**over).eval()
    with torch.no_grad():
        outs[nm] = m(x)
    check(f"{nm} forward finite", torch.isfinite(outs[nm]).all().item())
check("B != C at the output", not torch.allclose(outs["B"], outs["C"]),
      f"max|diff| {(outs['B'] - outs['C']).abs().max().item():.3e}")

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
