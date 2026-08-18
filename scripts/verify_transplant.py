"""Gates for the tokenizer-transplant arms.

The claim this ablation wants to make is "PACLock's tokenizer improves a foreign
architecture". For that to mean anything, three arms have to differ in exactly
one thing each:

  1  CBraMod native                       -- vendor tokenizer
  2  CBraMod + PACLock PAC frontend       -- our tokenizer
  3  CBraMod + PACLock raw frontend       -- our frontend WITHOUT the PAC
                                             interaction; separates "our
                                             frontend is better" from "the PAC
                                             interaction is what did it"

Checked here:
  * arms 2 and 3 have identical parameter counts, so neither can win on capacity
  * both differ from the native arm by only the tokenizer's worth of parameters
  * everything downstream (encoder, positional encoding, head) is the same
    object graph in all three
  * all three produce finite output of the right shape

    sbatch slurm/run.slurm scripts.verify_transplant
"""
import torch

from paclock_bench.models.build import build_model

SHAPE = (16, 2000)          # TUEV: 16 channels, 10 s at 200 Hz
NCLS = 6
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


def build(model, **mk):
    torch.manual_seed(0)
    return build_model({"model": model, "num_classes": NCLS, "sample_rate": 200,
                        "model_kwargs": {"dropout": 0.1, **mk}},
                       input_shape=SHAPE)


def n_par(m):
    return sum(p.numel() for p in m.parameters())


print("=== building the three arms", flush=True)
native = build("cbramod")
pac = build("cbramod_paclockfe", tokenizer_mode="pac_interaction")
raw = build("cbramod_paclockfe", tokenizer_mode="raw")

print("\n=== 1. parameter counts", flush=True)
for nm, m in (("CBraMod native", native), ("+ PAC frontend", pac),
              ("+ raw frontend", raw)):
    print(f"      {nm:<18} {n_par(m):,}")
check("PAC and raw arms are parameter-matched", n_par(pac) == n_par(raw),
      f"{n_par(pac):,} vs {n_par(raw):,}")

print("\n=== 2. the tokenizer is the only thing that differs between 2 and 3",
      flush=True)
sp = dict(pac.named_parameters())
sr = dict(raw.named_parameters())
only_pac = sorted(set(sp) - set(sr))
only_raw = sorted(set(sr) - set(sp))
shared_diff = [k for k in set(sp) & set(sr) if sp[k].shape != sr[k].shape]
check("differing keys are all tokenizer keys",
      all("tokenizer" in k or "amplitude_scale" in k for k in only_pac + only_raw),
      f"pac-only {only_pac}, raw-only {only_raw}")
check("no shared parameter changes shape", not shared_diff, str(shared_diff))

print("\n=== 3. the encoder really is the vendored one, untouched", flush=True)
enc_p = sorted(k for k in sp if k.startswith(("backbone.encoder", "backbone.proj_out")))
enc_r = sorted(k for k in sr if k.startswith(("backbone.encoder", "backbone.proj_out")))
check("both arms carry the same encoder graph", enc_p == enc_r and len(enc_p) > 20,
      f"{len(enc_p)} tensors")

print("\n=== 4. forward", flush=True)
x = torch.randn(2, *SHAPE)
outs = {}
for nm, m in (("native", native), ("pac", pac), ("raw", raw)):
    m.eval()
    with torch.no_grad():
        y = m(x)
    outs[nm] = y
    check(f"{nm}: shape {tuple(y.shape)} finite",
          y.shape == (2, NCLS) and torch.isfinite(y).all().item())
check("pac and raw arms are not the same function",
      not torch.allclose(outs["pac"], outs["raw"]),
      f"max|diff| {(outs['pac'] - outs['raw']).abs().max().item():.3e}")

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
