"""Gate: what actually transfers from the duplex 60k checkpoint at patch_len 50.

The finetune wave runs at patch_len=50 while pretraining ran at 200, so the
three patch_len-dependent tokenizers cannot transfer. They are excluded by
name in configs/pretrain_ft/*_duplex_pt.yaml rather than left to the loader's
silent shape check -- docs/FINDINGS.md records that silent drop as the reason
"pretrained" once quietly meant "pretrained encoder, tokenizer relearned".
This asserts the split is exactly what the configs claim, and in particular
that duplex's THIRD tokenizer (frontend.tokenizer, which feeds the fused rows
and does not exist in pac_interaction) is in the excluded set and not in the
silently-dropped one.

    python -m scripts.verify_duplex_transfer [checkpoint]
"""
import sys

import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.build import load_pretrained_backbone

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "pretrain_runs_60k/pretrain-duplex_base/checkpoint.pt"
CFG = "configs/pretrain_ft/chbmit_duplex_pt.yaml"
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


cfg = yaml.safe_load(open(CFG))
mk = dict(cfg["model_kwargs"])
mk.update(num_classes=cfg["num_classes"], sample_rate=cfg["sample_rate"],
          dataset=cfg["dataset"])
model = build_model({"model": "paclock", "num_classes": cfg["num_classes"],
                     "sample_rate": cfg["sample_rate"], "dataset": cfg["dataset"],
                     "model_kwargs": mk}, input_shape=(16, 2000))

before = {k: v.clone() for k, v in model.state_dict().items()}
rep = load_pretrained_backbone(model, CKPT,
                               exclude=tuple(cfg.get("checkpoint_exclude", ())))
after = model.state_dict()

print("=== report")
print("  loaded          : %d" % len(rep["loaded"]))
print("  skipped(shape)  : %d  %s" % (len(rep["skipped_shape"]), rep["skipped_shape"]))
print("  skipped(exclude): %d  %s" % (len(rep["skipped_excluded"]), rep["skipped_excluded"]))

print("\n=== 1. the three patch_len tokenizers are EXCLUDED, not shape-dropped")
for t in ("frontend.tokenizer.weight", "frontend.phase_tokenizer.weight",
          "frontend.amplitude_tokenizer.weight"):
    check("%s excluded" % t,
          any(e.startswith(t.rsplit(".", 1)[0]) for e in rep["skipped_excluded"]))
check("nothing dropped for shape", len(rep["skipped_shape"]) == 0,
      "" if not rep["skipped_shape"] else "unexpected: %s" % rep["skipped_shape"])

print("\n=== 2. the backbone that SHOULD transfer did")
must = ["frontend.sinc.low_hz_", "frontend.sinc.band_hz_", "frontend.amplitude_scale",
        "frontend.fusion_beta", "frontend.interaction_gate", "band_pe.emb.weight"]
for k in must:
    check("%s loaded" % k, k in rep["loaded"])
enc = [k for k in rep["loaded"] if k.startswith("encoder.")]
enc_model = [k for k in after if k.startswith("encoder.")]
check("every encoder tensor loaded", len(enc) == len(enc_model),
      "%d/%d" % (len(enc), len(enc_model)))

print("\n=== 3. loaded tensors actually changed; excluded ones did not")
changed = [k for k in rep["loaded"] if not torch.equal(before[k], after[k])]
check("loaded tensors differ from init", len(changed) > 0.8 * len(rep["loaded"]),
      "%d/%d changed" % (len(changed), len(rep["loaded"])))
for t in ("frontend.tokenizer.weight", "frontend.phase_tokenizer.weight"):
    check("%s untouched" % t, torch.equal(before[t], after[t]))
check("head is fresh", torch.equal(before["head.proj.weight"], after["head.proj.weight"]))
check("spatial_pe is fresh (montage is corpus-specific)",
      all(torch.equal(before[k], after[k]) for k in after if k.startswith("spatial_pe.")))

print("\n=== 4. forward still runs")
model.eval()
with torch.no_grad():
    y = model(torch.randn(2, 16, 2000))
check("forward finite", torch.isfinite(y).all().item(), str(tuple(y.shape)))

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
