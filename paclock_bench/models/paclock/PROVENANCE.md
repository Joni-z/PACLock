# Provenance

Vendored verbatim from `Joni-z/PACLock` @ the NVIDIA-cluster codebase
(`models/`), which is where this architecture was developed and validated.

Kept as-is so the architecture is bit-identical to the one that produced the
published-in-AGENT.md ablations. Only two things were removed, both out of scope
for the current baseline stage:

* `pretrain.py` -- MAE pretraining
(augment.py is retained: build.py imports it. Baseline configs pass
`augmentations: []`, so it is a no-op.)

ROCm compatibility of this code was verified on MI210 before vendoring; see
`tests/test_rocm_compat.py`. The FFT/complex/BF16 risks flagged in the original
`BIG_CLUSTER_HANDOFF.md` sec.7 do not materialise -- no AMD-specific rewrite
was needed.

**Do not** copy `scripts/preprocess_*.py` from that repo: it follows the BIOT
protocol, while this benchmark is frozen on the CBraMod protocol. See
`docs/PROTOCOLS.md`.
