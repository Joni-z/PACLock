# Paper reporting update — 2026-09-13

Title: **Cross-Frequency Relations as Token Content for EEG Representation Learning**.

The paper keeps the existing 2.7M folded encoder as a reference; it does not present ongoing factorized pilots as completed results or as a frozen final architecture. The story centers on complementarity between local waveform and cross-band relational content, with the historical tri-axial coupling-only/duplex tradeoff reported explicitly.

## Reproduction

From a directory containing the saved `runs/` tree:

```bash
python3 /path/to/PACLock/scripts/gen_tables.py --paper-dir /path/to/ICLR2027-Paper
```

The generator requires Python 3 and NumPy and reads only saved run artifacts. Run it locally or in an appropriate compute environment, not by importing model training dependencies on a cluster login node. In addition to result.json files, the per-class diagnostics require TUEV `test_scores.npz` for folded raw/duplex seed 0 and earlier tri-axial raw/duplex seeds 0,1,2. Missing prediction artifacts cause an explicit failure.

It emits all 21 numerical model-result tables and audit JSON under `results/tables/`. `--paper-dir` replaces exactly one matching table label per generated table, preserving surrounding prose. The corpus-description table contains protocol facts rather than experimental results and is retained separately. Re-run the generator when run results change, then review numerical claims in prose.

- Mean and sample SD for >=3 seeds; count marked for fewer.
- Paired effects use shared seed identifiers, not unmatched means.
- A recorded diagnostic failure marks the entire aggregate cell unavailable for ranking; raw values and flags are retained. Never improve a mean by dropping only its bad seed.
- Budget-stopped sources are marked and excluded from bold rankings, not silently treated as equal-budget completions.
- Class metrics are recomputed from saved logits/labels.
- `table_audit.json` records source hashes, raw metrics, diagnostic status and per-class calculations.

The September 13 snapshot contains 895 source artifacts actually consumed by tables (887 result files and 8 prediction archives). The local input snapshot contains other experiments that are not selected into these reference tables.

## Scientific corrections

- Refresh all available seed aggregates and explicitly disclose missing TUEV seed 1.
- Correct CBraMod additive CHB-MIT effect from a positive seed-0 result to a negative three-seed paired mean.
- Distinguish 8-row waveform reference from matched 16-row own-phase/measured-alignment controls.
- Describe learned band-index embeddings and absence of a separate frequency-attention sub-layer accurately.
- Correct 50 samples to .5 s on 100 Hz Sleep-EDF and .25 s at 200 Hz; distinguish 1e-8 weight/fallback guard from 1e-6 rotation guard.
- Qualify invariance outside fallback and amplitude preservation outside normalization guard.
- Add SleepPACNet and FAPEX; do not claim PAC features or dual branches themselves as novelty.
- Distinguish earlier 60k amplitude/PAC pretraining, separate 150k folded raw-patch pretraining, and CBraMod-tokenizer pretraining. Disclose the mixed-sampling-rate limitation of the folded pretraining experiment.
- Retain descriptive results and limitations without physiological-causality, universal SOTA, or validated foundation-model claims.

Paper publication must use `./push_overleaf.sh "message"` on the MacBook paper repository. Verify the remote HEAD after the script because its push pipeline can mask an error. Preserve unrelated untracked manuscript helper files.

## Verification receipt

Overleaf commit `b7ae616` compiled successfully to 17 pages, with the conclusion on main-text page 9. No undefined references or overfull boxes remained after the final build. All 895 consumed source hashes matched the local snapshot; all 21 generated result tables matched their manuscript blocks exactly; paired-seed and saved class-count checks passed.
