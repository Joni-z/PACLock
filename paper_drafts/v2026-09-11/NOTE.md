# paper_drafts/v2026-09-11 — category-1 (writing) revisions after the 09-10 review

Overleaf commits 0503c83 + 022f59d. 18 pages, 0 undefined references.
- Title: Cross-Frequency Coupling as Token Content: Encoder-Dependent Benefits for Clinical EEG
- Prop. 2 restricted to j>0 (band 0's interaction token is its own phase feature, not reference-invariant)
- Prop. 4 "exact recoverability" replaced by a gating statement; zero rows still occupy softmax mass -> stated
- Eq. 5: complex h_j laid out as real/imag parts (d real) before gating
- "no energy artifact" sentence removed; Z column positive-scale invariance stated (relative weights + geometry)
- Z described as a patch-resolution coupling statistic (cycle count of the slowest band; Aru et al. spurious CFC)
- Axis-free encoder no longer claimed to be unable to compute cross-band relations
- Framing: benefit depends on encoder inductive bias; "necessary / exactly when / cannot compute" removed
- Four-encoder table gains CHB-MIT/TUEP columns (CBraMod CHB-MIT +0.124 shown); d192 vs d256 "sensitive to capacity"
- Setup: seeds paragraph (tri-axial std is a reference, not a substitute); baseline-fairness limitation; appendix promises removed
- Pretraining demoted to appendix (scratch vs pretrained table), probe/label-fraction claims withdrawn
- Appendix tri-axial coupling table gets the TUAB row
