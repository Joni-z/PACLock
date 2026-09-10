# paper_drafts/v2026-09-11c — main text on one model (Overleaf 9bfa2b3, 18 pages)

- Tri-axial encoder removed from the main text: main table has one CroFreMo column; four-encoder table has four rows
  (CroFreMo, CBraMod, LaBraM, REVE); architectural controls, 12-corpus coupling table, per-class (tri-axial) and
  pretraining tables live in Appendix "An earlier design: the tri-axial encoder" (app:triaxial)
- Method §3.3 now describes only the folded-attention encoder (+ the CBraMod insertion)
- New main-text Table: TUEV per class for the FINAL model, waveform-only vs duplex, with counts and P/R/F1
  (macro-F1 0.48 -> 0.58; GPED 0.59 -> 0.83, PLED 0.50 -> 0.58, SPSW 0.07 -> 0.20; every class improves)
- Seeds paragraph and §5.2 scale sentence reference the earlier design's std only as a recipe reference
- Main-text tables now: main, duplex vs waveform-only, per-class, design variants, four-encoder, CBraMod rows, hosts
