# paper_drafts/v2026-09-11b — round-2 revisions (Overleaf 296050d, 18 pages, 0 undefined refs)

- "removing/switching off the coupling content" -> "compared with a waveform-only tokenizer, the duplex tokenizer improves" (abstract, contribution 2, §5.2, conclusion); Table 3 columns duplex / waveform-only
- §5.2 forward reference to non-existent controls replaced by an explicit "not in this version" statement
- Table 4: Intervention / Control columns; caption "four model families, five encoder configurations"; interpretation per intervention, no spectral-pathway attribution
- Leftover strong claims removed: method intro ("enter only through the token"), §3.3 title and paragraph heads ("dedicated frequency attention" / "folded attention"), "never starved", "coupling and nothing else", "redundant"
- Eq. (1): complex first moment M_ij defined, MVL = |M_ij|, preferred phase = angle(M_ij)
- Prop. 3: epsilon guard on |u_j| and on the alpha column sum stated; equality holds where |u_j| >= eps
- Prop. 4 removed; the two gate facts are a design note
- §3.4 pretraining objective moved to Appendix (app:objective); "evaluated to convergence" removed; "comparison of objectives" removed
- Related work: gate-value promise removed; setup: auxiliary-metrics promise removed
- Every numeric table cell regenerated from run files (results/cells_2026-09-11.json; sample std, one rounding rule): main table, arch controls, transplant tables, appendix coupling / pretraining / matrix
