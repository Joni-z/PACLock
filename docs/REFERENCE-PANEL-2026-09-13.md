# External reference panel — 2026-09-13

These are verified reference results, **not an exhaustive current SOTA ranking**.
They must not be used as thresholds on PACLock's different validation cohort.
Keep the metric, protocol and paper version with every number.

| Reference | Reported TUEV result | Interpretation |
|---|---|---|
| [CBraMod v3, Table 13](https://arxiv.org/html/2412.07236v3#A5.T13) | Kappa .6772 ± .0096; balanced accuracy .6671 ± .0107 | The separate pretraining-excluding-TUEV variant reports kappa .6744 ± .0121. These supersede the unversioned CBraMod numbers in the August archive. |
| [REVE v1, Table 15](https://arxiv.org/html/2510.21585v1#A3.T15) | Kappa .6783 ± .0199; balanced accuracy .6759 ± .0229 | Base fine-tuning result; its dataset appendix specifies the BIOT processing and training/validation split conventions. |
| [Uni-NTFM v2, Table 1](https://arxiv.org/html/2509.24222v2#S3.T1) | Kappa .7030 ± .0148; balanced accuracy .6991 ± .0170 | Large, pretrained and fine-tuned variant; five trials. The table prints scores multiplied by 100. This is a reference, not an independently established current SOTA ceiling. |
| [NeurIPT v1, Table 24](https://arxiv.org/html/2510.16548v1#A5.T24) | Kappa .6970 ± .0185 | Its appendix uses 64 Hz, .1–30 Hz conditioning and an 80/20 training-subject split; audit cohort identity before comparison. |
| [CodeBrain, ICLR 2026 Table 7](https://openreview.net/pdf?id=msJgEkjwh5) | Kappa .6912 ± .0101; excluding variant .6838 ± .0291 | Preserve which pretraining variant is being compared. |
| [TFM-Tokenizer v3, Table 1](https://arxiv.org/html/2502.16060v3) | Kappa .6189 ± .0302 with multi-dataset pretraining | Its Appendix C also reports .6591 ± .0106 for one CHB-MIT pretraining arrangement; this is a different setting. |
| [BrainOmni v3, Table 2](https://arxiv.org/html/2505.18185v3) | Balanced accuracy .622 ± .028 for base | This is not kappa. Its preprocessing is 256 Hz; TUH evaluation rotates training/validation folds while retaining official eval data. |

TFM's appendix specifies 16 bipolar channels, 200 Hz, .1–75 Hz filtering,
50 Hz notch and five seeds. These details matter when interpreting its
reference values against this repository's differently conditioned cohorts.
[Source](https://arxiv.org/html/2502.16060v3)

## Consequences for model selection

1. First compare candidate/control curves on identical validation examples,
   recipe, progress, runtime and seeds. Continue the currently registered pairs.
2. After a shortlist emerges, replicate across seeds and complementary corpora;
   retain one architecture policy rather than per-dataset winning branches.
3. For final >10-dataset evidence, audit actual held-out cohorts and strong
   baselines per task. Full recipe completion, source identity, selected metric,
   model size and pretraining exposure/compute must be reported explicitly.
4. A winning mean alone does not establish meaningful superiority. Report seed
   variability and class behavior, especially on imbalanced TUEV/CHB/TUSZ.

## Novelty boundary

[CodeBrain](https://arxiv.org/abs/2506.09110) already separates temporal and
frequency tokenization. Thus a generic two-branch or decoupled-tokenizer claim
is insufficient as PACLock's distinction. Our proposed contribution must be
the specific coupling-conditioned token construction, preservation of local
waveform information, and controlled evidence for the contribution of coupling.
The lane-scale screen is an implementation test, not a novelty claim by itself.

No new GPU runs were launched for this reference check. Historical test scores
were not used to choose a candidate or change a cohort.
