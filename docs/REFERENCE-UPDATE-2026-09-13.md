# Targeted reference update, 2026-09-13

This is a small primary-source check during the running candidate gates, not
an exhaustive SOTA survey. Published test results must not be ranked against
our live validation scores. No new experiment or pretraining allocation was
admitted from this check.

## Relevant evidence and limits

- **NeuroRVQ, v4 (17 May 2026).** The current version uses multiscale temporal
  convolutions, hierarchical RVQ, and a Fourier phase loss respecting circular
  angles, alongside amplitude and temporal reconstruction terms. Generic
  claims about preserving frequency information or using phase in a tokenizer
  therefore do not distinguish PACLock. Our proposed distinction needs
  explicit cross-frequency interaction coordinates and controlled evidence
  for their complementarity with waveform information. This is our inference
  about positioning, not a claim that our implementation already wins.
  [Current paper](https://arxiv.org/html/2510.13068v4).

- **MTDP, v1 (4 March 2026).** This work replaces masked reconstruction
  pretraining with a two-stage teacher-fusion/distillation procedure and
  evaluates a CBraMod architecture on twelve datasets. Its downstream setup
  uses 50-epoch joint fine-tuning and validation checkpoint selection. This
  makes the pretraining target a relevant alternative research direction,
  but does not justify importing another expensive pipeline before our
  current transfer gate finishes. Table 7's TUEV middle-column value is 71.03
  for the full-data variant, while its header says AUC-PR and the methods
  section defines the multiclass middle metric as Cohen's kappa. Treat the
  metric assignment as unresolved, not a verified new SOTA threshold. The
  PDF screenshot check failed; no visual verification is claimed. The paper
  also acknowledges teacher-feature storage and I/O costs.
  [Paper, sections 5.1.4 and C, Table 7](https://arxiv.org/html/2603.04478v1).

- **Parameter-efficient SSL adaptation, v1 (25 August 2026).** The comparison
  is adaptation followed by a linear probe versus a frozen-model linear
  probe. It is not a comparison with full supervised fine-tuning. The study
  updates a small encoder subset using unlabeled target-training data and
  investigates data/repetition under a fixed sample-epoch product. This
  supports considering a bounded adaptation gate if needed; it does not
  establish that adapting our current candidate will improve it or that
  additional TUEG pretraining is warranted. Do not transfer its data-fraction
  findings into a universal PACLock budget rule.
  [Paper, sections 3.1–3.3](https://arxiv.org/html/2608.24727v1).

- **ZIPBrain, v1 (7 August 2026).** Table 3 reports LaBraM TUEV kappa .6774
  with 20% token compression versus .6616 for its uncompressed model. These
  are a particular backbone/compression comparison, not a universal SOTA
  result. The method selects representation-space hyperparameters by model
  and task. Arbitrary token merging also does not directly preserve the
  electrode/band/patch grid expected by PACLock's triaxial encoder. No merging
  change is proposed without checking that structural constraint.
  [Paper, Table 3 and representation-space discussion](https://arxiv.org/html/2608.07033v1).

- **DIVER-1 version change.** The current arXiv record is v3 (23 May 2026),
  titled *Scaling Intracranial EEG Foundation Models for Transferable
  Representations*, with Neuroprobe and MAYO as its two main benchmarks.
  The older v1 scalp-EEG/TUEV tables must not be cited as current-v3 results.
  Keep an explicit version if using those historical fine-tuning observations.
  [Version history](https://arxiv.org/abs/2512.19097),
  [current paper](https://arxiv.org/html/2512.19097v3).

## Candidate decision

The existing bounded TUEV scale pair and 140k coupling-pretraining transfer
gate remain the next decision inputs. These references do not establish a
reason to resume paid B2 pretraining or choose a final architecture now.
The final candidate still needs one declared architecture policy, multiple
seeds, and competitive evidence across more than ten datasets. A novelty
claim must distinguish cross-frequency interactions from phase-aware signal
reconstruction and demonstrate their value beyond a generic two-content
tokenizer.
