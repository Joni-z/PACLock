# TUEV investigation — 2026-09-13

## Final action

Stopped four running **200 Hz f3** process trees: TUEV, CHB-MIT, TUSZ and ISRUC. The full FFT response of the implemented lowest filter has its maximum at **0 Hz**, with DC gain **70.156**, despite nominal cutoffs .5–.935 Hz. This is a concrete failure of the intended low-frequency bandpass contrast. It does not make the observed classification scores fake, but prevents treating this implementation as a clean low-frequency-coverage candidate. The already-completed TUAR f3 result is preserved and marked out of candidate selection by this diagnosis.

The 100 Hz Sleep-EDF f3 is retained as a diagnostic arm: the same construction peaks near .696 Hz there, so the 200 Hz finding does not justify applying the same stop automatically. All eleven prepared 200 Hz f3 configurations now carry an explicit disabled reason, which the trainer checks before building loaders or a model. No whole model family is eliminated from a single seed.

Owner, exact configuration basename, seed, SLURM cgroup and process-tree identities were checked before stopping. Only those four process trees were terminated; allocation shells and colocated experiments were preserved. Logs and per-run design_stop.json remain; these old processes had no durable model checkpoint. Full receipts are in docs/receipts/f3-design-stop-receipt.json.

## Status and correction

There is **no frozen new candidate**. f3 is a screening arm, not a demonstrated improvement over rot2. Describing it as the primary candidate before comparing historical validation curves was premature.

The historical rot2 seeds 0/1/2 have best validation kappa **.633889/.649116/.668153**, while their test kappas are .735155/.747546/.715623. All three peak at epoch 0 and then decline substantially. Historical duplex also peaks early (epoch 0 or 2). The live f3 improved to validation **.6562 at epoch 6** after the .4961 epoch-5 dip. A validation value around .64 does not itself show failure relative to the historical .733 average test result. It also cannot predict the new test result.

The most recent inspection covered f0–f3 through epoch 9. Best validations: f0 .5920, f1 .6294, f2 .6006, f3 .6562. f3's latest value was .5716. The historical and current runs use the same recorded TUEV split sizes/class counts and configured data location, and the same training recipe apart from architecture/frontend changes and the operational time cap.

## Evidence

- Class counts (train/val/test background fractions): 39,904/68,445 = 58.30%; 13,822/15,487 = 89.25%; 19,646/29,421 = 66.78%. Class prevalence and subject composition differ across splits. This is a reason to examine class metrics and not equate validation with test kappa, not proof that prevalence alone explains the gap.
- Training loss decreases toward .44–.45 while validation fluctuates or declines. With six classes and label smoothing .1, the optimum target entropy is .420956. This is consistent with fitting training data well while generalization deteriorates. Validation loss and per-class trajectories were not persisted by the old trainer, so a more specific mechanism is not established.
- No missing-gradient or non-finite-forward failure was found in the real-batch frontend diagnostic. f0 intentionally has zero local-projector gradient; f1/f2/f3 have nonzero gradients in both projectors.
- Initial token RMS on the same real TUEV training batch:

| arm | coupling RMS | local RMS |
|---|---:|---:|
| f0 | .22795 | 0 |
| f1 | .22795 | .72937 |
| f2 | .22795 | .19586 |
| f3 | .45455 | 3.89623 |

f3's local RMS is about 8.57 times the coupling RMS. Both lanes enter shared encoder normalization. The source confirms that sinc kernels normalize their central coefficient rather than equalizing passband gain. Changing the lower cutoff also changes effective gains. This motivates a controlled scale/filter-response experiment; it does not establish that normalization is the cause of the learning-curve decline, and it is an initialization measurement rather than a trained-checkpoint probe.

## Stopping reasoning before the filter-response result

Do not cancel the entire six-corpus pilot based on the mistaken comparison of .64 validation against .73 test. No common fatal tokenizer defect has been demonstrated. No new training sweep or pretraining job was launched during this investigation; GPU work was confined to diagnostics inside an existing allocation.

The current TUEV setting uses epochs=20 and patience=20 with one validation per epoch, so patience essentially cannot shorten a run. Retrospectively, patience=5 preserves all best points in the six old rot2/duplex curves and stops at epochs 6–8. But across 133 historical tri-axial curves it reduces 2,677 evaluations to 1,043 while missing later improvements in 16 runs, with a worst kappa loss .07669. Therefore it is a budget tradeoff, not a universally lossless stopping rule. Short screening runs must be labeled as screening; they must not permanently eliminate a model from one seed or be mixed with full-recipe results as equal-budget comparisons.

**Existing processes cannot hot-load the new stop handling.** Their best state is only in RAM until the final evaluation and is not saved as a model file even at normal completion. Sending SIGTERM to them would discard that state. The later filter-response measurement justified terminating the four affected f3 processes despite that limitation. No model checkpoint was claimed to have been preserved. The other arms continue; the operator-stop safeguards apply only to newly started processes.

## Implemented safeguards for new processes

`training/run_control.py` and the trainer integration provide:

- Atomic `best.pt` writes at every validation improvement. This contains the model/config/selection metric and is explicitly not an optimizer-resume checkpoint.
- Atomic `progress.json`: validation loss, training loss, learning rates, all reported validation metrics, confusion matrix, per-class F1 and predicted/true class counts at each validation.
- Cooperative SIGTERM/SIGINT or a per-run `STOP` file. The trainer checks at most 50 batches apart, validates the partial epoch, retains the selected checkpoint, writes `stopped.json`, and **does not score the test set** for an operator stop. No completed `result.json` is fabricated.
- Worker processes retain normal signal termination semantics. An existing run directory with results/progress/checkpoints/STOP is rejected to avoid overwriting or mixing records.
- `run_monitor: false` preserves the legacy opt-out; monitored versus unmonitored smoke runs produce exactly the same validation curves and final metrics.

Do not send signals to allocation shells or controllers to stop one configuration. Use a run's STOP file for new monitored processes, or first verify a process's owner, config, seed and SLURM allocation before any process-level action. A STOP request is not complete until `stopped.json` exists and the process has exited.

## Validation

GPU smoke executes the actual trainer with a tiny synthetic task through `slurm/smoke_gpu.slurm` on verified idle GPU 1 inside allocation 416483. It covers normal completion, both stop mechanisms, checkpoint strict reload, no test scoring after operator stop, unchanged metrics, worker termination, binary output shapes and protection against existing-run overwrite. The separate scale diagnostic reads a fixed real TUEV training batch. Both stay off the login node.

Artifacts: `results/factorized-scale-diagnostic-20260913.json`, `results/run-control-smoke-20260913.txt`, and `docs/receipts/tuev-investigation-20260913.json`. The inspection does not prove a new final model or a generalization improvement from a proposed scale adjustment.

The full factorized launch smoke now checks realized first-band response via `smoke/filter_response.py` and hashes that helper in its receipt. The diagnostic verifies that the current 200 Hz f3 fails this response gate while f0/f1/f2 pass. Full pilot launch remains blocked until a corrected filter design passes the gate and a fresh smoke receipt is obtained.
