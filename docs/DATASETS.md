# Where the nine corpora come from

Everything here is a download plus a version. The exact preprocessing that turns
each into `processed*/` is `docs/PROTOCOLS.md`, and is executed by
`slurm/preprocess.slurm <protocol> <dataset>`.

## Access

| corpus | version | access | note |
|---|---|---|---|
| TUAB | v3.0.1 | TUH EEG Corpus — **registration required**, credentials by email | abnormal/normal, recording-level |
| TUEV | v2.0.1 | same TUH account | 6-class events, 390 patients |
| TUSZ | v2.0.6 | same TUH account | seizure onset/offset, subject-disjoint splits |
| CHB-MIT | 1.0.0 | PhysioNet, open | paediatric seizures |
| Sleep-EDF | sleep-cassette | PhysioNet, open | |
| PhysioNet-MI | EEGMMIDB 1.0.0 | PhysioNet, open | motor imagery |
| ISRUC | Subgroup 1 | ISRUC-SLEEP site, open | `slurm/download_isruc.slurm` automates it |
| FACED | — | request from the authors | emotion, 32 channels |
| BCI-IV-2a | — | BNCI Horizon 2020, open | motor imagery, 22 channels |

The three TUH corpora share one account and are the only ones needing a human
in the loop. Apply first; everything else can be fetched while waiting.

## Versions are not optional

Each `processed*/<corpus>/manifest.json` records the **SHA256 of every source
file**, the subject IDs in each split, and the per-class window counts. If a
re-download differs — a corpus revision, a partial mirror, a different
subgroup — the arrays it produces are not comparable with any result already in
`runs/`, and the whole matrix has to be re-run rather than extended.

Check before trusting anything:

```bash
sbatch slurm/run.slurm scripts.verify_processed --dataset tuev
```

This is not a formality. Two preprocessed corpora were lost to disk pressure on
the source cluster (`processed_biot/tuab`, `processed_labram/tuab`) and have to
be rebuilt rather than copied, which makes them the first place a version drift
would show up.

## Splits are frozen, not recomputed

For TUH: the official train/eval split is used, and the official *train* is cut
80/20 by **sorted subject ID** into train/val — never randomly, so the split is
identical on any machine without carrying a file around. `eval` is the test set
and is never touched during selection. Same rule for the others, per corpus, in
`docs/PROTOCOLS.md`.

## Order of operations on a new cluster

```bash
# 1. apply for TUH access, download everything into $PACLOCK_DATA
sbatch slurm/download_isruc.slurm

# 2. frozen protocol first -- groups A, C, D and CBraMod all read it
for ds in tuab tuev tusz chbmit sleepedf isruc physionet_mi faced bci_iv_2a; do
    sbatch slurm/preprocess.slurm frozen $ds
done

# 3. the per-model protocols, only if group B is being run
sbatch slurm/preprocess.slurm biot   tuab      # BIOT and TFM-Tokenizer
sbatch slurm/preprocess.slurm labram tuab      # LaBraM: 23 unipolar channels

# 4. the PAC-protocol arm, only for the group-D sensitivity analysis
sbatch slurm/preprocess.slurm pac tuev

# 5. verify before running anything that will be reported
sbatch slurm/run.slurm scripts.verify_processed --dataset tuev
```

Step 2 is enough for all PACLock work. Steps 3 and 4 exist because hard rule 2
requires each model to run its own repo's preprocessing.
