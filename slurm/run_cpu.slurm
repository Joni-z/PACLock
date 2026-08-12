#!/bin/bash
#SBATCH --job-name=run_cpu
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
#     sbatch slurm/run_cpu.slurm scripts.fill_xlsx --xlsx results/_in.xlsx ...
#
# For work that never imports torch -- collecting results, filling/normalizing
# the xlsx, generating configs, checking py3.9 compatibility. These were going
# through slurm/run.slurm, which asks for --exclusive on mi2104x: a whole
# 4-GPU node, for a script that never touches a GPU, on the one partition that
# is permanently full across every user sharing this cluster. `devel` is the
# same job for these scripts and is idle far more often, so this queues in
# seconds instead of minutes to hours.
#
# devel's own wall-clock cap is 30 minutes; the 25 here just leaves margin.
#
# Deliberately does not `module load pytorch` -- the bare system python3 here
# already has numpy/openpyxl/pyyaml/scikit-learn (verified 2026-08-13), and
# paclock_bench.paths / paclock_bench.training.metrics import cleanly without
# torch. If a module genuinely needs torch (anything building or running a
# model), use slurm/run.slurm instead -- this one will fail it, not silently
# run it wrong, because torch simply is not importable here.
set -euo pipefail
MODULE=${1:?usage: sbatch slurm/run_cpu.slurm <module.path> [args...]}
shift
cd "$SLURM_SUBMIT_DIR"
python3 -u -m "$MODULE" "$@"
