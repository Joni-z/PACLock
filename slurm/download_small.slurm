#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Small open-data downloads, kept off the login node like everything else.
#   sbatch slurm/download_small.slurm <dest_dir> <url> [url...]
set -euo pipefail
DEST=$1; shift
mkdir -p "$DEST"; cd "$DEST"
for u in "$@"; do
  echo "== $u"
  wget -c -q --show-progress --no-check-certificate "$u" 2>&1 | tail -2 || wget -c "$u"
done
ls -la
