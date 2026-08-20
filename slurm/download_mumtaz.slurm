#!/bin/bash
#SBATCH --job-name=dl_mumtaz
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd /work1/chenyuyou/yifanwang/data/mumtaz
# figshare ndownloader needs redirect-following and a browser UA; wget got a
# 0-byte file named "2" out of it
curl -L -A 'Mozilla/5.0' -o mumtaz_mdd_eeg.zip   'https://figshare.com/ndownloader/articles/4244171/versions/2'
ls -la
file mumtaz_mdd_eeg.zip 2>/dev/null || true
unzip -l mumtaz_mdd_eeg.zip 2>/dev/null | head -8 || echo 'NOT A ZIP -- inspect'
