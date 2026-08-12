#!/bin/bash
#SBATCH --job-name=B-vendor
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Clone the remaining group-B upstream repos. Runs under Slurm because it is
# network + disk work, not something for the login node.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/vendor"

clone() {
  local url=$1 dst=$2
  [ -d "$dst" ] && { echo "$dst already present"; return; }
  git clone --depth 1 "$url" "$dst" 2>&1 | tail -2
}

clone https://github.com/BINE022/EEGPT.git eegpt
clone https://github.com/Jathurshan0330/TFM-Tokenizer.git tfm

echo "--- sizes ---"
du -sh */ 2>/dev/null

for r in eegpt tfm; do
  echo "--- $r layout ---"
  find "$r" -maxdepth 2 -type d -not -path '*/.git*' | head -12
  echo "--- $r checkpoints ---"
  find "$r" -type f \( -iname '*.ckpt' -o -iname '*.pth' -o -iname '*.pt' -o -iname '*.bin' \) 2>/dev/null | head -6
  echo "--- $r readme (weights section) ---"
  grep -inE 'checkpoint|pretrain|weight|download|huggingface|drive.google' "$r"/README.md 2>/dev/null | head -8
done
