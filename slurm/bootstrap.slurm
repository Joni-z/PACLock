#!/bin/bash
#SBATCH --job-name=bootstrap
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# First thing to run on a new cluster:  sbatch slurm/bootstrap.slurm
#
# Clones the five upstream repos and reports which pretrained checkpoints are
# still missing. Network and disk work, so it runs under Slurm rather than on
# the login node.
#
# It cannot fetch every checkpoint. Three of them are behind Google Drive or an
# author request and have to be copied from the old cluster -- the report at the
# end says which, and where each comes from.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p vendor logs
cd vendor

clone() {
    local url=$1 dst=$2
    if [ -d "$dst/.git" ] || [ -d "$dst" ]; then
        echo "  present   $dst"
    else
        echo "  cloning   $dst"
        git clone --depth 1 "$url" "$dst" 2>&1 | tail -1
    fi
}

echo "=== upstream repos ==="
clone https://github.com/ycq091044/BIOT.git              biot
clone https://github.com/935963004/LaBraM.git            labram
clone https://github.com/wjq-learning/CBraMod.git        cbramod
clone https://github.com/BINE022/EEGPT.git               eegpt
clone https://github.com/Jathurshan0330/TFM-Tokenizer.git tfm

echo
echo "=== CBraMod weights (the one that is scriptable) ==="
if [ -f cbramod/pretrained_weights/pretrained_weights.pth ]; then
    echo "  present   cbramod/pretrained_weights/pretrained_weights.pth"
else
    module load pytorch/2.7.1
    python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="weighting666/CBraMod",
                      local_dir="cbramod/pretrained_weights")
print("  downloaded to", p)
PY
fi

echo
echo "=== checkpoint inventory ==="
miss=0
check() {
    if [ -f "$1" ]; then
        printf "  OK       %-58s %s\n" "$1" "$(du -h "$1" | cut -f1)"
    else
        printf "  MISSING  %-58s <- %s\n" "$1" "$2"
        miss=$((miss + 1))
    fi
}
check biot/pretrained-models/EEG-PREST-16-channels.ckpt        "BIOT repo release assets"
check biot/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt   "BIOT repo release assets"
check biot/pretrained-models/EEG-six-datasets-18-channels.ckpt "BIOT repo release assets"
check labram/checkpoints/labram-base.pth                       "LaBraM README, author Google Drive"
check labram/checkpoints/vqnsp.pth                             "LaBraM README, author Google Drive"
check cbramod/pretrained_weights/pretrained_weights.pth        "HuggingFace weighting666/CBraMod"
check eegpt/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt        "EEGPT README, 973 MB, Google Drive"

echo
if [ "$miss" -eq 0 ]; then
    echo "all seven checkpoints present -- group B can run"
else
    echo "$miss checkpoint(s) missing. Group B (pretrained baselines) cannot run"
    echo "until they are in place; groups A/C/D and all PACLock work are fine"
    echo "without them. Copying them from the previous cluster is faster than"
    echo "chasing the download links."
fi

echo
echo "next:  sbatch slurm/run.slurm scripts.verify_frontend"
echo "       sbatch slurm/run.slurm scripts.smoke_paths"
