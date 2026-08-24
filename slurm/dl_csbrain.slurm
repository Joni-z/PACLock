#!/bin/bash
#SBATCH --job-name=dl_csbrain
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /work1/chenyuyou/yifanwang/Zhizhe/PACLock/vendor
if [ ! -d csbrain ]; then git clone -q https://github.com/yuchen2199/CSBrain csbrain; fi
cd csbrain && git log --oneline -1 && ls
mkdir -p weights && cd weights
gdown --folder "https://drive.google.com/drive/folders/1-GsVVewRM0B93H08yts5m53yU2whxYvj" 2>&1 | tail -5 || echo "FOLDER DOWNLOAD FAILED"
find . -maxdepth 3 -type f | head -20
du -sh .
echo DONE
