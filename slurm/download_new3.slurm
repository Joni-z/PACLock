#!/bin/bash
#SBATCH --job-name=dl_caueeg_siena
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
D=/work1/chenyuyou/yifanwang/data

echo "== CAUEEG (Google Drive, committee-approved 2026-08-24) =="
mkdir -p "$D/caueeg" && cd "$D/caueeg"
python3 -m pip install --user -q gdown 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
if [ ! -s caueeg.zip ]; then
    gdown 1Ox6V7Zz6D6U2imAJ-Ghzr3RyXe70YTHO -O caueeg.zip --no-cookies || \
    gdown "https://drive.google.com/uc?id=1Ox6V7Zz6D6U2imAJ-Ghzr3RyXe70YTHO" -O caueeg.zip
fi
ls -l caueeg.zip 2>/dev/null
if unzip -t -P cnir-caueeg caueeg.zip > /dev/null 2>&1; then
    echo "zip OK, extracting"
    unzip -q -n -P cnir-caueeg caueeg.zip
    find . -maxdepth 2 | head -20
else
    echo "CAUEEG ZIP BAD OR DOWNLOAD FAILED"
fi
du -sh "$D/caueeg"

echo "== Siena Scalp EEG (PhysioNet 1.0.0) =="
mkdir -p "$D/siena" && cd "$D/siena"
wget -q -c "https://physionet.org/static/published-projects/siena-scalp-eeg/siena-scalp-eeg-database-1.0.0.zip"
ls -l *.zip
if unzip -t siena-scalp-eeg-database-1.0.0.zip > /dev/null 2>&1; then
    echo "zip OK, extracting"
    unzip -q -n siena-scalp-eeg-database-1.0.0.zip
    rm -f siena-scalp-eeg-database-1.0.0.zip
    find . -maxdepth 2 -name "*.edf" | head -5
else
    echo "SIENA ZIP BAD OR INCOMPLETE (kept for resume)"
fi
du -sh "$D/siena"
df -h /work1 | tail -1
echo DONE
