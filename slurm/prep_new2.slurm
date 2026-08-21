#!/bin/bash
#SBATCH --job-name=prep_new2
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
module load pytorch/2.7.1
D=/work1/chenyuyou/yifanwang/data/eegmat
cd "$D"
echo "== resume eegmat download =="
wget -c -q "https://physionet.org/static/published-projects/eegmat/eeg-during-mental-arithmetic-tasks-1.0.0.zip"
ls -l *.zip
unzip -t eeg-during-mental-arithmetic-tasks-1.0.0.zip > /dev/null && echo "zip OK" || { echo "zip STILL BAD"; }
unzip -q -n eeg-during-mental-arithmetic-tasks-1.0.0.zip && find . -maxdepth 2 -name "*.edf" | head -5
find . -name "*.csv" -exec echo "-- {}" \; -exec head -5 {} \;
echo "== EDF headers =="
python3 - <<'PYEOF'
import mne, glob
for pat in ["/work1/chenyuyou/yifanwang/data/eegmat/**/Subject00_1.edf",
            "/work1/chenyuyou/yifanwang/data/eegmat/**/Subject00_2.edf",
            "/work1/chenyuyou/yifanwang/data/mumtaz/H S1 EC.edf",
            "/work1/chenyuyou/yifanwang/data/mumtaz/MDD S1 EC.edf",
            "/work1/chenyuyou/yifanwang/data/mumtaz/H S1 TASK.edf"]:
    hits = glob.glob(pat, recursive=True)
    if not hits:
        print("MISSING:", pat); continue
    f = hits[0]
    raw = mne.io.read_raw_edf(f, preload=False, verbose="ERROR")
    print("--", f)
    print("   sfreq", raw.info["sfreq"], "| dur %.1fs" % (raw.n_times/raw.info["sfreq"]), "| nch", len(raw.ch_names))
    print("   chans:", raw.ch_names)
PYEOF
echo "== mumtaz inventory =="
ls /work1/chenyuyou/yifanwang/data/mumtaz/*.edf | sed 's/.*\///' | awk '{print $1}' | sort | uniq -c
ls /work1/chenyuyou/yifanwang/data/mumtaz/*.edf | sed 's/.* //;s/\.edf//' | sort | uniq -c
echo DONE
