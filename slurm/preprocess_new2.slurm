#!/bin/bash
#SBATCH --job-name=prep_mumtaz_eegmat
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"

D=/work1/chenyuyou/yifanwang/data/eegmat
if ! find "$D" -name "Subject*_1.edf" | grep -q .; then
    echo "== unzip eegmat =="
    (cd "$D" && unzip -q -n eeg-during-mental-arithmetic-tasks-1.0.0.zip)
fi
find "$D" -name "*.edf" | wc -l

echo "== mumtaz =="
python -m preprocessing.mumtaz --config configs/datasets/mumtaz.yaml
echo "== eegmat =="
python -m preprocessing.eegmat --config configs/datasets/eegmat.yaml
echo "== output =="
du -sh "$PACLOCK_PROC"/processed/mumtaz "$PACLOCK_PROC"/processed/eegmat
python3 -c "
import json
for ds in ('mumtaz','eegmat'):
    m = json.load(open('/work1/chenyuyou/yifanwang/Zhizhe/processed/%s/manifest.json' % ds))
    print(ds, {k: (v['n_windows'], v['class_counts']) for k, v in m['splits'].items()},
          'excluded:', len(m.get('excluded', [])))
"
echo DONE
