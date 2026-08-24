#!/bin/bash
#SBATCH --job-name=x_caueeg
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd /work1/chenyuyou/yifanwang/data/caueeg
python3 - <<'PYEOF'
import zipfile, time
t0 = time.time()
z = zipfile.ZipFile("caueeg.zip")
z.extractall(pwd=b"cnir-caueeg")
print("extracted in %.1f s" % (time.time() - t0))
PYEOF
du -sh caueeg-dataset
ls caueeg-dataset | head
python3 -c "
import json
m = json.load(open('caueeg-dataset/annotation.json'))
print('annotation keys:', list(m.keys())[:8] if isinstance(m, dict) else type(m))
d = json.load(open('caueeg-dataset/dementia.json')) if __import__('os').path.exists('caueeg-dataset/dementia.json') else None
print('dementia.json exists:', d is not None)
if isinstance(d, dict): print('dementia keys:', list(d.keys())[:8])
"
ls caueeg-dataset/signal/edf | wc -l
echo DONE
