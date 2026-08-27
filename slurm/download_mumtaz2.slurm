#!/bin/bash
#SBATCH --job-name=dl_mumtaz2
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd /work1/chenyuyou/yifanwang/data/mumtaz
rm -f mumtaz_mdd_eeg.zip
# The articles/<id>/versions/<v> bulk URL returns empty here; go through the
# public API instead -- list the article's files, then fetch each by its own
# ndownloader URL.
curl -s 'https://api.figshare.com/v2/articles/4244171' > article.json
python3 - <<'PYX'
import json
a = json.load(open('article.json'))
print('title:', a.get('title'))
for f in a.get('files', []):
    print(f['name'], f['size'], f['download_url'])
open('urls.txt','w').write('\n'.join(f['download_url'] + ' ' + f['name'] for f in a.get('files', [])))
PYX
while read -r url name; do
  echo "== $name"
  curl -L -A 'Mozilla/5.0' -o "$name" "$url"
done < urls.txt
ls -la
