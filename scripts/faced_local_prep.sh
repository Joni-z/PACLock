#!/bin/bash
# Prepare FACED on a machine that has the download, then ship only the result.
#
# Why locally: the 52 GB archive would take ~37 h to push to the cluster at the
# measured 0.4 MB/s uplink, while the preprocessed output is ~2.5 GB (~1.8 h).
# Preprocessing is deterministic and the manifest records the source SHA256s, so
# doing it here rather than on the cluster changes nothing about the result.
#
#   bash scripts/faced_local_prep.sh inspect <FACED.zip>
#   bash scripts/faced_local_prep.sh run     <extracted_dir> <out_dir>
#   bash scripts/faced_local_prep.sh upload  <out_dir>
#
# Needs: python3 with numpy, scipy, pyyaml.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_DEST="/work1/chenyuyou/yifanwang/Zhizhe/processed/faced"

usage() { sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 1; }
[ $# -ge 1 ] || usage

case "$1" in

inspect)
  # Answers the one question everything else depends on: is this the official
  # pre-processed release (what the protocol pins) or the raw one?
  ZIP="${2:?usage: inspect <FACED.zip>}"
  echo "== archive =="
  ls -lh "$ZIP"
  echo
  echo "== top-level entries =="
  unzip -l "$ZIP" | awk 'NR>3 {print $4}' | grep -v '^$' \
    | awk -F/ '{print $1"/"$2}' | sort -u | head -20
  echo
  echo "== file types =="
  unzip -l "$ZIP" | awk 'NR>3 {print $4}' | grep -oE '\.[A-Za-z0-9]+$' \
    | sort | uniq -c | sort -rn | head -10
  echo
  echo "== verdict =="
  if unzip -l "$ZIP" | grep -qiE '\.pkl'; then
    n=$(unzip -l "$ZIP" | grep -ciE '\.pkl')
    echo "  found $n .pkl files"
    echo "  -> looks like the PRE-PROCESSED release. This is what the protocol"
    echo "     wants. Extract the .pkl and use: $0 run <dir> <out>"
  else
    echo "  no .pkl found -> this is probably the RAW release."
    echo "  -> STOP and check with Zhizhe. The frozen protocol requires the"
    echo "     official pre-processed data (it already has ICA eye-movement"
    echo "     removal, which we cannot reproduce bit-for-bit). Using raw is a"
    echo "     protocol deviation and needs an explicit decision + CHANGELOG entry."
  fi
  ;;

run)
  SRC="${2:?usage: run <extracted_dir> <out_dir>}"
  OUT="${3:?usage: run <extracted_dir> <out_dir>}"
  n_pkl=$(find "$SRC" -name '*.pkl' | wc -l | tr -d ' ')
  echo "found $n_pkl .pkl under $SRC"
  if [ "$n_pkl" -eq 0 ]; then
    echo "ERROR: no .pkl -- see 'inspect' output; raw data needs a decision first." >&2
    exit 1
  fi
  mkdir -p "$OUT"
  cd "$REPO"
  python3 -m preprocessing.faced \
      --config configs/datasets/faced.yaml \
      --raw-root "$SRC" \
      --out-dir "$OUT"
  echo
  echo "== output =="
  du -sh "$OUT"
  ls -lh "$OUT"
  ;;

upload)
  OUT="${2:?usage: upload <out_dir>}"
  echo "uploading $(du -sh "$OUT" | cut -f1) to the cluster (resumable -- rerun if it drops)"
  # -P keeps partial files and shows progress, so an interrupted run resumes.
  rsync -avP --partial "$OUT"/ "amd:$CLUSTER_DEST/"
  echo
  echo "verifying the far side:"
  ssh amd "ls -lh $CLUSTER_DEST && python3 -c \"
import json;m=json.load(open('$CLUSTER_DEST/manifest.json'))
print('splits:', {k: v['n_windows'] for k, v in m['splits'].items()})
print('qc:', m['qc'].get('n_subjects_found'), 'subjects')\""
  ;;

*) usage ;;
esac
