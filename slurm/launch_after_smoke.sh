#!/bin/bash
# Wait for the coupling_self smoke job to finish, check it passed, then launch the self-inclusive arms.
cd /work1/chenyuyou/yifanwang/Zhizhe/PACLock
for i in $(seq 1 180); do
  st=$(sacct -n -X -o state -j 416522 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in COMPLETED|FAILED|TIMEOUT|CANCELLED*) break;; esac
  sleep 60
done
f=$(grep -l "coupling_self=" logs/smoke_gpu-*.out 2>/dev/null | head -1)
echo "smoke state=$st log=$f"
[ -n "$f" ] && grep -E "coupling_self=|SMOKE_OK|Error|Traceback" "$f" | tail -5
if [ -n "$f" ] && grep -q "SMOKE_OK" "$f"; then
  n=0
  while read spec; do
    chunk[$n]="$spec"; n=$((n+1))
    if [ $n -eq 4 ]; then sbatch -J SELF_$RANDOM slurm/configs_packed.slurm "${chunk[@]}" 2>&1 | grep -E "Submitted|error"; n=0; fi
  done < results/selfcoup_specs.txt
  [ $n -gt 0 ] && sbatch -J SELF_$RANDOM slurm/configs_packed.slurm "${chunk[@]:0:$n}" 2>&1 | grep -E "Submitted|error"
  echo "self-inclusive arms launched"
else
  echo "SMOKE DID NOT PASS -- nothing launched"
fi
