#!/bin/bash
# When both pretraining logs of a pool say "pretraining done", submit the 8 finetunes (2 packed nodes). Args: <pool-tag>
# (login-node python has no torch, so completion is read from the logs, not from the checkpoint)
cd /work1/chenyuyou/yifanwang/Zhizhe/PACLock; tag=$1
while true; do
  ok=0
  for m in native crofremo; do f=$(ls -t logs/pre_cbr_${m}_${tag}-*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -q "pretraining done" "$f" && ok=$((ok+1)); done
  [ $ok -eq 2 ] && break; sleep 300
done
D=configs/cf1
SEED=0 sbatch -J FT_${tag}_a slurm/configs_packed.slurm $D/tuev_cbramod_ptn_${tag}.yaml $D/tuev_cbramod_ptc_${tag}.yaml $D/iiic_cbramod_ptn_${tag}.yaml $D/iiic_cbramod_ptc_${tag}.yaml 2>&1 | grep -E "Submitted|error" >> logs/auto_launch.log
SEED=0 sbatch -J FT_${tag}_b slurm/configs_packed.slurm $D/chbmit_cbramod_ptn_${tag}.yaml $D/chbmit_cbramod_ptc_${tag}.yaml $D/tusz_cbramod_ptn_${tag}.yaml $D/tusz_cbramod_ptc_${tag}.yaml 2>&1 | grep -E "Submitted|error" >> logs/auto_launch.log
echo "$(date) launched finetunes for $tag" >> logs/auto_launch.log
