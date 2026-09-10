#!/bin/bash
# When a pretraining pair's two checkpoints reach their final step, submit the 8 finetunes (2 packed nodes). Args: <pool-tag> <steps>
cd /work1/chenyuyou/yifanwang/Zhizhe/PACLock; tag=$1; steps=$2
while true; do
  ok=0
  for m in native crofremo; do f=pretrain_runs_cbramod/pretrain-cbramod_${m}_${tag}/checkpoint.pt; [ -f "$f" ] && s=$(python3 -c "import torch;print(torch.load('$f',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) && [ "$s" = "$steps" ] && ok=$((ok+1)); done
  [ $ok -eq 2 ] && break; sleep 300
done
D=configs/cf1
SEED=0 sbatch -J FT_${tag}_a slurm/configs_packed.slurm $D/tuev_cbramod_ptn_${tag}.yaml $D/tuev_cbramod_ptc_${tag}.yaml $D/iiic_cbramod_ptn_${tag}.yaml $D/iiic_cbramod_ptc_${tag}.yaml >> logs/auto_launch.log 2>&1
SEED=0 sbatch -J FT_${tag}_b slurm/configs_packed.slurm $D/chbmit_cbramod_ptn_${tag}.yaml $D/chbmit_cbramod_ptc_${tag}.yaml $D/tusz_cbramod_ptn_${tag}.yaml $D/tusz_cbramod_ptc_${tag}.yaml >> logs/auto_launch.log 2>&1
echo "$(date) launched finetunes for $tag" >> logs/auto_launch.log
