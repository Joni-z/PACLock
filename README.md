# PACLock-Bench

AMD/ROCm 集群上的 PACLock baseline 评测矩阵实现。

目标:完成 `PACLock_baseline_matrix.xlsx` 中 9 个数据集 × 4 组模型的评测矩阵。

## 事实来源

| 文件 | 内容 |
|---|---|
| `../PACLock_baseline_matrix.xlsx` | **最高权威**。评测矩阵与冻结协议 |
| `docs/PROTOCOLS.md` | xlsx 的仓库内转录版。代码以此为准 |
| `docs/CHANGELOG.md` | 协议变更记录(任何冻结项的修改必须留痕) |

> 参考仓库 `Joni-z/PACLock`(NVIDIA 集群)的预处理协议与本仓库**不同**——
> 那边沿用 BIOT 协议,这边是 CBraMod 协议。不要跨仓库拷贝预处理脚本。
> 模型架构可以借鉴。

## 集群使用规则(必读)

**不要在登录节点跑任何计算。** 登录节点只用于外部 I/O、基础编译,以及与
资源管理器交互(`squeue` / `sbatch` / `sacct` / `scancel`)。它由约 20 位用户共用,
管理员会直接终止违规进程。

一切其他工作走 Slurm——包括看起来"很快"的东西:

| 动作 | 正确做法 |
|---|---|
| 预处理、训练、评测 | `sbatch slurm/*.slurm` |
| 单元测试 / 前向检查 / 参数量统计 | `sbatch`(在负载高的节点上并不快) |
| 长时间下载、重试守护 | `sbatch slurm/download_*.slurm`,**不要 `nohup`** |
| `squeue` / `sacct` / `scancel` / 读小日志 / `rsync` 代码 | 登录节点可以 |

用户指南:https://amdresearch.github.io/hpcfund/jobs.html

注:`devel` 分区虽有 30 分钟短队列,但其节点缺少 `pytorch/2.7.1` 依赖的
`rocm/6.3.1` 模块,短测试也用 `mi2104x`。

## 环境

集群自带 ROCm PyTorch 模块,不需要自建 conda 环境:

```bash
module load pytorch/2.7.1     # torch 2.7.1+rocm6.3, cpython-39, 依赖 rocm/6.3.1
```

ROCm 兼容性已验证(`tests/test_rocm_compat.py`,MI210 上 8/8 通过):
matmul、rfft/irfft、Hilbert 解析信号、复数 autograd、BF16 autocast、
autocast 内 FFT、SDPA、`conv1d(k=201)`。BIG_CLUSTER_HANDOFF 里标记为
「未评估」的三个 AMD 风险点均无需改写。

额外依赖(已装到 `~/.local`):`mne moabb tensorpac braindecode einops timm`。

## 数据

原始数据在 `/work1/chenyuyou/yifanwang/data/`(与 youran 共用,只读):

| 数据集 | 路径 | 状态 |
|---|---|---|
| TUAB v3.0.1 | `tuh/tuab` | ✅ |
| TUEV v2.0.1 | `tuh/tuev` | ✅ |
| TUSZ v2.0.6 | `tuh/tusz` | ✅ |
| CHB-MIT v1.0.0 | `chbmit` | ✅ |
| Sleep-EDF (SC) | `sleep-edf` | ✅ |
| PhysioNet-MI | `physionet-mi` | ✅ |
| BCI-IV-2a | `bci-iv-2a` | ✅ |
| ISRUC Subgroup I | `isruc` | ⏳ 部分(Mega 带宽限额) |
| SEED-V | — | ⛔ 待 SJTU BCMI 授权 |

预处理产物写到 `/work1/chenyuyou/yifanwang/Zhizhe/processed/<dataset>/`,
**不写回共用的 data 目录**。

## 用法

```bash
# 预处理(每个数据集一次,产出 npy + manifest.json)
python -m preprocessing.tuab --config configs/datasets/tuab.yaml

# 训练
python -m paclock_bench.training.train --config configs/experiments/<exp>.yaml

# Slurm
sbatch slurm/train.slurm configs/experiments/<exp>.yaml
```

## 目录

```
configs/datasets/      每个数据集的冻结协议参数(与 docs/PROTOCOLS.md 一一对应)
configs/models/        模型超参
configs/experiments/   实验矩阵:数据集 × 模型 × seed
preprocessing/         预处理脚本,每个数据集一个,共享 common.py
paclock_bench/data/    torch Dataset,读预处理产物
paclock_bench/models/  baselines/ foundation/ paclock/
paclock_bench/training/ 训练循环与指标
scripts/               工具(结果收集、manifest 校验)
slurm/                 作业脚本
tests/                 烟测(提交 GPU 作业前的正确性检查)
```

## 硬规则(摘自 xlsx README sheet)

1. B 组模型进表前必须先用**自己的 recipe** 复现自己论文的数字(≥3 seeds,
   发表值落在 mean±2std 内),否则标 `not reproduced`。
2. 每个模型用自己的 recipe(B、D 组);架构对照用对称超参(C 组)。
3. val 曲线峰值在 epoch 0 ⇒ 标 `mis-configured`,拒绝写入。
4. 全部 3 seeds,报 mean ± std。
5. 所有数据集从零预处理,不复用来源不明的缓存。

完整协议见 `docs/PROTOCOLS.md`。
