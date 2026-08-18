# 进度(2026-08-18)

上一版停在 2026-08-06「A 组完成,下一步 B/C/D」。此后 B/C/D 组、60k 预训练、
预训练行入表、等级三诊断、两处架构改动全部完成,本文件据此重写。

数字全部来自 `runs/`,由 `scripts/status_snapshot.py` 生成;括号内是 seed 数,
**n=1 的一律视为线索而非结论**(项目硬规则 4:少于 3 seed 不进表)。

---

## 文档导航

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| `STATUS.md` | 本文件 —— 现状、在跑的实验、遗留问题、两个集群怎么用 | 先读这个 |
| `PROTOCOLS.md` | 冻结的预处理与评测协议、九个语料的来源、baseline 配方审计 | 改任何预处理或复现协议之前 |
| `FINDINGS.md` | 架构搜索的每一波结论、性能修复、交付配置及其依据 | 想改模型之前 —— 大部分想法已经试过了 |
| `PRETRAIN.md` | 预训练方案与实际执行情况 | 要再跑一次预训练时 |
| `PAPER.md` | 论文需要的实验矩阵,以及还缺什么 | 写论文 / 排投稿前的实验时 |
| `CHANGELOG.md` | 按时间的变更日志,含被否决的方案和原因 | 想知道"这个为什么是现在这样" |

---

## 1. 一句话现状

九个语料的完整对比矩阵已建成并填满(770 个 run,101 个单元格,0 个缺 seed、
0 个误配置)。我们在 **TUSZ / CHB-MIT / TUEV** 上领先全部 baseline,在
**TUAB / Sleep-EDF / ISRUC** 上互有胜负,在 **PhysioNet-MI / FACED / BCI-IV-2a**
上明显落后。落后的原因已定位,且**不是 tokenizer**(见 §4)。

## 2. 主表现状

| 语料 | 指标 | 最强外部 baseline | 我们(冻结 v2) | 我们(最好变体) |
|---|---|---|---|---|
| TUEV | kappa | tfm_pretrained 0.6519 | 0.7076 (3) | **rotation 0.7328 (3)** |
| TUSZ | PR-AUC | ffcl 0.5449 | 0.5882 (3) | **rotation 0.6884 (1)** |
| CHB-MIT | PR-AUC | tfm_pretrained 0.6269 | 0.5464 (3) | **pt_base 0.6830 (3)** |
| TUAB | AUROC | eegpt_pretrained 0.9028 | 0.8829 (3) | pt_large 0.8869 (3) |
| Sleep-EDF | kappa | contrawr 0.6916 | 0.6459 (3) | pt_base 0.6651 (3) |
| ISRUC | kappa | cbramod_pretrained 0.7540 | 0.6952 (3) | rawtok 0.7013 (3) |
| PhysioNet-MI | BAcc | cbramod_pretrained 0.6129 | 0.2722 (5) | raw_large 0.4159 (3) |
| BCI-IV-2a | BAcc | sparcnet 0.6440 | 0.3588 (3) | rawpt_large 0.4483 (3) |
| FACED | BAcc | cbramod_pretrained 0.5509 | 0.1477 (3) | rawpt_large 0.1690 (2) |

产物:`results/PACLock_baseline_matrix_filled.xlsx`,含三列 delta
(vs from-scratch / vs pt-base / vs pt-large)、灰显未被 baseline 原论文覆盖的格子。

## 3. 两处已验证的架构改动(均零新增参数)

### 3.1 `interaction_mode: rotation`

`token = a_j · aligned_phase_j / |aligned_phase_j|` —— 耦合**旋转**幅度 token
而不是同时缩放它。强制性与 `product` 完全相同(token 相位仍完全由耦合决定,
旁边没有裸的高频 token,没有可学习旁路),但 `|h_j| = |a_j|` 精确成立。

| 语料 | product | rotation | delta |
|---|---|---|---|
| TUSZ | 0.5882 (3) | **0.6884 (1)** | **+0.100** |
| TUEV | 0.7076 (3) | **0.7328 (3)** | **+0.025** |
| PhysioNet-MI | 0.2722 (5) | 0.2961 (1) | +0.024 |
| BCI-IV-2a | 0.3588 (3) | 0.3708 (3) | +0.012 |
| FACED | 0.1477 (3) | 0.1514 (1) | +0.004 |
| Sleep-EDF | 0.6459 (3) | 0.6449 (1) | −0.001 |
| TUAR | 0.5780 (1) | 0.5568 (1) | −0.021 |

TUEV 三个 seed 逐个看:rotation 最差的 seed(0.7156)高于 product 的均值
(0.7076),product 有一个 seed 掉到 0.6718 而 rotation 最低 0.7156 —— 均值和
稳定性同时改善。TUSZ/CHB-MIT/TUAB/ISRUC 的确认 run 在跑。

验证(`scripts/verify_rotation.py`,全部通过):`product`/`concat` 与改动前
**逐位一致**(所有冻结格子不动);`|h|=|a|` 误差 2.4e-7;相位内容仍在(不是退化
成纯幅度 tokenizer);**规范不变性首次被真正验证**(`product` 和 `rotation` 都
通过),这是 `_pac_interaction` docstring 一直声称却从未测过的核心主张。

### 3.2 `head: spatial`

不在电极轴上池化,保留电极身份。`mean`/`band`/`attn` 三种头都会把电极轴平均掉,
而等级三的标签本身就是空间模式。

| 语料 | mean 头 | spatial 头 | delta |
|---|---|---|---|
| PhysioNet-MI | 0.2722 (5) | **0.3560 (1)** | **+0.084** |
| BCI-IV-2a | 0.3588 (3) | **0.4144 (1)** | **+0.056** |
| FACED | 0.1477 (3) | 0.1514 (1) | +0.004 |

两个运动想象语料大幅改善(对侧感觉运动区 mu/beta 去同步就是空间对比),
情绪语料无反应。**叠加有效**:BCI rotation+spatial = 0.4344,高于任一单项。

修掉一个会让该实验作废的 bug:`spatial` 头原本在 `forward()` 里懒构建投影层,
而 optimizer 早已捕获参数列表 —— 它会带着随机权重训练全程,然后被记录成
「试过没用」。现改为按 `cfg['n_channels']` 提前构建
(`scripts/verify_head.py` 验证:`mean`/`band`/`attn` 逐位一致,投影层在首次
forward 前已在 `parameters()` 中,能收到梯度并被优化器更新)。

## 4. 等级三落后的原因:已排除的与仍存活的

按排除顺序记录,每条都是实测否掉的,免得重走:

| 假设 | 判定 | 证据 |
|---|---|---|
| PAC tokenizer 在 band-power 任务上退化到随机 | **混淆** | 0.259 来自 `processed_pac` 预处理;同预处理下是 0.3588,PAC 对 raw 只差 ~0.06 |
| 训练样本太少 | **否** | TUEV 砍到 2160 窗口(=BCI 规模)仍得 0.6523,仍领先 SPaRCNet +0.161;32 倍数据缩减只掉 0.055 |
| 分类头在高电极数下参数爆炸 | **否** | 九个语料的 `n_params` 恒为 1.60–1.63 M |
| recipe(patch_len / lr)没调好 | **否** | 已跑过的 `patch200`/`lr3e5` 2×2 最多 +0.011,`lr3e5` 为负 |
| 幅度信息在 PAC token 里不可及 | **否** | 线性可读性探针:频带功率从 rotation token 二次可读 R²=0.907、从 concat 线性可读 0.853,均**高于**赢的 raw(0.469/0.031) |
| 耦合显著性可以当门控 | **否** | surrogate 校准后,TUEV 与 BCI 的显著边比例都是 36% 且类间平坦。**显著性 ≠ 判别性** |
| **电极轴被平均掉** | **成立(部分)** | spatial 头在两个 MI 语料上 +0.056 / +0.084;FACED 无效 |

FACED 仍是完全没解决的:15 个 PACLock 变体全部落在 0.110–0.169 之间(随机 0.111),
而 CBraMod 从零训练就有 0.2469、预训练 0.5509。空间头没救回来,原因未知。

## 5. 预训练现状

* 预训练**全部在 b2 做**,checkpoint 传回 AMD 做微调。
* 60k 步的正式 checkpoint 在 `pretrain_runs_60k/`(base / large,均 pac tokenizer)。
  `pretrain_runs/` 下的同名文件是 6000 步的早期试跑,**只被 `*_ft_*` 那批配方
  实验引用**,矩阵行未受影响。
* `pretrain_runs/pretrain-raw_large`(60k,raw tokenizer)与
  `pretrain-excl_szdet`(60k,pac,预训练池剔除 TUSZ/CHB-MIT)也在 AMD。

**排除消融**(答「你拿下游数据做预训练」的质疑):把 TUSZ/CHB-MIT 从预训练池
剔除后,预训练收益仍保留 **66%**(TUSZ:+0.0339 of +0.0516)和 **69%**
(CHB-MIT:+0.0946 of +0.1366)。只有约三分之一来自域内数据。(剔除组 n=1。)

**raw vs pac 预训练**(两边同为 60k、d_model=256、每语料 `patch_len` 一致):

| 语料 | raw 预训练 | PAC 预训练 | raw − pac |
|---|---|---|---|
| PhysioNet-MI | 0.3633 | 0.3140 | +0.049 |
| BCI-IV-2a | 0.4483 | 0.4122 | +0.036 |
| FACED | 0.1690 | 0.1640 | +0.005 |

TUSZ / CHB-MIT 的对应格子从未跑过,现已提交。

**一个待修的配方问题**:预训练用 `patch_len=200`,而 TUSZ/CHB-MIT/BCI 的微调
配置是 `patch_len=50`,卷积核形状对不上,加载时 tokenizer 权重被直接跳过 ——
这几个语料上的「预训练」**只迁移了 encoder,tokenizer 从头重学**。FACED 和
PhysioNet-MI(`patch_len=200`)才真的迁移了 tokenizer。每语料内 raw/pac 两边
一致,所以对比公平,但意味着**我们从未在 TUSZ/CHB-MIT 上测过「预训练过的 PAC
tokenizer」**,而那正是最该体现其价值的地方。

## 6. 负面结果(必须进论文)

* **TUAR**:pac 0.5780 vs raw 0.6289,raw 赢 0.05。所以「PAC 在事件形态类任务上
  普遍占优」**不成立**,TUEV 目前是唯一的强证据。
* **TUSL**:全语料仅 300 个事件(每类 100),在花训练算力之前就放弃了。
* `interaction_mode: concat`、`pac_token_mode: uniform`、`spatial_pe: xyz`、
  per-window / per-subject 归一化、FACED 伪迹清洗、提高学习率、降低容量 ——
  全部实测无效,记录在 `docs/FINDINGS.md`。

## 7. 遗留问题

1. **FACED 完全学不动**(§4),未知原因。
2. **`patch_len` 不匹配导致 tokenizer 不迁移**(§5),需要一组
   `patch_len=200` 的 TUSZ/CHB-MIT 微调才能测出预训练过的 PAC tokenizer。
3. **rotation 需补 seed**:TUSZ/PhysioNet-MI/FACED/Sleep-EDF 目前 n=1。
4. **spatial 头需补 seed**,且需在等级一上做安全性检查(TUEV 的在跑)。
5. **`scripts/audit_runs.py` 只列 A 组 45 个格子**,C/D 组和预训练行只进总数
   不进明细,建议扩表。
6. 6 个 baseline 单元格 seed 离散 20–43%(BIOT 配方 `lr=1e-3` 无 scheduler 固有
   不稳),论文附录应给各 seed 值而非只报均值。

## 8. 集群规范

所有计算走 Slurm;登录节点只用 `squeue`/`sbatch`/`sacct` 和代码同步。

* **AMD**(`ssh amd`,ROCm/MI210,免费):微调与全部消融在此。
* **b2**(`ssh b2`,PSC Bridges-2,CUDA):只做预训练。**配额只剩 147 / 700 SU**,
  该账号无 CPU 分区(连 `run_cpu_b2.slurm` 都要申请一块最便宜的 GPU),所以
  新实验一律放 AMD。
* 两个集群的 `runs/` **不要互相 tar 覆盖**,同名 seed 目录会被静默替换、把不同
  硬件混进同一个格子的统计里(见 `docs/CHANGELOG.md`)。

---

# 附录:集群使用与迁移

*(原 `docs/STATUS.md`)*

Written for the full pretraining run. Everything the repo needs is either in
git, regenerable, or listed here as something to copy.

## 1. What to copy, and what not to

| | size | how |
|---|---|---|
| the git repo | ~30 MB | `git clone` / `git bundle`. Includes `runs/` and `archive/` — every stored result, config and val curve |
| `vendor/` | **1.6 GB** | **copy, do not clone.** Five upstream repos *plus their pretrained checkpoints*. `slurm/vendor_clone.slurm` re-clones the code but not the weights |
| `$PACLOCK_PROC/processed*` | **~490 GB** | copy if the bandwidth is there, otherwise regenerate — see §4 |
| `$PACLOCK_DATA` (raw corpora) | ~510 GB | only needed to regenerate the above |

`logs/` is ignored and not worth moving.

## 2. The two environment variables

Nothing else is configured. `REPO` is derived from `paclock_bench/paths.py`'s
own location, so the checkout finds `vendor/` wherever it lands.

```bash
export PACLOCK_DATA=/new/path/raw          # raw corpora, read-only
export PACLOCK_PROC=/new/path              # parent of processed*/
```

Both default to the paths they had on the AMD HPC Fund, so the repo keeps
working there with nothing set. Configs say `$PACLOCK_PROC/processed/tuev` and
are expanded by `paths.expand()` as they are read; an absolute path passes
through untouched, so old configs still work.

Check the move with:

```bash
python3 -m scripts.smoke_paths      # builds real loaders + models from 3 configs
```

## 3. Cluster assumptions baked into `slurm/`

These are AMD HPC Fund specifics and will need editing:

* `--partition=mi2104x`, and **no GPU GRES** — `scontrol show node` reports
  `Gres=(null)`, so `--gpus=1` is rejected and the only way to get a GPU is
  `--exclusive`, which hands over all four MI210s. Both packing scripts exist
  because of that: `seeds_packed.slurm` runs three seeds of one config on one
  node, `configs_packed.slurm` runs up to four different configs. On a cluster
  with proper GRES, ask for one GPU per job and drop both.
* `module load pytorch/2.7.1` (ROCm 6.3.1).
* **24-hour wall limit**, enforced by a submit filter. `max_hours` in a config
  makes a run stop cleanly and still write its `result.json` rather than being
  SIGKILLed with nothing.
* `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` are set per process to
  node-local `/tmp`. Concurrent processes sharing one MIOpen SQLite cache on a
  shared filesystem corrupt it ("database disk image is malformed"). Keep this
  on any ROCm cluster; harmless on CUDA.

## 4. Regenerating the preprocessed corpora

Four protocols, because hard rule 2 says each model runs its own repo's
preprocessing (`docs/PROTOCOLS.md`):

```bash
sbatch slurm/preprocess.slurm         <dataset>   # frozen protocol -> processed/
sbatch slurm/preprocess_pac.slurm     <dataset>   # PAC protocol    -> processed_pac/
sbatch slurm/preprocess_biot.slurm    <dataset>   # BIOT + TFM      -> processed_biot/
sbatch slurm/preprocess_labram.slurm  <tuab|tuev> # LaBraM          -> processed_labram/
```

Two are currently missing on the source cluster and will have to be rebuilt
rather than copied, both lost to disk pressure: `processed_biot/tuab` and
`processed_labram/tuab`. `processed_tfm/` never existed; the TFM cells read
`processed_biot` through the same loader.

## 5. Read these before changing anything

* `docs/FINDINGS.md` — the 9.4x speed fix. `set_seed()` forces
  `cudnn.deterministic=True`, which on ROCm made the `in_channels=1` patch
  convolutions pick an atomics-free backward-weights kernel. The fix replaced
  them with a GEMM. **If the new cluster is CUDA this may not reproduce** — the
  2x2 in that document is the way to check, not assumption.
* `docs/FINDINGS.md` — the PAC estimation window result, and the two
  predictions it falsified.
* `docs/FINDINGS.md` — the configuration to pretrain and the evidence for
  each choice.
* `docs/PROTOCOLS.md` — every baseline recipe deviation and its source file.

## 6. The one number that governs how results are compared

Seed spread is **measured per corpus, never assumed**: ISRUC sd 0.0021, TUEV sd
0.0235 — eleven times apart, because 57% of TUEV cells peak at their first
evaluation and 0% of ISRUC cells do. A delta has to clear roughly twice its own
corpus's sd before it means anything.

And the reason `archive/runs_conv_tokenizer/` exists: a change that was
mathematically identical and verified to rel 8.9e-7 moved an ISRUC result by
0.0276, **13.4 seed standard deviations**. After any numerical change to the
frontend, re-run rather than reuse.

---

## 7. The move that actually happened: PSC Bridges-2 (`ssh b2`), 2026-08-12 onwards

This document was written before the move. What follows is what the move cost
and what had to be different, so the next one is cheaper.

**Division of labour**: pretraining runs on b2 (CUDA), and the finished
checkpoint is copied back to AMD, where all finetuning and every ablation runs.
AMD is free; b2 is metered.

### The allocation is GPU-only, and metered

* **There is no CPU partition.** Every RM / RM-shared / RM-small / RM-512
  submission fails `Invalid qos specification`, and GPU-shared refuses
  `--gpus=0` outright. `slurm/run_cpu_b2.slurm` therefore requests the cheapest
  GPU (`v100-16:1`) purely to get a schedulable job. Anything CPU-only that will
  run repeatedly belongs on AMD's `devel` partition instead.
* **`--gpus=type:n`, not `--gres=gpu:n`.** Bridges-2's own docs call this out;
  `--gres` was observed to still allocate but is not the supported form here.
* Billing per GPU-hour: `l40s-48` 24, `h100-80` 12, `v100-32` 5.
* **Balance is 147 / 700 SU as of 2026-08-18** — about six l40s-hours. Plan
  accordingly: do not queue anything on b2 that AMD can run.

### `l40s-48` is the default GPU, not `v100-16`

GPU-shared memory scales with GPU type (`sinfo -o "%G %m"`). A single `v100-16`
allocation gives ~24 GB, which OOM-killed the first full-pool (10-corpus)
pretraining smoke test once several corpora's `WindowDataset` preload landed in
RAM at once. `size_large` on the big corpora also OOM'd a `v100-32` (5120 tokens,
32 GB) and had to move to AMD's 64 GB MI210s.

### The environment cannot nest a venv inside the module

The `pytorch` module's torch is not importable from inside a fresh venv, so
extra packages are installed with `pip install --target <dir>` and that directory
is put on `PYTHONPATH`. `slurm/run_b2.slurm` sets `PACLOCK_DATA`, `PACLOCK_PROC`
and that `PYTHONPATH` together.

### Transfer notes

* `scp` to b2 is unreliable in practice; `cat`-piping over ssh
  (`cat local | ssh b2 "cat > remote"`) has been dependable.
* **Never write scratch scripts to a shared `/tmp`.** A `/tmp/inspect.py` owned
  by another user on b2 shadowed the stdlib `inspect` module and broke imports.
  Put them in the repo.
* Password auth, not keys: PSC never registered a public key for this account,
  so `~/.ssh/config` sets `PubkeyAuthentication no` and
  `PreferredAuthentications keyboard-interactive,password`. `ControlMaster` /
  `ControlPersist 8h` means the password is entered once per session.
* **The account is shared.** `Youran/` and `Wenhao/` under
  `/ocean/projects/cis260249p/qren2/` belong to other people, and their jobs show
  up in `squeue -u $USER`. Check `%Z` (work dir) before cancelling anything.

### The rule that came out of this move

**Never tar `runs/` between the two clusters.** Same-named seed directories are
silently replaced, which mixes two hardware platforms into one cell's statistics.
It happened once: b2's TUSZ seed0 (0.6448) overwrote AMD's (0.6989), and it was
caught only because the `wall_time` matched b2's elapsed rather than AMD's.
26 directories had to be quarantined by their `checkpoint` path and re-run.
One cluster per matrix; copy checkpoints, never results.

### Checkpoint provenance

Copy checkpoints to a **distinct directory name**, never over an existing one.
`pretrain_runs_60k/` holds the finished 60k-step pretrainings; `pretrain_runs/`
still holds 6000-step pilots of the same names, and 8 early `*_ft_*` runs
reference those. Overwriting would have destroyed the provenance of both sets.
`scripts/ckpt_steps.py` maps every run to the checkpoint and step count it
actually used — run it before trusting any pretrained comparison.
