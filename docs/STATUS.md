# 进度(2026-08-18 晚)

上一版写于同日上午,此后做了一次方向重估并开了两条新线,故重写。

数字全部来自 `runs/`,由 `scripts/status_snapshot.py` / `scripts/xlsx_gap.py` 生成;
括号内是 seed 数。当前阶段**单 seed 可用**(诊断为主),但 seed 数一律标出——
把 1 seed 的数字当 3 seed 讲是这份文档唯一可能误导人的地方。

---

## 文档导航

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| `STATUS.md` | 本文件 —— 现状、在跑的实验、遗留问题、两个集群怎么用 | 先读这个 |
| `PROTOCOLS.md` | 冻结的预处理与评测协议、语料来源、baseline 配方审计、**存储清理记录** | 改预处理或复现协议之前 |
| `FINDINGS.md` | 架构搜索每一波的结论、性能修复、交付配置及依据 | 想改模型之前 —— 大部分想法已经试过了 |
| `PRETRAIN.md` | 预训练方案与实际执行 | 要再跑预训练时 |
| `PAPER.md` | 论文需要的实验矩阵,以及**被证伪的预注册预测** | 写论文 / 排投稿实验时 |
| `CHANGELOG.md` | 按时间的变更日志,含被否决的方案和原因 | 想知道"这个为什么是现在这样" |

---

## 1. 必须先说的:novelty 和战绩对不上

我们在 TUEV / TUSZ / CHB-MIT 上领先全部外部 baseline。但拆开看领先来自哪里:

| 语料 | 我们的最好成绩 | 来自什么 |
|---|---|---|
| TUEV | 0.7328 | **PAC tokenizer**(对 raw +0.172) |
| TUSZ | 0.6884 | rotation;但 raw tokenizer 本身就有 0.6710,PAC 从零只有 0.5882 |
| CHB-MIT | 0.6830 | **预训练**;PAC 从零 0.5464,raw 0.6672 |

**PAC tokenizer 只在 TUEV 一个语料上是获胜的原因。**另外两个的领先来自 raw
tokenizer 或预训练 —— 把我们的 novelty 整个拿掉,这两格不降反升。

这不只是"效果不够",是**论文完整性问题**:主张 A,战绩来自 B。审稿人读消融表
会发现,我们自己已经发现了。

两次扩展尝试都失败:

* **TUAR**(唯一一次检验 TUEV 优势能否推广到同类事件形态任务):pac 0.5780
  vs raw 0.6289,**raw 反赢 0.05**。
* **预注册的分层预测**(增益应随各语料 PAC 生理学证据强度排序):**被自己的数据
  证伪**,增益不分层,是单点的(见 `PAPER.md`)。

## 2. 当前正在推的两条线

两条线针对的是同一个事实:PAC tokenizer 从未在它应该起作用的条件下被测试过。

### 线 A —— 预训练过的 PAC tokenizer,从来没测过

预训练在 `patch_len=200` 下做,而 TUSZ/CHB-MIT/BCI 的微调配置是 `50`,tokenizer
的 Conv1d 核形状对不上,**加载时被形状检查直接丢弃**。也就是说这些语料上的
"预训练"一直是「预训练 encoder + 从零重学的 tokenizer」。
**CHB-MIT 的最好成绩 0.6830 就是用一个随机初始化的 PAC tokenizer 跑出来的。**

对齐 `patch_len` 能修好这件事,但同时会改变 token 网格和 PAC 估计窗口 ——
`FINDINGS.md` 记载那是全项目最大的单一架构因素。所以设计成三臂,**全部固定在
`patch_len=200`**:

| 臂 | 配置 | 含义 |
|---|---|---|
| A | 从零 | 参照点 |
| B | 预训练,tokenizer 迁移 | 完整预训练 |
| C | 预训练,`checkpoint_exclude` 掉 tokenizer | 只迁移 encoder |

**B − C 就是预训练 tokenizer 的净贡献**,分辨率不动。

`scripts/verify_ckpt_exclude.py`(job 377036,全通过)证实:B 在 200 下**真的**
加载了 tokenizer(153 个张量 vs C 的 150),C 的 tokenizer 与随机初始化**逐位
相同**,B 和 C 加载**同样的 144 个 encoder 张量**、差异只在那 3 个 tokenizer
张量上。

任务 `PK_p200_{chbmit,tusz,bci_iv_2a}`,CHB-MIT 优先 —— 它是 PAC 输给 raw 最惨的
一格(−0.121),最能说明"预训练能否救活 PAC tokenizer"。

### 线 B —— tokenizer 移植进 CBraMod

一个**不需要我们的架构在任何地方获胜**的、更窄的主张:把 PACLock 的前端装进
CBraMod 原样 vendored 的 encoder,对上 CBraMod 自己的 tokenizer。
Pilot:TUEV 0.6280(1 seed)vs CBraMod scratch 0.5638。

原设计缺一个决定性对照,现已补上:**第三臂 = CBraMod + PACLock 的 raw 前端**。
没有它,赢了也只能说"我们的前端比他们的好" —— 前端还捎带了学习式 sinc 滤波器组
和分频带 token 轴。同宿主内 PAC vs raw 才能把交互项和滤波器组分开。

`scripts/verify_transplant.py`(job 377014,全通过):两臂参数量精确相等
(**30,660,422**,原生 30,646,006),差异只在 tokenizer 张量,192 个 encoder
张量完全相同。

语料刻意选成:TUEV(我们模型里 PAC 赢)+ BCI、CHB-MIT(PAC 输得最惨)。
**若移植后在后两个也赢,说明 tokenizer 是好的、问题出在我们的架构** —— 那是个
完全不同也更好讲的故事。

**已知弱点**:移植目前**全是 from scratch**,适配器不加载任何 CBraMod 预训练
权重。所以只能对上 CBraMod 的 scratch(TUEV 0.5638),对不上它真正的
**0.6449**。审稿人会说"CBraMod 的价值就在预训练,你把它弄失效了再跟它没预训练
的版本比"。**待办:第四臂 = CBraMod 预训练 encoder + 我们的 tokenizer**,
需要先确认 `load_pretrained` 能跳过 `patch_embedding` 做部分加载。

## 3. 两处已落地的架构改动(均零新增参数)

### `interaction_mode: rotation`

`token = a_j · aligned_phase_j / |aligned_phase_j|` —— 耦合**旋转**幅度 token
而不是同时缩放它。强制性与 `product` 相同,但 `|h_j| = |a_j|` 精确成立。

九个语料 **六正三负**,但只有两格是 3 seed:

| 语料 | product | rotation | delta | seed |
|---|---|---|---|---|
| TUSZ | 0.5882 | 0.6884 | +0.100 | 1 |
| **TUEV** | 0.7076 | **0.7328** | **+0.025** | **3** |
| PhysioNet-MI | 0.2722 | 0.2961 | +0.024 | 1 |
| ISRUC | 0.6952 | 0.7104 | +0.015 | 1 |
| **BCI-IV-2a** | 0.3588 | **0.3708** | **+0.012** | **3** |
| FACED | 0.1477 | 0.1514 | +0.004 | 1 |
| Sleep-EDF | 0.6459 | 0.6449 | −0.001 | 1 |
| TUAB | 0.8829 | 0.8806 | −0.002 | 1 |
| CHB-MIT | 0.5464 | 0.5100 | **−0.036** | 1 |

TUEV 逐 seed 看:rotation 最差的(0.7156)高于 product 的均值(0.7076),
product 有一个 seed 掉到 0.6718。均值与稳定性同时改善。

CHB-MIT 的 −0.036 要放进上下文:product 自己三 seed 是
**[0.5809, 0.5382, 0.5200]**,跨度 0.061,单个 −0.036 落在约一个标准差内,
**什么也没证明**,但也不能说它是好的。

方法学副产品:`scripts/verify_rotation.py` **第一次真正验证了规范不变性**
(`p_i → e^{iδ_i}p_i, Z_ij → e^{iδ_i}Z_ij` 下 1.. 频带不动),`product` 与
`rotation` 都通过 —— 这是 `_pac_interaction` docstring 一直声称、却从未被测过的
核心主张。

### `head: spatial` —— 帮助真实,但**不属于 PAC**

不在电极轴上池化。`mean`/`band`/`attn` 都会把电极轴平均掉,而运动想象的判别量
就是空间对比(对侧感觉运动区 mu/beta 去同步)。

**关键对照(2026-08-18 晚落地,推翻了此前的说法):**

| BCI-IV-2a | mean 头 | spatial 头 |
|---|---|---|
| PAC tokenizer | 0.3588 (3) | 0.4317 (3) |
| raw tokenizer | 0.4192 (3) | **0.5274 (1)** |

把头固定住,**raw 比 PAC 高 0.096** —— 比用 mean 头时的 0.060 差距**更大**。
先前"BCI 上 PAC+spatial 赢过 raw+mean"的说法**作废**:那是拿好头的 PAC 比
差头的 raw。空间头对两者都有效,对 raw 更有效。

0.5274 是 PACLock 家族在 BCI 上的历史最好成绩(此前 raw_headattn 0.4545),
但它**加强的是"我们的读出头选错了"这个结论,不是 PAC 的地位**。

TUEV 上空间头是 **−0.056**(0.6520 vs 0.7076):等级一安全检查未通过,但**符合
机制预测** —— 等级一标签空间弥散,保留电极身份只是白花参数。符号翻转本身是机制
成立的证据。结论:**必须按语料条件化,不能设全局默认**,和 `patch_len` 一样。

## 4. 等级三:排除掉的与仍存活的

| 假设 | 判定 | 证据 |
|---|---|---|
| PAC 在 band-power 任务上退化到随机 | **混淆** | 0.259 来自 `processed_pac` 预处理;同预处理下 0.3588 |
| 训练样本太少 | **否** | TUEV 砍到 2160 窗口仍得 0.6523,仍领先 SPaRCNet +0.161 |
| 分类头参数爆炸 | **否** | 九语料 `n_params` 恒为 1.60–1.63 M |
| recipe 没调好 | **否** | `patch200`/`lr3e5` 2×2 最多 +0.011 |
| 幅度信息不可及 | **否** | 探针:频带功率从 rotation token 二次可读 R²=0.907、concat 线性可读 0.853,均高于获胜的 raw(0.469/0.031) |
| 耦合显著性可当门控 | **否** | surrogate 校准后 TUEV 与 BCI 的显著边比例都是 36% 且类间平坦。**显著性 ≠ 判别性** |
| **电极轴被平均掉** | **成立,但不救 PAC** | 空间头两个 MI 语料 +0.07~0.10,**对 raw 帮助更大**;FACED 无效 |

**FACED 仍完全未解**:15 个 PACLock 变体全在 0.110–0.169(随机 0.111),
CBraMod 从零 0.2469、预训练 0.5509。空间头没救回来,原因未知。

## 5. 预训练现状

* 预训练**全部在 b2 做**,checkpoint 传回 AMD 微调。正式 60k checkpoint 在
  `pretrain_runs_60k/`;`pretrain_runs/` 下同名文件是 6000 步早期试跑,只被
  `*_ft_*` 配方实验引用,**矩阵行未受影响**(曾误报为混淆,系
  `scripts/ckpt_steps.py` 早期版本用 `basename(dirname())` 剥掉父目录所致,已修)。
* **排除消融**:预训练池剔除 TUSZ/CHB-MIT 后,收益仍保留 **66%**(TUSZ)和
  **69%**(CHB-MIT)。只有约三分之一来自域内数据。(剔除臂 n=1。)
* **raw vs pac 预训练**(同 60k、同 d256、每语料 `patch_len` 一致):raw 在三个
  等级三语料上分别 **+0.049 / +0.036 / +0.005**。TUSZ 与 CHB-MIT 的对应格子
  此前从未跑过,现正在跑。

## 6. 负面结果(必须进论文)

* **TUAR**:pac 0.5780 vs raw 0.6289。"PAC 在事件形态类任务上普遍占优"**不成立**。
* **TUSL**:全语料仅 300 个事件(每类 100),花训练算力前放弃。
* **预注册分层预测被证伪**(`PAPER.md`)。
* `interaction_mode: concat`、`pac_token_mode: uniform`、`coupling_gate:
  significance`、`spatial_pe: xyz`、per-window / per-subject 归一化、FACED 伪迹
  清洗、提高学习率、降低容量 —— 全部实测无效,记录在 `FINDINGS.md`。

## 7. 遗留问题

1. **FACED 完全学不动**,未知原因。
2. **线 B 缺第四臂**(CBraMod 预训练 encoder + 我们的 tokenizer),否则对不上
   CBraMod 真正的 0.6449。
3. rotation 多数格子仍 n=1(补 seed 已主动停掉 —— 那是在一条平均输给 raw 的
   路径上做 +0.025 的改良,补齐误差棒不改变判决)。
4. 空间头需在更多语料上确认,且需要 `raw + spatial` 的完整对照(BCI 已有,
   PhysioNet-MI 在跑)。
5. `scripts/audit_runs.py` 只列 A 组 45 格,C/D 组与预训练行只进总数不进明细。
6. 6 个 baseline 单元格 seed 离散 20–43%(BIOT 配方固有),论文附录应给各 seed 值。

## 8. 集群规范

所有计算走 Slurm;登录节点只用 `squeue`/`sbatch`/`sacct` 和代码同步。

* **AMD**(`ssh amd`,ROCm/MI210,免费):微调与全部消融。
* **b2**(`ssh b2`,PSC Bridges-2,CUDA):**只做预训练**。配额仅剩 147/700 SU,
  该账号无 CPU 分区,新实验一律放 AMD。密码认证(PSC 未注册公钥),
  `ControlMaster` 每会话输一次密码。
* **默认单 seed。** 除非 Zhizhe 明确要求,所有实验只跑一个 seed。硬规则 4
  (进论文正表需要 3 seed)不变,但那是这一格要进表了才付的代价 —— 探索阶段
  跑一个、拿到答案,不要保险起见顺手排 seed 1 和 2。2026-08-18 排队的 27 个
  任务里有 20 个是冗余 seed,占掉共享分区 21 个节点里的 14 个,把真正决定方向的
  那个实验堵在自己后面 —— 而它们确认的只是一条已知平均输给自身对照的路径上的
  +0.025。报数时 seed 数必须写出来。
* **`mi2104x` 一个节点有 4 块 GPU,而 `run.slurm`/`train.slurm` 用 `--exclusive`
  会整节点独占** —— 单任务提交等于浪费 3/4 算力。诊断实验用
  `slurm/configs_packed.slurm`(一节点 4 个不同配置,单 seed);矩阵格子用
  `slurm/seeds_packed.slurm`(一节点一个配置的 3 个 seed)。
  打包后节点占用从 27 降到 10,吞吐反升。
* 打包时**按运行时长分组**:同语料的多个臂时长一致、同时结束;把 7 小时的
  CHB-MIT 和 16 分钟的 BCI packed 在一起会让三块 GPU 空转 6 小时。
* 两个集群的 `runs/` **不要互相 tar 覆盖**,同名 seed 目录会被静默替换。
* **存储**:`/work1` 的 1.9 T 是**项目配额**,不是集群容量(底层 382 T)。
  看告警要用 `df -h /work1/chenyuyou/yifanwang`。清理记录与重建方法见
  `PROTOCOLS.md` 附录 C。

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
