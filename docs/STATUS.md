# 进度(2026-08-21)

上一版写于 08-18 晚。此后完成了结构收敛的四波单 seed 消融
(hybrid/fused → duplex → H1–H4 单因子 → 组合 + 旗舰),**旗舰式全组件堆叠被
自己的隔离实验证伪**,骨干判决随之收敛;12 个下游数据集的数据侧全部齐备。
本版重写现状部分;08-18 版的两条线(线 A patch200 三臂、线 B CBraMod 移植)
已收尾,过程与结论并入 `FINDINGS.md`。

数字全部来自 `runs/`;括号内是 seed 数,当前阶段默认单 seed(探索),
进论文正表才补 3 seed(硬规则 4 不变)。

---

## 文档导航

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| `STATUS.md` | 本文件 —— 现状、在跑的实验、点火条件、两个集群怎么用 | 先读这个 |
| `PROTOCOLS.md` | 冻结的预处理与评测协议、语料来源、baseline 配方审计、存储清理记录 | 改预处理或复现协议之前 |
| `FINDINGS.md` | 架构搜索每一波的结论、性能修复、交付配置及依据 | 想改模型之前 —— 大部分想法已经试过了 |
| `PRETRAIN.md` | 预训练方案与实际执行、扩展阶梯 | 要再跑预训练时 |
| `PAPER.md` | 论文实验矩阵、12 数据集名单、被证伪的预注册预测 | 写论文 / 排投稿实验时 |
| `CHANGELOG.md` | 按时间的变更日志,含被否决的方案和原因 | 想知道"这个为什么是现在这样" |

---

## 1. 活动目标

> 用单 seed 消融确定**最有前景做预训练的模型结构**(结构,不是训练参数);
> 最终选 12 个下游数据集,每一个都要有好表现,不允许出现输十几个点的格子。

## 2. 骨干判决(当前证据下)

**预训练骨干 = `tokenizer_mode: duplex` + `interaction_mode: rotation` +
`n_bands: 8` + 线性 tokenizer + 三轴 encoder。**

它是唯一在全部三个等级一语料上同时超过外部 baseline 的网格(单 seed):

| 语料 | duplex | 最强外部 baseline | delta |
|---|---|---|---|
| TUEV | 0.7094 | 0.6519 (TFM) | +0.058 |
| TUSZ | 0.6328 | 0.5449 (FFCL) | +0.088 |
| CHB-MIT | 0.7130 | 0.6269 (TFM) | +0.086 |

对照:fusegate 在 TUSZ/CHB 更高(0.6950/0.7441)但 TUEV 崩(0.5879 —— TUEV
需要行分离的交互 token);raw 在 TUEV 输 0.14。duplex 是没有短板的那个。

**读出头按任务族在微调时选,不属于骨干。**这不是妥协:CBraMod 自己就带 13 个
`model_for_*.py`,每个下游数据集一个头 —— 预训练交付的是骨干
(tokenizer + encoder),头本来就是不迁移的微调期部件。我们的头族:
`mean`(临床/事件)、`spatial`(MI,电极身份)、`flatten`(线索锁定的
功率轨迹任务)。

## 3. 组件判决表(旗舰证伪波,2026-08-20/21)

「零初始化保底 ⇒ 全局可安全叠加」被证伪:保底只保第 0 步,训练动力学照样跑偏。
每个语料只有一个约束在起作用,组件不叠加。

| 组件 | 判决 | 证据(单 seed) |
|---|---|---|
| duplex 网格 | **进骨干** | 上表;唯一无短板 |
| `n_bands: 16` | 不进骨干,语料条件项 | BCI +0.031(0.5583);TUEV(pac)安全 0.7223;**全家桶里与其他组件互毁** |
| 深 conv stem(H1) | 语料条件项 | TUEV +0.020、FACED +0.034、PMI +0.015;**CHB −0.023、TUSZ −0.044** —— 伤癫痫语料 |
| 学习式蒙太奇(H2) | 语料条件项 | 仅 PMI +0.049,其余无效 |
| flatten 头(H4) | 语料条件项 | FACED +0.072(0.2344);伤 BCI/PMI |
| gated_meanspatial 头 | **弃用** | γ 零初始化保底在真实训练中不成立:TUSZ 0.6595 vs mean 头 0.6950;BCI 0.4174 vs spatial 头 0.5583 |
| 旗舰全家桶 | **弃用** | BCI 旗舰 0.3661,比单用 nb16+spatial 低 0.19 |

## 4. 12 个下游数据集名单现状(已全部实测,判决见 §4.9)

数据、baseline、我们的臂三者齐备。**每个语料都有 15 个 baseline 单元格**
(A 组 5 轻量监督 + B 组 5 官方预训练 FM + C 组 5 同款 from-scratch),
判决按"该语料 15 个 baseline 里的最好者"计。

| 状态 | 数据集 |
|---|---|
| 赢(5) | TUSZ +0.088、CHB-MIT +0.086、TUEV +0.057、ADFD +0.034、TUEP +0.017 |
| 平(1) | TUAB −0.004 |
| 噪声边缘(2) | Sleep-EDF −0.017、Mumtaz −0.022(基准已饱和) |
| 真输(3) | ISRUC −0.042、TUAR −0.086、**EEGMat −0.129** |
| 建议除名(2) | **APAVA**(语料容量不足,见 §4.9)、**Mumtaz**(基准饱和,留着不输但证明不了东西) |

候补池:TUSL、SHU-MI、SEED-VIG、BCI-IV-2a / PhysioNet-MI(需要预训练翻盘,
而预训练目前是负贡献,见 §4.9)。FACED 已除名。

新语料的 baseline 配方政策:整套从**任务形状相同的现有语料**原样复制
(二分类照 CHB-MIT,三分类照 TUEV),只换语料事实(dataset / data_root /
num_classes / loss),**一个超参都不 per-corpus 调**。baseline 的数字之所以
有意义就在于"它跑的是自己发表的配方";没有参考时保守做法是沿用参考语料的
epochs/patience,宁可让 baseline 训过头也不让它训不足。data_root 一律
`processed/<ds>`:BIOT 和 LaBraM 只在 TUAB/TUEV(它们论文覆盖的语料)读
各自的 `processed_biot` / `processed_labram`。

协议注意:Mumtaz/EEGMat 对齐 CBraMod 的预处理(5 s 窗、19 通道 10-20),但
划分改为我们的被试不相交标准 —— CBraMod 的 Mumtaz 划分按文件排序切,同一
被试的 EC/EO 会跨 train/test(泄漏),我们不抄。

## 4.9 十二语料判决表(2026-08-22,首次完整;全部单 seed)

75 个新 baseline 单元格(5 语料 x 15 模型)+ duplex scratch + duplex 预训练微调
全部落地。每格取该语料 15 个 baseline 里的最好者作对手。

| 语料 | 主指标 | duplex scratch | duplex 预训练 | 最强 baseline | 判决 |
|---|---|---|---|---|---|
| TUSZ | AUC-PR | **0.6328** | 0.6040 | 0.5449 ffcl | **+0.088** |
| CHB-MIT | AUC-PR | **0.7130** | 0.6635 | 0.6269 tfm_pt | **+0.086** |
| TUEV | κ | **0.7094** | 0.6891 | 0.6519 tfm_pt | **+0.057** |
| ADFD | bacc | **0.5617** | — | 0.5279 biot_scr | **+0.034** |
| TUEP | AUROC | **0.8052** | — | 0.7884 labram_pt | **+0.017** |
| TUAB | bacc | 0.8157 | (跑) | 0.8198 st_trans | −0.004 平 |
| Sleep-EDF | κ | 0.6533 | 0.6746 | 0.6916 contrawr | −0.017 |
| Mumtaz | AUROC | 0.9775 | — | 0.9999 biot_pt | −0.022（饱和）|
| ISRUC | κ | 0.7117 | 0.6948 | 0.7540 cbramod_pt | −0.042 |
| TUAR | κ | 0.6289 | — | 0.7147 cbramod_pt | −0.086 |
| EEGMat | AUROC | 0.7263 | — | 0.8557 cbramod_pt | **−0.129** |

**5 胜 1 平 5 负。**

### 三个结论

**1. 预训练在伤我们,不是在帮。**同配置下只加 checkpoint(每个 run 日志都记录
"loaded 152 pretrained backbone tensors",排除 5 个 patch_len 相关张量):
TUEV −0.020、TUSZ −0.029、CHB −0.050、ISRUC −0.017,只有 Sleep-EDF +0.021。
**4/5 为负。**这与旧 pac 预训练在 CHB 上 +0.137 的结果相反,差别在于 duplex 的
encoder 是在 2nb 网格上按幅度目标训练的,而微调时三个 tokenizer 全部重新初始化
——encoder 收到的 token 统计量与预训练时不同,比从零开始还糟。这直接支持
§5.5 之前的判断:**瓶颈是目标函数**(只考幅度的题,配一个专门编码耦合的前端)。

**2. 胜负按任务类型分得很干净。**赢的五个全是**阵发性/瞬态临床事件**
(癫痫发作、癫痫样事件、癫痫状态)加一个痴呆诊断;输的是**持续状态分类**
(睡眠分期)、伪迹形态、以及极小的认知语料。这不是随机分布,是机制性的:
耦合 token 刻画的是短窗内的跨频结构,那正是瞬态痫样活动的特征;睡眠分期靠的是
持续谱态,30 秒窗口下简单模型反而占优。

**3. 两个语料应当除名。**
* **APAVA**:22 个被试、测试集 3 人 46 窗口、类别 2:44,AUROC 建立在 2 个负样本上,
  任何数字都是噪声。不是模型问题,是语料容量问题。
* **Mumtaz**:BIOT 0.9999、ST-Transformer 0.9933、LaBraM 0.9858 —— 一个 1M 参数的
  小模型拿到 0.993 的基准不区分模型,极可能是站点/记录条件可分而非病理可分。
  留着不会输(我们 0.9775 在噪声内),但也证明不了任何东西。

### 缺口

按"每格至少在噪声内"的标准,现在够格的是 8 个(5 胜 + TUAB 平 + Sleep-EDF/Mumtaz
在噪声边缘),**ISRUC / TUAR / EEGMat 是真输**,其中 EEGMat −0.129 属于明确禁止的
两位数亏损,失效模式与 BCI/FACED 相同(训练集仅 1,199 窗口)。

## 5. 在跑 / 排队(amd)

| 任务 | 内容 | 状态 |
|---|---|---|
| DPT_long(381359) | TUAB 的 duplex 预训练微调(TUSZ/CHB 已落) | 在跑 |

其余全部落地。本轮(75 个 baseline 单元格 + 8 个我们的臂)实际消耗
**18 node-hours** —— 远低于按 24 小时墙估算的 600,因为 baseline 本身很便宜
(一个语料跑满 15 个中位 11.1 GPU-小时 ≈ 4.4 node-hours)。

**已停**:`FL_long`(379887)跑满 22h50m 只为产出 CHB 旗舰格,而旗舰已在四个
语料全部输给家族最好、结论闭合 —— 这一格不改变任何判断却占着整节点。
**一个这样的任务 ≈ 五个语料全部 baseline 的总成本**,这是本轮最贵的教训。

## 5.5 预训练池的被试级泄漏(2026-08-21 发现并修复)

TUEG 切片此前只用 TUEG 自带的 `sessions_tueg_common_with_tusz.list` 排除,
那是**会话**清单;而 TUAB/TUEV/TUSZ/TUEP/TUAR 全都是 TUEG 的子集,同一病人的
其他会话仍会进池。切片占预训练采样 **36.8%**,实测重叠:

| 语料 | test 泄漏 | val 泄漏 |
|---|---|---|
| **TUAB** | **88/253 = 34.8%** | 158/424 = 37.3% |
| TUEV | 0/80 = 0%(官方 eval 集) | 28/58 = 48.3% |
| TUSZ | 17/43 = 39.5% | 0% |
| TUEP | 11/28 = 39.3% | 9/26 = 34.6% |
| TUAR | 9/32 = 28.1% | 11/32 = 34.4% |

**修复**(commit 35be6dc):`preprocessing/tueg.py` 增 `exclude_subject_manifests`,
按**被试**剔除五个 TUH 下游语料所有 split 的全部被试;缺 manifest 直接报错
(少排就是这个 bug 本身)。**不损失数据**:排除 3,182 个被试后仍有 11,885 个
合格被试,照样选满 5,245 文件 ≈ 2000 h,泄漏检查 0。重建进
`processed/tueg_slice_clean`(旧切片保留不覆盖,job 44047303)。

**后果 —— 已由决定关闭(2026-08-21,Zhizhe:"泄露一点都不用管")**:
1. ~~预训练行须用干净 checkpoint 重跑~~ —— 采用与 CBraMod/LaBraM 同口径的
   原切片,不重跑。测量数据(上表)与附录 D 的领域调研保留,只作记录。
2. ~~排除消融需重做~~ —— 同上,不重做。
   干净切片 `processed/tueg_slice_clean`(700,110 窗口 / 5,226 被试)已建好并
   保留,若日后要量化泄漏值多少分,一次 60k(约 25 SU)即可,不必重新预处理。

### 决定(2026-08-21,Zhizhe):rung-1 用**未排除**的原切片

分析摆在上面,选择是发**脏版**,即与 CBraMod/LaBraM 同一口径的原切片
(`processed/tueg_slice`)。理由是可比性:对手的公开数字全部建立在未排除的
全量 TUEG 上,单方面清洁会让我们的 TUAB/TUSZ 与注水后的数字对比。

作业:`pt_duplex_base` 44058728(b2 h100,60k,duplex+rotation,d128/depth6,
输出 `pretrain_runs_60k/pretrain-duplex_base/`)。

**清洁切片仍然构建并保留**(`processed/tueg_slice_clean`,job 44058306),
所以"泄漏能把分数抬高多少"随时可以补一个 60k 测出来(约 40 SU / 1h40m),
附录 D 的调研与 §5.5 的测量数据也都保留 —— 论文里要不要用、怎么用,
是后面的事,不是现在被关掉的门。

## 5.6 池子构成:六进六出,是特性不是缺陷

全池采样占比(n 加权):TUEG 36.8%、TUSZ 17.0%、CHB-MIT 16.4%、TUAB 15.5%、
Sleep-EDF 6.4%、ISRUC 3.6%、TUEV 3.6%,**PhysioNet-MI 0.33%、FACED 0.35%、
BCI 0.11%**。后三个合计 0.79%,日志里经常整整 250 步拿到 **0 个 batch**
(`physionet_mi=nan(0)`)——它们在预训练里等于不存在,留着只为通道数多样性
(64/32/22 通道),删不删都不影响结果。

12 语料名单里 **6 个在池中**(TUAB/TUEV/TUSZ/CHB-MIT/ISRUC/Sleep-EDF),
**6 个完全不在**(TUEP/TUAR/ADFD/APAVA/Mumtaz/EEGMat)。不补进池,理由:
ADFD/APAVA/Mumtaz/EEGMat 各自 <1%,加进去和 FACED 一样是统计噪声;TUEP
(约 7%)是唯一有分量的,但把它留在池外反而更值钱——**论文里这就是一个
自带的迁移实验:六个语料有域内训练数据,六个一点没有,两边都要赢。**

## 6. rung-1 预训练:已执行,结果为负

`pt_duplex_base` 44058728(b2 h100,60k 步,duplex+rotation+nb8,d128/depth6,
patch_len 200,10 语料全池 + 原 TUEG 切片,配对行掩码目标)。
**2 小时 05 分,约 25 SU**;h100 计费 12/小时,是 l40s 的一半且更快。
最终重建 loss:TUAB 0.0320 / TUEV 0.0357 / TUSZ 0.0351 / CHB 0.0382 ——
在 base 参数量(1.671M)上补掉了 base→large 差距的约 70–75%。

**但下游是负的**(§4.9):TUEV −0.020、TUSZ −0.029、CHB −0.050、ISRUC −0.017,
只有 Sleep-EDF +0.021。checkpoint 迁移本身无误(`scripts/verify_duplex_transfer.py`
16/16;每个 run 的日志记录 "loaded 152 pretrained backbone tensors")。

配套代码已落地并验证:配对行掩码(commit 7f12853,`scripts/verify_duplex_pretrain.py`
16/16)—— 掩码打在 8 个物理频带上,同时遮住该频带的融合行与交互行,否则可见的
交互行会把重建目标直接递给模型。

## 7. 遗留问题

1. **预训练是负贡献**(§4.9 结论 1)。这是当前最大的问题:一个"基础模型"论文
   的前提是预训练要有用。机制假说:目标只问幅度,而前端的全部卖点是相位耦合;
   且 patch_len 200→50 使三个 tokenizer 全部重初始化,encoder 收到没见过的
   token 统计量。**未验证**。
2. **三个真输的格子**:ISRUC −0.042、TUAR −0.086、EEGMat −0.129。EEGMat 属于
   明令禁止的两位数亏损,失效模式同 BCI/FACED(训练集 1,199 窗口)。
3. 睡眠分期(ISRUC / Sleep-EDF)系统性偏弱 —— 与"瞬态事件强、持续状态弱"的
   机制解释一致,但没有针对性尝试过。
4. FACED 完全学不动(已除名,原因未知)。
5. 所有判决均为**单 seed**;进论文正表需 3 seed(硬规则 4),按语料分批补。
6. 新语料的 A 组 baseline 已跑,但 `scripts/audit_runs.py` 仍只列旧的 45 格。

## 8. 集群规范

所有计算走 Slurm;登录节点只用 `squeue`/`sbatch`/`sacct` 和代码同步。

* **AMD**(`ssh amd`,ROCm/MI210,免费):微调与全部消融。
* **b2**(`ssh b2`,PSC Bridges-2,CUDA):**只做预训练**。配额仅剩 147/700 SU,
  该账号无 CPU 分区,新实验一律放 AMD。密码认证(PSC 未注册公钥),
  `ControlMaster` 每会话输一次密码。
* **devel 分区**：墙钟上限 0.5 h，且没有 pytorch/2.7.1 模块（rocm/6.3.1 缺失）——只能跑纯 CPU/网络任务（下载、解压），适合在 mi2104x 排队时插小活。
* 集群内节点间 ssh 需 home 的 `authorized_keys` 含自己公钥（08-21 已配），且只允许进有本人作业的节点；等价方式 `srun --jobid=<id> --overlap --pty bash`。
* **默认单 seed。** 除非 Zhizhe 明确要求,所有实验只跑一个 seed。硬规则 4
  (进论文正表需要 3 seed)不变,但那是这一格要进表了才付的代价 —— 探索阶段
  跑一个、拿到答案,不要保险起见顺手排 seed 1 和 2。2026-08-18 排队的 27 个
  任务里有 20 个是冗余 seed,占掉共享分区 21 个节点里的 14 个,把真正决定方向的
  那个实验堵在自己后面 —— 而它们确认的只是一条已知平均输给自身对照的路径上的
  +0.025。报数时 seed 数必须写出来。
* **预算(共享账号,必须计划着用)**:AMD 计费单位是 **node-hours**,
  1,107 / 2,250 已用。b2 是 **SU,按单卡**,h100 = 12 SU/小时,余额 696 SU
  = 58 卡·小时。**换算:AMD 打包后最多 4,640 卡·小时,b2 只有 58** ——
  b2 的单位价值约是 AMD 的 35 倍,只该用在 AMD 做不了的事(预训练)上。
  把 baseline 搬去 b2 是反方向:五个语料的 baseline 在 AMD 约 25–30
  node-hours(剩余额度 2.5%),在 b2 要 666 SU(几乎全部预算)。
* **纪律**:任何单任务超过 8 小时,先问"这一格改变结论吗"。
  实测参考:一个语料跑满 15 个 baseline 中位 11.1 GPU-小时 ≈ 4.4 node-hours;
  TFM 是唯一大头(中位 3.0h,最长 16.8h)。
* **存储比算力更紧**:1.6T / 1.9T,只剩 260G。加数据集会先撞存储墙。
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
