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

## 4. 12 个下游数据集名单现状

名单已提名、**数据全部在手**,锁定要等实测数字齐:

| 状态 | 数据集 | 说明 |
|---|---|---|
| 有实测、稳(3) | TUEV、TUSZ、CHB-MIT | duplex 赢 baseline +0.06~0.09 |
| 有旧实测、duplex 确认臂在跑(3) | TUAB、ISRUC、Sleep-EDF | v2 三 seed 数字在矩阵;duplex 臂在 DUP_slate_long |
| 数据就绪、首测排队(4) | TUEP、TUAR、ADFD、APAVA | duplex scratch 单 seed,380980/380981 |
| loader 刚落地、预处理排队(2) | Mumtaz2016、EEGMat | `preprocessing/{mumtaz,eegmat}.py`(commit 4ca79f3),预处理任务 381007 |

候补池(哪个实测崩了就换):BCI-IV-2a、PhysioNet-MI(预训练翻盘才进)、
TUSL、SHU-MI、SEED-VIG。FACED 已除名(15 个变体全在随机附近,原因未知)。

协议注意:Mumtaz/EEGMat 对齐 CBraMod 的预处理(5 s 窗、19 通道 10-20),但
划分改为我们的被试不相交标准 —— CBraMod 的 Mumtaz 划分按文件排序切,同一
被试的 EC/EO 会跨 train/test(泄漏),我们不抄。

## 5. 在跑 / 排队(amd)

| 任务 | 内容 | 状态 |
|---|---|---|
| FL_long / FL_short(379887/379886) | 旗舰臂收尾(TUEV/TUSZ/CHB/PMI 旗舰;作用已变为补齐"全家桶失败"的证据链) | 跑了 ~10 h,CHB 臂有 24 h 撞墙风险 |
| DUP_slate_long(380980) | tuab/tuep/isruc/sleepedf duplex scratch | 排队 |
| DUP_slate_short(380981) | tuar/adfd/apava duplex scratch | 排队 |
| prep_mumtaz_eegmat(381007) | 两语料预处理 | 排队 |

全部落地后:12 数据集 × 预期表现判决表 → 骨干定稿写入本文件与 `FINDINGS.md`。

## 6. 预训练点火条件

b2 余额 ~147 SU,rung-1(duplex 骨干,base 档)约 110 SU —— **基本一发定音**,
所以点火前置条件全部客观化:

1. FL 波落地,旗舰证伪证据链闭合(骨干定稿不再变);
2. 7 个新语料 duplex scratch 数字回来,没有输十几个点的格子(有则先换名单);
3. Mumtaz/EEGMat 处理完成并测过 scratch;
4. duplex 预训练的配对行掩码实现并过 verify(`triaxial.py` 目前对
   hybrid/duplex + `return_amp_target` 是显式 raise 的守卫);
5. Zhizhe 批准动用 SU。

## 7. 遗留问题

1. FACED 完全学不动(已除名,但原因未知 —— CBraMod scratch 0.2469 我们只有
   0.24 上下,其预训练 0.5509 说明差距主要在预训练)。
2. duplex/hybrid 的**配对行掩码**未实现(预训练侧;守卫已在,见 §6.4)。
3. BCI/PMI 的最终判定悬在预训练之后。
4. 小语料过拟合调参波(aug/wd/patience)只有 PMI +0.021、FACED +0.018,
   BCI 无改善 —— 调参不是出路,结构才是(已并入判决表)。
5. `scripts/audit_runs.py` 只列 A 组 45 格;新语料的 A 组 baseline
   (TUEP/TUAR/ADFD/APAVA/Mumtaz/EEGMat)是未来阶段,费用未排。
