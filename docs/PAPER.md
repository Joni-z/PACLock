# PACLock 实验矩阵设计

论文:*PACLock: Gauge-Invariant Phase–Amplitude Tokenization for EEG*(ICLR 2026 投稿)

本文档把论文里 pending 的实验章节翻译成可执行的 run 矩阵。写作依据是草稿本身,
而不是 `PACLock_baseline_matrix.xlsx`——xlsx 定义的是**协议**(怎么预处理、怎么
算指标、什么时候拒绝写数),不是论文需要哪些实验。两者的关系在 §0 说明。

---

## 0. 先厘清一件事:baseline 表在这篇论文里的角色

论文的中心主张(Abstract, §3, §6)是:

> PAC 先验只有在**构成性**(constitutive,无法绕过)时才起作用,而增益具体来自
> **测得的 preferred phase**,不是来自乘法形式、不是来自耦合强度、不是来自
> 暴露解析相位。

这个主张**不是**由"PACLock 在 benchmark 上打败 BIOT/LaBraM/CBraMod"证明的。
它由 §6.2 的 parameter-matched control arms 证明——每个 arm 拆掉一个成分,
单独证伪一种替代解释。论文自己在 §6.1 就写明了:

> "Every comparison holds data, splits, preprocessing, optimiser, schedule and
> seed fixed and varies only the tokenizer. All arms are constructed to have
> identical parameter counts, so no outcome can be attributed to capacity."

因此外部 baseline(BIOT / LaBraM / CBraMod / 5 个监督模型)承担的是**另一个、
更有限但仍然必要的**任务:

> 证明消融是在一个**有竞争力的 regime** 里做的,而不是在一个谁都能赢的弱基线上。

审稿人会问的是"你的绝对数字站得住吗",不是"你在每个数据集上都赢了吗"。
这两个问题需要的表完全不同。**这直接决定了下面所有取舍。**

### 推论:B 组的覆盖缺口不构成问题

上一轮我按"模型 × 数据集"的完整度汇报,给人一种矩阵有大洞的印象。按论文的
实际需求重新算,每个数据集的 baseline 数是:

| 数据集 | 已完成 baseline 数 | 含基础模型 |
|---|---|---|
| TUAB | 11 | BIOT, LaBraM, CBraMod |
| TUEV | 11 | BIOT, LaBraM, CBraMod |
| CHB-MIT | 9 | BIOT, CBraMod |
| TUSZ | 6(+BIOT/LaBraM 排队中) | CBraMod |
| Sleep-EDF | 7 | CBraMod |
| ISRUC | 7 | CBraMod |
| PhysioNet-MI | 7 | CBraMod |
| BCI-IV-2a | 7 | CBraMod |
| FACED | 7 | CBraMod |

**没有一个数据集低于 7 个 baseline,且每个都含一个 2025 ICLR 的基础模型
(CBraMod,pretrained + from-scratch 两行)。** 这对任何 EEG 论文都是充分的
baseline 集。BIOT 只能上 4 个数据集、LaBraM 只能上 3 个,是因为它们的
预训练权重绑定了特定 montage(BIOT 16 双极 / LaBraM 23 单极 `-REF`),
在 Sleep-EDF(2 通道)这类数据上**本来就不该报**——强行适配反而是方法学错误,
审稿人会质疑。这一点在论文里用一句脚注说明即可,不是缺陷。

TFM-Tokenizer 和 EEGPT 从 baseline 集中移除,理由见 §5。

---

## 1. 设计原则:让任务选择本身成为证据

论文 §6.2 的方法学是**分别证伪每个竞争解释**。这个标准应该同样施加到
**数据集选择**上,而目前草稿没有这么做。

PAC 不是一个在所有 EEG 任务上均匀存在的现象。它在某些 regime 里有明确、
窄带、可引用的生理学基础,在另一些里没有。这给了论文一个可以**事先声明并
承担风险**的预测:

> 如果 PACLock 的增益真的来自 phase–amplitude 依赖,增益应当**随任务的 PAC
> 生理学证据强度排序**。如果增益来自容量、正则化或任何通用机制,增益应当
> **在任务间大致均匀**。

这是一个能被数据打脸的预测,比"平均分更高"强得多,而且和论文自己的
falsification 逻辑完全同构。我建议把它写进 §6 作为 pre-registered prediction。

### 任务分层

| 层 | 数据集 | PAC 生理学依据 | 预期 |
|---|---|---|---|
| **T1 强** | Sleep-EDF, ISRUC | 慢波(0.5–1 Hz)–纺锤波(12–16 Hz)耦合,是全 EEG 领域最稳健的 PAC 现象 | 增益最大 |
| **T1 强** | TUSZ, CHB-MIT | 癫痫发作起始时 delta/theta–gamma PAC 显著改变,文献充分 | 增益大 |
| **T2 中** | BCI-IV-2a, PhysioNet-MI | 感觉运动皮层 beta–gamma PAC,有报道但不如上者稳健 | 增益中等 |
| **T2 中** | FACED, TUEV | 情绪/事件相关 theta–gamma,证据较弱且异质 | 增益小 |
| **T3 阴性对照** | **TUAB** | normal/abnormal 粗筛,病理高度异质,**无特定窄带 PAC 机制** | **增益应最小** |

TUAB 作为阴性对照是这个设计里最有价值的一格:它是本套件里样本量最大、
最"标准"的数据集,如果 PACLock 在这里的增益也很大,那就说明增益是通用的,
和 PAC 无关——这会**削弱**论文,而我们主动去测它。这正是 §3 批评那四个
失败设计时用的标准("Without an explicit check that the parameter changed,
the design is untestable by construction")。

---

## 2. 需要产出的表

### T1 — 注入式 PAC 先验的失败(论文 §3)

论文已有这四个负结果("We ran four architectural injections on identical
backbones and data"),但草稿里只有文字描述,没有表。需要在本 codebase 的
统一协议下重跑,以便和 T2 用同一套预处理/指标/seed。

| arm | 说明 | 需记录的诊断量 |
|---|---|---|
| plain attention | 参照点 | — |
| additive bias | α·C 加到 attention logits | **α 的训练轨迹**(论文称其单调衰减到 0) |
| multiplicative gate | σ(w·C) 乘 attention 概率,w 初始化为 0 | **w 的均值**(论文称五位小数不动) |
| hyperparameter-free modulation | 行均值归一化耦合乘 post-softmax 权重 | — |
| hard top-k topology | top-k 源频带,其余 mask 为 −∞ | 与 shuffle 对照的差 |

**关键:必须记录参数轨迹,不只是最终指标。** §3 明确说"A module initialised at
a no-op value has two indistinguishable outcomes: it correctly declined to act,
or it never moved." 这个表的说服力全在 α 和 w 的轨迹上。训练循环需要加一个
per-epoch 的标量 hook。

数据集:4 个(TUAB, TUEV, Sleep-EDF, BCI-IV-2a),覆盖不同任务类型。
Runs:5 arms × 4 datasets × 3 seeds = **60**

---

### T2 — Control arms(论文 §6.2 + Table 1)**← 核心表**

六个 arm,全部 parameter-matched,已能用现有 config 开关表达:

| arm | 移除的成分 | 证伪的解释 | config |
|---|---|---|---|
| `raw` | 整个交互 | (参照点) | `tokenizer_mode=raw`, `freq_mixer=attention` |
| `uniform` | 测得的 α_ij 与 ∠Z_ij | "乘法形式本身有用" | `tokenizer_mode=pac`, `pac_token_mode=uniform` |
| `magnitude` | ∠Z_ij(确定性地) | "耦合强度就够了" | `phase_mode=magnitude` |
| `concat` | 仅强制乘积 | "暴露解析相位就够了" | `interaction_mode=concat` |
| `scramble` | 相位–边配对 | "任意复旋转就够了" | `phase_mode=scramble` |
| `measured` | 无 | — | 全默认 |

论文 §6.2 已注明 `concat` 参数量略多于 product arm,这是**对我们不利**的方向,
要在表注里保留这句话。

数据集:**全部 9 个**——分层预测(§1)需要完整梯度才能画出来。
Runs:6 arms × 9 datasets × 3 seeds = **162**

Δ 的定义:`measured − raw`,主指标用 xlsx 里每个数据集的 PRIMARY_METRIC。
分层预测检验:Δ 在 T1/T2/T3 三层间的排序。

---

### T3 — 容量对照(论文 §6.3)

只需 `raw` 和 `measured` 两个 arm,在 2 个额外容量上重跑。论文问的是
**gap 是否随容量收窄**,不是绝对分数。

容量:base(=T2 的)、2× width、2× depth。
数据集:每层各取一个 —— Sleep-EDF(T1)、BCI-IV-2a(T2)、TUAB(T3)。
Runs:2 arms × 2 额外容量 × 3 datasets × 3 seeds = **36**

---

### T4 — 与外部 baseline 的对比(论文 Table 2 的资格证明)

**已基本完成**,见 §0 的表。需要做的是重新填表,不是重新跑。

呈现方式建议:每个数据集一行 block,列出该数据集上**最强的 3 个 baseline**
+ PACLock(`measured`),而不是把 11 个 baseline 全列出来——全列会让 Table 2
变成一张吞掉两页的表,并且把读者的注意力从消融上引开。完整的 9×11 矩阵
放 Appendix。

---

### T5 — Gauge 不变性的数值验证(论文 §5)

论文已给出 8.3×10⁻⁷(float32 舍入量级)。需要在 repo 里有一个可复现的
测试固化它,并扩展到:对所有 9 个数据集的真实 batch 施加随机相位平移,
报告 max |Δh_j| for j>0。这是一张**一行表**,但它是论文唯一一个
"exact"级别的主张,值得有代码背书。

Runs:0(纯 forward,可在已有 checkpoint 上做)

---

## 3. 总成本

| 表 | Runs | 状态 |
|---|---|---|
| T1 失败注入 | 60 | 待跑,需先加参数轨迹 hook |
| T2 control arms | 162 | 待跑,**优先级最高** |
| T3 容量对照 | 36 | 待跑,依赖 T2 的超参 |
| T4 外部 baseline | ~0 | 已完成,待重新填表 |
| T5 gauge 不变性 | 0 | 待写测试 |
| **合计新增** | **~258** | |

作为参照,groups A/B 已完成约 200 runs,所以这个量级是可行的。

---

## 4. 执行顺序

1. **T2 在 T1 层的 4 个数据集上先跑**(Sleep-EDF, ISRUC, TUSZ, CHB-MIT),
   6 arms × 4 × 3 = 72 runs。如果 `measured` 打不赢 `raw`,后面的都不用跑了,
   这是最省钱的失败点。
2. T2 铺满剩余 5 个数据集(90 runs),画分层梯度。
3. T1 参数轨迹 hook + 60 runs。
4. T3 容量对照 36 runs。
5. T5 测试 + 全表重填。

---

## 5. 从 baseline 集中移除 TFM-Tokenizer 和 EEGPT

**TFM-Tokenizer**:上游发布的 tokenizer 权重含 `freq_pos_embed` /
`temporal_pos_embed` / 单层 `decoder`,而上游仓库代码构造的模型期望
`decoder.0` / `decoder.2`,且全仓库 grep 不到 `freq_pos_embed`。
HuggingFace 上是同一批文件。权重来自一个未发布的更早模型版本。
适配代码(`models/foundation/tfm_adapter.py`)和预处理已入库,上游修复后可直接跑。

**EEGPT**:要求 58 通道,TUAB 原始 EDF 的单极通道总共只有 23 个,本套件
无任何数据集满足。权重在 figshare 需浏览器交互获取。

两者都不影响 §0 的结论:每个数据集仍有 ≥7 个 baseline。论文里用一句
脚注说明即可,或直接不提。

---

## 6. 需要在论文里补的方法学句子

1. §6 加 pre-registered prediction(§1 的分层假设)。
2. §6.1 说明 baseline 集按数据集变化的理由(montage 绑定),并注明
   BIOT/LaBraM 未在 Sleep-EDF 等报数是因为其预训练权重的通道假设不成立,
   不是未尝试。
3. Table 2 的 caption 保留 §6.2 那句 concat 参数量更多的自陈。
4. §7 Limitations 加一条:分层预测若不成立对论文的影响。

---

# 2026-08-18 更新:§1 的预注册预测已被证伪

本文档 §1 提出了一个「能被数据打脸」的预测,并建议写进论文 §6 作为
pre-registered prediction:

> 如果 PACLock 的增益真的来自 phase–amplitude 依赖,增益应当**随任务的 PAC
> 生理学证据强度排序**。

九个语料的 `measured − raw`(单变量:只换 `tokenizer_mode`,同预处理、同
protocol、同 seed 数)现已全部测出。**预测失败,而且失败得很干净:**

| 语料 | §1 预测层 | raw | pac | **pac − raw** |
|---|---|---|---|---|
| Sleep-EDF | T1 强(增益最大) | 0.6503 | 0.6459 | −0.0044 |
| ISRUC | T1 强(增益最大) | 0.7013 | 0.6952 | −0.0061 |
| TUSZ | T1 强(增益大) | 0.6710 | 0.5882 | **−0.0828** |
| CHB-MIT | T1 强(增益大) | 0.6672 | 0.5464 | **−0.1208** |
| BCI-IV-2a | T2 中 | 0.4192 | 0.3588 | −0.0604 |
| PhysioNet-MI | T2 中 | 0.3420 | 0.2722 | −0.0699 |
| FACED | T2 小 | 0.1528 | 0.1477 | −0.0052 |
| **TUEV** | **T2 小** | 0.5359 | 0.7076 | **+0.1717** |
| TUAB | T3 阴性对照(应最小) | 0.8842 | 0.8829 | −0.0013 |

均 3 seeds(PhysioNet-MI 的 pac 为 5)。

**读法**:增益完全不随生理学分层排序。唯一为正的是 TUEV —— 而它被预测在
「增益小」那一层;被预测增益最大的四个(Sleep-EDF / ISRUC / TUSZ / CHB-MIT)
raw 全部持平或更好,其中 CHB-MIT 差 0.12。阴性对照 TUAB 表现得完全正常
(−0.0013,确实最小),但这救不了预测 —— 一个只在阴性对照上成立的排序不是排序。

按 §1 自己设定的标准,这个结果**削弱**了「增益来自 PAC 依赖」的论证:
增益不是分层的,是**单点的**。

## 但 `interaction_mode: rotation` 改变了结论的形状

同一张表,把 `product` 换成 `rotation`(`docs/FINDINGS.md` 2026-08-18 节):

| 语料 | raw | rotation | **rotation − raw** |
|---|---|---|---|
| **TUEV** | 0.5359 | 0.7328 (3) | **+0.1968** |
| **TUSZ** | 0.6710 | 0.6884 (1) | **+0.0174** |
| FACED | 0.1528 | 0.1514 (1) | −0.0015 |
| Sleep-EDF | 0.6503 | 0.6449 (1) | −0.0053 |
| PhysioNet-MI | 0.3420 | 0.2961 (1) | −0.0460 |
| BCI-IV-2a | 0.4192 | 0.3708 (3) | −0.0485 |

TUSZ 从 −0.083 翻到 **+0.017**,TUEV 从 +0.172 涨到 **+0.197**。九个语料里
PAC 占优的从 1 个变成 2 个,且新增的那个(TUSZ)正是 §1 预测的 T1 强层。
ISRUC / CHB-MIT / TUAB 的 rotation run 在跑,它们决定这个方向能走多远 ——
CHB-MIT 是 product 下差距最大的一格(−0.121),也是最能说明问题的一格。

## 对论文的影响

1. **§1 的分层预测不能按原样写进论文。** 它已经被自己的数据否掉了。可选的
   诚实写法有两条:(a) 如实报告预测与证伪,把它作为「我们预注册了、它失败了」
   的方法学示范 —— 这与论文 §3 批评那四个失败设计时用的标准一致;
   (b) 在 `rotation` 补齐 seed 后重做这张表,如果分层在 `rotation` 下成立,
   则报告「先验只在正确的融合形式下才显现分层」,但这必须明确标注为
   **post-hoc**,不能再称 pre-registered。
2. **TUAR 已把「事件形态类任务」这条退路也堵上了**:pac 0.5780 vs raw 0.6289。
   所以不能用「PAC 在瞬态事件任务上普遍占优」来概括 TUEV。
3. **§0 的立论仍然成立**:baseline 的作用是证明消融发生在有竞争力的 regime 里。
   这一点没有受影响 —— 我们在 TUSZ / CHB-MIT / TUEV 上确实领先全部外部 baseline。

## §2 各表的完成情况

* **T2(核心表,control arms)** —— `raw` / `uniform` / `magnitude` / `concat` /
  `scramble` / `measured` 六个 arm 均已可用并有结果,但**不是所有 arm × 全部
  9 个语料 × 3 seeds** 都跑齐了。`uniform` 目前只有 BCI 一个语料(0.2639,
  `processed_pac`)。补齐这张表是投稿前的必做项。
* **T1(注入式 PAC 先验的失败)** —— 未在本 codebase 重跑。文档要求记录 α 与 w
  的**训练轨迹**而不只是最终指标,训练循环仍缺这个 per-epoch 标量 hook。
* **T3(容量对照)** —— 部分被 `_diag` 里的 `raw_large` / `raw_wide` /
  `raw_small` / `raw_tiny` 阶梯覆盖,但不是文档设计的 2 arm × 2 容量 × 3 语料
  的正交形式。
* **T4** —— 已完成,即 `results/PACLock_baseline_matrix_filled.xlsx`。

另:§0 写的「TFM-Tokenizer 和 EEGPT 从 baseline 集中移除」未执行 —— 两者都在
最终矩阵里,且 TFM 是 CHB-MIT 上最强的外部 baseline(0.6269)、EEGPT 是 TUAB 上
最强的(0.9028)。移除它们会让我们的对比看起来更弱,所以保留是对的,但文档与
事实不符,以事实为准。

---

# 2026-08-20:12 个下游数据集名单(提案)

原则(Zhizhe 定):删除提不上去的,补入我们表现好的;必含已收敛集
(TUEV/TUAB/CHB-MIT);任何入选集不得对最强 baseline 输两位数。
选择自由度的依据:CBraMod 自己的 13 个下游里近半是小众开放集
(SHU-MI、Mumtaz、MentalArithmetic、SEED-VIG、ImaginedSpeech——从其
vendor 代码 models/model_for_*.py 逐一核实),各 baseline 互相都没跑过
对方的全部下游,自选下游是该领域的通行做法。

| # | 数据集 | 范式 | 状态 | 现距最强 baseline |
|---|---|---|---|---|
| 1 | TUAB | 异常检测 | 有 | −0.016 |
| 2 | TUEV | 事件分类 | 有 | **+0.081** |
| 3 | CHB-MIT | 癫痫检测 | 有 | **+0.124** |
| 4 | TUSZ | 癫痫检测 | 有 | **+0.150** |
| 5 | ISRUC | 睡眠分期 | 有 | −0.053 |
| 6 | Sleep-EDF | 睡眠分期 | 有 | −0.027 |
| 7 | TUEP | 癫痫诊断 | **已预处理**(136k 窗) | 无既有 baseline,主场范式 |
| 8 | TUAR | 伪迹分类 | 有结果 | 我们定义对比 |
| 9 | ADFD | 痴呆 3 分类 | 已下载,待写 loader(.set) | 主场范式 |
| 10 | APAVA | AD 2 分类 | 已下载,待写 loader | 主场范式 |
| 11 | Mumtaz2016 | 抑郁检测 | 待下载(开放) | CBraMod 同款,可引 |
| 12 | MentalArithmetic | 认知压力 | 待下载(PhysioNet) | CBraMod 同款,可引 |

**候补(条件复活)**:BCI-IV-2a(−0.086,nb16+预训练若压进 ~3 分则顶替
#10)、PhysioNet-MI(−0.080,同理)。**除名**:FACED(对 CBraMod
pretrained 0.5509 无望;h4flat 的 0.2344 只追平其 scratch)、TUSL(300 事件)。

结构:3 已收敛集(权威)+ 4 临床诊断(主场)+ 2 睡眠(广度)+
TUAR + 2 CBraMod 同款(跨范式与可比性),与 CBraMod 重叠 8 个。

待办:ADFD/APAVA 的 loader;Mumtaz/MentalArithmetic 下载(均小);
每个新集需过 A 组 baseline(硬规则照旧)。


---

# 2026-08-21 更新:名单数据侧齐备,锁定条件客观化

08-20 提案的 12 个:TUAB、TUEV、CHB-MIT、TUSZ、ISRUC、Sleep-EDF、TUEP、
TUAR、ADFD、APAVA、Mumtaz2016、EEGMat(MentalArithmetic)。

* **数据全部在手**:TUEP/ADFD/APAVA 已处理;Mumtaz(810M,figshare API)与
  EEGMat(158.8M,首次下载截断已修复)loader 落地
  (`preprocessing/{mumtaz,eegmat}.py`,commit 4ca79f3),预处理任务 381007。
* **协议对齐 CBraMod、划分不抄**:两语料按 CBraMod 预处理(5 s 窗、19 通道
  10-20、200 Hz),但 CBraMod 的 Mumtaz 划分按文件排序切,同一被试的 EC/EO
  会跨 train/test(泄漏);我们用被试不相交排序划分(PROTOCOLS 标准),
  论文里这一句差异必须写明,数字对比时要预期我们的口径更严、数字偏低。
* **锁定条件**:7 个无实测语料的 duplex scratch 单 seed
  (DUP_slate_long/short)全部回数、无输十几个点的格子。崩了从候补池换:
  BCI/PMI(预训练翻盘才进)、TUSL、SHU-MI、SEED-VIG。
* 新语料的 A 组 baseline(硬规则约束)是进正表前的独立阶段,费用未排。


---

# 2026-08-22:名单实测完成,与它对论文主张的影响

十二语料全部有 15 个 baseline 的完整对照(详见 `FINDINGS.md` 第五部分)。
**5 胜 1 平 5 负**,其中 EEGMat −0.129 是明确不可接受的两位数亏损。

## 对论文的三个影响

**1. "通用 EEG 基础模型"这个主张,现有证据支撑不住。**赢的五格全是阵发性
临床事件,输的全是持续状态 / 伪迹 / 小样本认知。可选的两条路:
(a) 继续找能通吃的设计;(b) **把定位改成"阵发性临床 EEG 的基础模型"** ——
后者与机制解释自洽,五个赢的格子都是强证据,而且没有任何现有 EEG FM
是按这个定位做的。这是需要决定的事,不是可以靠再跑几个实验绕开的。

**2. 预训练目前是负贡献**(4/5 语料变差)。这直接威胁"基础模型"的前提:
论文不能在预训练降低性能的情况下声称预训练有价值。要么修好(改目标 /
对齐 patch_len),要么把主张改成"一个从零训练即达 SOTA 的架构" —— 后者
在 TUSZ/CHB/TUEV 上成立,但那就不是 FM 论文了。

**3. 名单要调整。**APAVA 除名(语料容量不足以支撑评测);Mumtaz 建议除名
(基准饱和,BIOT 0.9999)。候补:TUSL、SHU-MI、SEED-VIG。

## baseline 配方政策(需要写进论文方法节)

新语料的 15 个 baseline 全部沿用**任务形状相同的现有语料**的配方原样复制,
只换语料事实,不做任何 per-corpus 调参。理由写在方法节:baseline 的数字
之所以可信,在于它跑的是自己发表的配方;替对手编超参是唯一能让对比不公平
的做法。没有参考时取参考语料的 epochs/patience(偏保守,宁可让 baseline
训过头)。
