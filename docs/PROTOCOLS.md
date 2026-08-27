# 冻结的预处理与评测协议

本文件逐条转录自 `PACLock_baseline_matrix.xlsx`(2026-08-04 版本)。

**本文件是仓库内的事实来源。** 代码与本文件冲突时,以本文件为准;本文件与 xlsx
冲突时,以 xlsx 为准。任何一条标记「冻结」的参数,不得在实验过程中修改——需要
修改时先改 xlsx,再改本文件,再改代码,并在 `docs/CHANGELOG.md` 记录原因。

> **注意:参考仓库 `Joni-z/PACLock` 的预处理协议与本文件不同。**
> 那个仓库沿用 BIOT 的协议(随机 shuffle 划分 + per-channel q95 归一化);
> 本工作簿采用 CBraMod 协议(带通+notch + subject 排序划分 + ÷100 归一化)。
> **不要从参考仓库拷贝 `scripts/preprocess_*.py`。**

---

## 0. 硬规则(README sheet)

1. **复现门**:B 组(官方预训练权重)任一模型进表前,先用它自己的 recipe 复现自己
   论文的数字。跑 ≥3 seeds,要求发表值落在 `mean ± 2·std` 内。过不了就标
   `not reproduced`,不并列进表。
2. **每个模型用自己的 recipe**(B、D 组);架构对照用对称超参(C 组)。对称 ≠ 公平。
3. **epoch-0 峰值拒绝**:任何 baseline 的 val 曲线峰值在 epoch 0 → 标
   `mis-configured`,拒绝写入。(TFM-Tokenizer 论文里 CBraMod 的 TUAB 行就是这种:
   BAcc 0.5000±0.0000。)
4. **全部 3 seeds**,报 `mean ± std`。single seed 的数字形式上不能进表。
5. **所有数据集必须按本文件协议从零生成**;不得复用来源不明的旧缓存。

### 行分组

| 组 | 含义 | 要求 |
|---|---|---|
| A 轻量监督基线 | pipeline 校准用 | 对不上 TFM-Tokenizer(ICLR26) 已发表值 ⇒ 问题在 pipeline 不在模型。**先跑这一组。** |
| B FM · 官方预训练权重 | 各自 repo 的预处理 + 归一化 + finetune recipe | 不要塞进我们的预处理(0.6772→0.4436 就是这么来的) |
| C FM · 同 pipeline from-scratch | 论文主表 | 对称超参、参数量对齐 |
| D PACLock | 仅 from-scratch 完整体 | 当前阶段不含 raw / 容量对照 / 消融 / 预训练 |

### 当前阶段范围

只跑 baseline 与 PACLock from-scratch 完整体。预训练、消融和容量对照留到后续阶段。

---

## 1. TUAB — 异常 EEG 检测(二分类)

**主指标:AUROC**;同时报告 Balanced Accuracy、PR-AUC。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据版本 | TUAB v3.0.1 | TUH 官方下载页 |
| 原始规模 | 2,994 recordings;train 2,718(normal 1,371 / abnormal 1,347),eval 276(normal 150 / abnormal 126);约 2,329 unique subjects | PMC12052170 |
| 原始通道/采样率 | 记录间可变;通常 27–36 EEG channels;250/256/512 Hz,以 250 Hz 为主 | Banville 2022 |
| 标签 | recording-level normal / abnormal 二分类 | TUH 官方 |
| 官方 split | train / eval;受试者不重叠;eval 作为 test | TUAB 数据说明 |
| **使用通道** | 16 个双极导联:`FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, F8-T4, T4-T6, T6-O2, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2` | CBraMod `preprocessing_tuab.py` |
| 目标采样率 | 200 Hz | CBraMod |
| 滤波 | 0.3–75 Hz band-pass;60 Hz notch;转换为 μV | CBraMod |
| 窗口/stride | 10 s / 10 s,尾部不足 10 s 丢弃 | CBraMod |
| train/val/test | 官方 train 内,normal 与 abnormal **分别按 subject ID 排序**后前 80%/后 20%;官方 eval=test | 冻结 |
| 归一化 | 每段数值 **除以 100**(输入单位 μV) | CBraMod `tuab_dataset.py` |
| 类别不平衡 | 不加 class weight,不做重采样;**BCEWithLogitsLoss** | CBraMod `finetune_trainer.py` |
| PR-AUC 定义 | `precision_recall_curve` 后对 recall–precision 梯形积分 `auc(recall, precision)` | CBraMod `finetune_evaluator.py` |
| manifest 产物 | train/val/test subject IDs、recording IDs、各 split/类别窗口数、原始文件 SHA256 | 冻结 |

> **实现注记(不改协议).** 「分别按 subject ID 排序」是按类独立切分,而 TUAB 官方
> train 中有 54 名受试者同时拥有 normal 与 abnormal 记录,因此有 3 名受试者会跨
> train/val。BIOT 与 CBraMod 的官方预处理同样按类切分,已发表数字均由此产生,故
> **保持通行做法**,重叠如实记入 manifest。官方 eval(=test)与 train 受试者零重叠,
> test 指标不受影响。详见 `docs/CHANGELOG.md`。

---

## 2. TUEV — 事件分类(6 类)

**主指标:Cohen's Kappa**;同时报告 Balanced Accuracy、Weighted F1。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据版本 | TUEV v2.0.1 | TUH 官方 |
| 受试者 | 390 patients | medRxiv 2025.05.23 |
| 原始通道/采样率 | 23 EEG channels,256 Hz | openreview 66h1sCMm7F |
| 标签 | SPSW, GPED, PLED, EYEM, ARTF, BCKG 六类 | TUH 官方 |
| 官方 split | train / eval,受试者不重叠;eval 作为 test | PMC6423064 |
| 使用通道 | 与 TUAB **相同**的 16 个双极导联及顺序 | CBraMod `preprocessing_tuev.py` |
| 目标采样率 | 200 Hz | CBraMod |
| 滤波 | 0.3–75 Hz band-pass;60 Hz notch;转换为 μV | CBraMod |
| **窗口** | 每条约 1 s annotation 取**事件前 2 s + 事件 + 后 2 s**,形成 5 s event-centered sample | CBraMod |
| stride | 不适用:每条 annotation 生成一个样本,不是连续滑窗 | CBraMod |
| 样本规模 | 112,491 个 5 s 样本 | openreview NPNUHgHF2w |
| train/val/test | 官方 train subjects 按 ID 排序后前 80%/后 20%;官方 eval=test | 冻结 |
| 归一化 | 每段数值除以 100 | CBraMod `tuev_dataset.py` |
| 类别不平衡 | **不使用逆频率权重**;unweighted CE,label smoothing=0.1 | CBraMod trainer(原表「逆频率 CE 权重」已作废) |
| manifest 产物 | train/val/test subject IDs、recording/annotation IDs、各 split/类别样本数、原始文件 SHA256 | 冻结 |

---

## 3. TUSZ — 癫痫发作检测(二分类)

**主指标:PR-AUC**(CBraMod 梯形定义);同时报告 Balanced Accuracy、AUROC。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据版本 | TUSZ v2.0.6 | TUH 官方 |
| 原始规模 | 约 6,100 recordings、约 600 subjects;精确计数以下载后的 v2.0.6 manifest 为准 | medRxiv |
| 原始通道/采样率 | 随记录变化;多数 250 Hz,亦有 256/400/512 Hz | PMC4865520 |
| 标签 | seizure onset/offset、channel 和 seizure type;二分类使用 seizure / background | TUH 官方 |
| 官方 split | subject-disjoint train / dev / eval → 映射为 train / val / test | v2.0.6 资料 |
| 使用通道 | 与 TUAB 相同 16 双极导联和顺序;**缺任一必需原始通道的记录整条排除并记日志** | 冻结 |
| 目标采样率 | 200 Hz | 冻结 |
| 滤波 | 0.3–75 Hz band-pass;60 Hz notch;转换为 μV | 冻结 |
| 窗口/stride | 10 s / 10 s;尾部不足 10 s 丢弃 | 冻结 |
| **标签规则** | 窗口与任一 seizure interval 的交集长度 **> 0** 即标正;边界按**半开区间 [start,end)** 实现 | 冻结 |
| 负样本采样 | 保留全部负样本,不下采样;不使用 class weight;**BCEWithLogitsLoss** | 冻结 |
| 归一化 | 每段数值除以 100 | 冻结 |

---

## 4. CHB-MIT — 小儿癫痫发作检测(二分类,~1% 正样本)

**主指标:PR-AUC**;同时报告 Balanced Accuracy、AUROC。

> 采用**严格 subject-disjoint** benchmark;**修正**窗口重叠标签规则。
> TFM 已发表值仅作外部校准,**不做逐点复现**。论文中必须注明 split 与标签规则不同。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据版本 | CHB-MIT v1.0.0 | physionet.org/content/chbmit/1.0.0/ |
| 原始规模 | 23 cases、22 subjects;664 EDF;198 seizures;**chb01 与 chb21 为同一受试者** | PhysioNet 官方 |
| 原始通道/采样率 | 多数文件 23 EEG signals(少数 24/26),256 Hz,16-bit | PhysioNet 官方 |
| 标签 | seizure onset/end interval;窗口映射为 seizure / non-seizure | PhysioNet 官方 |
| 官方 split | **不存在** | PhysioNet 未发布 |
| **使用通道** | 16 双极导联:`FP1-F7, F7-T7, T7-P7, P7-O1, FP2-F8, F8-T8, T8-P8, P8-O2, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2` | TFM `process_2.py` |
| 目标采样率 | 200 Hz(从 256 Hz 在 loader 中重采样) | TFM `data_loaders.py` |
| **窗口/stride** | 基础窗口 10 s / 10 s;**每个 seizure 从 onset−1 s 到 offset+1 s 以 5 s stride 增补 10 s 正样本** | TFM official code |
| **窗口标签** | 基础 10 s 窗口与任一 seizure interval 的交集长度 > 0 即标正;边界按半开区间 `[start,end)`;**增补窗口一律标正** | 修正 TFM 原实现「发作完整覆盖窗口」的标签缺陷,所有模型统一使用 |
| **最终 split** | train:`chb01–chb19 + chb21`;val:`chb20, chb22`;test:`chb23, chb24` | chb01 与 chb21 同一受试者放入同一 split,保证受试者不重叠 |
| 归一化 | 每通道除以**该窗口绝对值 95% 分位数** + 1e−8 | TFM loader |
| 类别不平衡 | **focal loss**;不额外采样 | TFM `downstream_transformer_finetuning.py` |

---

## 5. Sleep-EDF — 睡眠分期(5 类)

**主指标:Cohen's Kappa**;同时报告 Balanced Accuracy、Weighted F1;按所有 test epochs 汇总一次。

> 无可比的已发表数字(BIOT / LaBraM / CBraMod / TFM-Tokenizer 均未报此数据集)。仅做内部比较。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据版本/范围 | Sleep-EDF Expanded v1.0.0,**仅 Sleep Cassette(SC)子集**:78 subjects、153 PSG recordings | physionet.org/content/sleep-edfx/1.0.0/ |
| 原始通道/采样率 | EEG Fpz-Cz、Pz-Oz;100 Hz | PhysioNet 官方 |
| 原始标签 | W, R, 1, 2, 3, 4, M, ?;R&K 人工标注 | PhysioNet 官方 |
| 官方 split | 不存在 | PhysioNet 未发布 |
| 使用通道 | Fpz-Cz、Pz-Oz,**顺序固定** | 冻结 |
| 目标采样率 | **100 Hz,不重采样** | 冻结 |
| epoch/stride | 30 s / 30 s,与 hypnogram 边界对齐 | 冻结 |
| **标签映射** | `W→0, 1→1, 2→2, 3/4→3 (N3), R→4`;**M 与 ? 丢弃** | 冻结 |
| **清醒段裁剪** | 仅保留**首个睡眠 epoch 前 30 min** 至**末个睡眠 epoch 后 30 min** | 冻结 |
| split | 按 SC subject number 排序:train 前 54 人;val 接续 12 人;test 最后 12 人 | 确定性 54/12/12 |
| train IDs | `00–38`(不含 39),`40–54` | 冻结 |
| val IDs | `55–66` | 冻结 |
| test IDs | `67,70,71,72,73,74,75,76,77,80,81,82` | 冻结 |
| 滤波 | 0.3–35 Hz band-pass;**不额外 notch** | 冻结 |
| **归一化** | 每通道使用 **train subjects 全部保留 epoch** 计算 mean/std;val/test **只应用 train 统计量** | 冻结 |
| 类别不平衡 | 不加 class weight,不重采样;unweighted CE + label smoothing 0.1 | 冻结 |

---

## 6. ISRUC — 睡眠分期(5 类)

**主指标:Cohen's Kappa**;同时报告 Balanced Accuracy、Weighted F1;test 序列的 epoch 预测展平后汇总。

> 无可比的已发表数字。仅做内部比较。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据范围 | ISRUC-Sleep **Subgroup I**:100 subjects、每人 1 个夜间记录 | doi 10.1016/j.cmpb.2015.10.013 |
| 原始通道/采样率 | `F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1`;200 Hz | PMC10557479 |
| 标签 | W, N1, N2, N3, REM(原文件代码 `0,1,2,3,5`) | CBraMod mapping |
| 官方 split | 不存在 | 原数据未发布 |
| split | subject `1–80` train,`81–90` val,`91–100` test | CBraMod `isruc_dataset.py` |
| 通道/采样率 | 上述 6 通道;**保持 200 Hz** | CBraMod |
| 滤波 | 0.3–35 Hz band-pass;**50 Hz notch** | CBraMod `prepare_ISRUC_1.py` |
| **epoch/序列** | 30 s non-overlap epoch;**每 20 个连续 epoch 组成一条序列**;每个 subject 尾部不足 20 epoch 丢弃 | CBraMod |
| 归一化 | 数值除以 100 | CBraMod loader |
| 类别不平衡 | 不加 class weight,不重采样;unweighted CE + label smoothing 0.1 | CBraMod trainer |
| **样本数 QC** | **不预写 89,240**;处理后输出各 split/各类 epoch 数及丢弃尾部数,与 manifest 一起冻结 | 原固定样本数证据不足 |

---

## 7. PhysioNet-MI — 运动想象分类(4 类)

**主指标:Balanced Accuracy**;同时报告 Cohen's Kappa、Weighted F1。

> 外部论文协议差异较大(2/3/4 类、窗口和划分不一),暂不直接抄单一数字。
> 先固定 subject-independent 4 类协议,再做内部 head-to-head。

| 参数 | 最终值 | 来源 |
|---|---|---|
| 数据版本 | EEG Motor Movement/Imagery Dataset v1.0.0 | physionet.org/content/eegmmidb/1.0.0/ |
| 原始规模 | 109 subjects;每人 14 runs | PhysioNet 官方 |
| 原始通道/采样率 | 64 EEG channels,160 Hz | PhysioNet 官方 |
| 原始标签 | T0=rest;T1/T2 含义依 run 而变 | PhysioNet 官方 |
| 官方 split | 不存在 | PhysioNet 未发布 |
| **任务/runs** | 仅 motor imagery runs `04,06,08,10,12,14`;排除 movement runs 和 baseline runs | CBraMod `preprocessing_physio.py` |
| **四类映射** | `04/08/12`:T1=left fist, T2=right fist;`06/10/14`:T1=both fists, T2=both feet;**T0 丢弃** | PhysioNet 标签定义 + CBraMod |
| split | `S001–S070` train,`S071–S089` val,`S090–S109` test | CBraMod |
| 通道 | 64 通道,按 CBraMod `selected_channels` 固定顺序;**average reference** | CBraMod |
| 目标采样率 | 200 Hz | CBraMod |
| 滤波 | 0.3 Hz high-pass(**无额外 low-pass**);60 Hz notch | CBraMod |
| trial/stride | 每个 annotation onset 起取 4 s(800 points @200 Hz);**不做 baseline correction**;stride 不适用 | CBraMod |
| 归一化 | 输入 μV 后除以 100 | CBraMod `physio_dataset.py` |
| 类别不平衡 | 不加 class weight,不重采样;unweighted CE + label smoothing 0.1 | CBraMod trainer |

---

## 8. FACED — 情绪识别(9 类)

**主指标:Balanced Accuracy**;同时报告 Cohen's Kappa、Weighted F1。

> 2026-08-05:本节由 SEED-V 替换为 FACED(见 `docs/CHANGELOG.md`)。
> FACED 协议转录自 xlsx 的初版,该版本已包含完整的 FACED sheet。

> FACED 原始论文提供数据说明与基准,但不同工作在切窗、标签聚合和跨受试者协议上
> 差异明显。主表只放同一 pipeline 重跑结果;发表值留作校准。

| 参数 | 最终值 | 来源 / 执行要求 |
|---|---|---|
| 数据版本 | FACED Synapse syn50614194;**使用官方发布的 pre-processed `.pkl`** | doi 10.7303/syn50614194 |
| 原始规模 | 123 subjects(S000–S122),32 electrodes,28 videos | Nature s41597-023-02650-w |
| 原始采样率 | raw 为 250 或 1000 Hz;**官方 pre-processed 统一为 250 Hz** | 官方论文 |
| **官方预处理** | 每视频取末 30 s;统一 250 Hz;0.05–47 Hz;坏导联插值;ICA 去眼动;common-average reference;统一通道顺序 | 冻结;官方论文。**我们不得再滤一次** |
| 标签 | anger, disgust, fear, sadness, neutral, amusement, inspiration, joy, tenderness 九类;视频数 **3,3,3,3,4,3,3,3,3** | 官方论文 + CBraMod label array |
| 官方 split | 无固定 train/val/test;原论文 cross-subject 10-fold | 官方论文 |
| **benchmark split** | `S000–S079` train,`S080–S099` val,`S100–S122` test | CBraMod `preprocessing_faced.py` |
| 使用通道 | 官方 pre-processed 32 通道及其发布顺序 | 官方论文 |
| 目标采样率 | 250 Hz → **200 Hz** | CBraMod |
| **窗口/stride** | 每个 30 s trial 切为 **3 个不重叠 10 s 窗口**;stride=10 s | CBraMod |
| **数据泄漏约束** | **先按 subject split,再切窗**;同一 subject/video 的窗口只能属于一个 split | 冻结 |
| 归一化 | 数值除以 100 | CBraMod loader |
| 类别不平衡 | 不加 class weight,不重采样;unweighted CE + label smoothing 0.1 | CBraMod trainer |

---

## 9. BCI Competition IV-2a — 运动想象(4 类)

**主指标:Balanced Accuracy**;同时报告 Cohen's Kappa、Weighted F1。
按全部 9 名受试者的 test trials 汇总一次;另保存 per-subject 指标用于附录。

| 参数 | 最终值 | 来源 / 执行要求 |
|---|---|---|
| 数据集/获取 | BCI Competition IV Dataset 2a;优先通过 **MOABB `BNCI2014_001`** 自动下载并缓存 | 官方:22 EEG + 3 EOG,9 subjects,2 sessions,250 Hz |
| 任务 | 4 类:left hand / right hand / both feet / tongue | 仅使用四类 MI trials;丢弃非任务和 EOG 事件 |
| 通道 | **22 个 EEG 通道**,官方顺序;丢弃 3 个 EOG 通道 | manifest 保存 `channel_names` 和 `channel_order`;任一文件缺通道则**停止** |
| 采样率 | 250 Hz → **200 Hz** | `resample_poly` 或 MNE resample;全部模型读取同一 processed data |
| 滤波 | 0.5–75 Hz band-pass;**50 Hz notch**;转换为 μV | 原采集已有 0.5–100 Hz + 50 Hz notch;处理脚本仍显式执行并记录参数 |
| **trial 窗口** | **cue onset 后 0–4 s**,即原 trial 时间 `t=2–6 s`;800 points @200 Hz | **不使用 cue 前 baseline**;每个 trial 生成一个样本 |
| 官方 session | **Session T 为开发集;Session E 为最终 test** | 保持跨 session 评测,**不把 Session E 用于调参** |
| train/val | 每个 subject 的 Session T:**run 1–5 train,run 6 val** | 确定性 run-level split;每人 train 240 trials、val 48 trials;避免同一 run 跨 split |
| test | 每个 subject 的 Session E 全部 288 trials | **最终 checkpoint 仅评估一次**;不按 test 结果调参 |
| 标签映射 | `left_hand=0, right_hand=1, feet=2, tongue=3` | manifest 保存原事件码与映射后标签 |
| 归一化 | 每 trial、每通道:输入 μV 后除以 100 | 不使用 test 统计量 |
| 类别不平衡 | 不加 class weight,不重采样;unweighted CE + label smoothing 0.1 | 四类设计均衡 |
| manifest/QC | subject、session、run、trial、label、processed_path、raw SHA256 | 检查每人 `T:288 / E:288` trials、`22×800` shape、split 交集为空、四类计数一致 |

---

## 附录 A:主指标与损失函数速查

| 数据集 | 类别数 | 主指标 | 损失函数 |
|---|---|---|---|
| TUAB | 2 | AUROC | BCEWithLogitsLoss |
| TUEV | 6 | Cohen's Kappa | CE + label smoothing 0.1 |
| TUSZ | 2 | PR-AUC | BCEWithLogitsLoss |
| CHB-MIT | 2 | PR-AUC | **focal loss** |
| Sleep-EDF | 5 | Cohen's Kappa | CE + label smoothing 0.1 |
| ISRUC | 5 | Cohen's Kappa | CE + label smoothing 0.1 |
| PhysioNet-MI | 4 | Balanced Accuracy | CE + label smoothing 0.1 |
| FACED | 9 | Balanced Accuracy | CE + label smoothing 0.1 |
| BCI-IV-2a | 4 | Balanced Accuracy | CE + label smoothing 0.1 |

## 附录 B:归一化方式速查

| 方式 | 数据集 |
|---|---|
| ÷100(μV) | TUAB, TUEV, TUSZ, ISRUC, PhysioNet-MI, FACED, BCI-IV-2a |
| per-window per-channel q95 | CHB-MIT |
| train-set mean/std | Sleep-EDF |

## 附录 C:采样率速查

| 采样率 | 数据集 |
|---|---|
| 200 Hz | TUAB, TUEV, TUSZ, CHB-MIT, ISRUC, PhysioNet-MI, FACED, BCI-IV-2a |
| 100 Hz(不重采样) | Sleep-EDF |

---

# 附录 A:语料来源与版本

*(原 `docs/PROTOCOLS.md`)*

Everything here is a download plus a version. The exact preprocessing that turns
each into `processed*/` is `docs/PROTOCOLS.md`, and is executed by
`slurm/preprocess.slurm <protocol> <dataset>`.

## Access

| corpus | version | access | note |
|---|---|---|---|
| TUAB | v3.0.1 | TUH EEG Corpus — **registration required**, credentials by email | abnormal/normal, recording-level |
| TUEV | v2.0.1 | same TUH account | 6-class events, 390 patients |
| TUSZ | v2.0.6 | same TUH account | seizure onset/offset, subject-disjoint splits |
| CHB-MIT | 1.0.0 | PhysioNet, open | paediatric seizures |
| Sleep-EDF | sleep-cassette | PhysioNet, open | |
| PhysioNet-MI | EEGMMIDB 1.0.0 | PhysioNet, open | motor imagery |
| ISRUC | Subgroup 1 | ISRUC-SLEEP site, open | `slurm/download_isruc.slurm` automates it |
| FACED | — | request from the authors | emotion, 32 channels |
| BCI-IV-2a | — | BNCI Horizon 2020, open | motor imagery, 22 channels |

The three TUH corpora share one account and are the only ones needing a human
in the loop. Apply first; everything else can be fetched while waiting.

## Versions are not optional

Each `processed*/<corpus>/manifest.json` records the **SHA256 of every source
file**, the subject IDs in each split, and the per-class window counts. If a
re-download differs — a corpus revision, a partial mirror, a different
subgroup — the arrays it produces are not comparable with any result already in
`runs/`, and the whole matrix has to be re-run rather than extended.

Check before trusting anything:

```bash
sbatch slurm/run.slurm scripts.verify_processed --dataset tuev
```

This is not a formality. Two preprocessed corpora were lost to disk pressure on
the source cluster (`processed_biot/tuab`, `processed_labram/tuab`) and have to
be rebuilt rather than copied, which makes them the first place a version drift
would show up.

## Splits are frozen, not recomputed

For TUH: the official train/eval split is used, and the official *train* is cut
80/20 by **sorted subject ID** into train/val — never randomly, so the split is
identical on any machine without carrying a file around. `eval` is the test set
and is never touched during selection. Same rule for the others, per corpus, in
`docs/PROTOCOLS.md`.

## Order of operations on a new cluster

```bash
# 1. apply for TUH access, download everything into $PACLOCK_DATA
sbatch slurm/download_isruc.slurm

# 2. frozen protocol first -- groups A, C, D and CBraMod all read it
for ds in tuab tuev tusz chbmit sleepedf isruc physionet_mi faced bci_iv_2a; do
    sbatch slurm/preprocess.slurm frozen $ds
done

# 3. the per-model protocols, only if group B is being run
sbatch slurm/preprocess.slurm biot   tuab      # BIOT and TFM-Tokenizer
sbatch slurm/preprocess.slurm labram tuab      # LaBraM: 23 unipolar channels

# 4. the PAC-protocol arm, only for the group-D sensitivity analysis
sbatch slurm/preprocess.slurm pac tuev

# 5. verify before running anything that will be reported
sbatch slurm/run.slurm scripts.verify_processed --dataset tuev
```

Step 2 is enough for all PACLock work. Steps 3 and 4 exist because hard rule 2
requires each model to run its own repo's preprocessing.

## Added after the original nine

| corpus | version | access | why it was added | status |
|---|---|---|---|---|
| TUEG | v2.0.1 | same TUH account | the pretraining pool | in use — see below |
| TUAR | v3.0.1 | same TUH account | test whether the PAC tokenizer's TUEV advantage generalises to another event-morphology task | **answered: no.** pac 0.5780 vs raw 0.6289 |
| TUSL | v2.0.1 | same TUH account | the sharpest available same-type test (slowing vs seizure differ in waveform structure, not band power) | **abandoned before spending training compute** — 300 events total, 100 per class |

Built by `preprocessing/tueg.py` (pretraining slice) and
`preprocessing/tuh_events.py` (TUAR/TUSL). The latter differs from `tuev.py` in
three ways that are easy to get wrong: annotations are per-channel CSV and one
event spans many channel rows, so rows are collapsed to unique
`(start, stop, label)` intervals; event durations vary, so the window is centred
on the event midpoint rather than anchored before its start; compound TUAR labels
(`eyem_musc`, `musc_elec`, …) are dropped rather than folded into a parent class,
because folding them would make the class definition depend on annotation style.

### TUAR is 3 classes, not 6

`configs/datasets/tuar.yaml` records each drop and its reason in `_class_note`:

* **`bckg` yields 0 events** — this is not a parse bug. All 41 annotation files
  carrying `bckg` intervals ship without a matching EDF (352 csv vs 310 edf).
* **`chew` was dropped** — present in only 25 of 212 subjects, and 0 windows land
  in the validation split.
* `shiv` and the stray seizure labels (`fnsz`, `gnsz`) are dropped too; the
  latter belong to TUSZ's task.

Final map: `eyem: 0, musc: 1, elec: 2`, 5 s windows.

### The TUEG pretraining slice excludes TUSZ sessions

`DOCS/sessions_tueg_common_with_tusz.list` lists the sessions TUEG shares with
TUSZ; the sampler skips them, so a model pretrained on this slice has not seen
TUSZ's evaluation subjects. Sampling is subject-diverse rather than
sequential. The writer streams per-worker temp `.npy` files into a preallocated
`np.lib.format.open_memmap` — an earlier version accumulated 92 GB in the parent
process and was OOM-killed.

The separate `pretrain-excl_szdet` checkpoint goes further and removes CHB-MIT
and TUSZ from the pool entirely; it exists to answer the "you pretrained on your
downstream data" objection (`docs/STATUS.md` §5).

---

# 附录 B:baseline 配方保真审计

*(原 `docs/PROTOCOLS.md`)*

Hard rule 2 says every model runs the recipe from its own repository. This
records what was checked against the vendored source, what was found off it, and
what was deliberately left alone. Each row cites the file that settles it, so a
reader can disagree with a decision without having to re-derive it.

Reproduce the machine-checkable part with:

    python -m scripts.fix_recipe_fidelity     # dry run, exits clean when aligned

---

## Corrected

| Model | Setting | Was | Now | Source |
|---|---|---|---|---|
| CBraMod | `grad_clip` | `null` | `1.0` | `finetune_main.py` `--clip_value default=1`; `finetune_trainer.py` calls `clip_grad_norm_` every step |
| CBraMod | `label_smoothing` (TUEV) | `0.0` | `0.1` | `finetune_main.py` `--label_smoothing default=0.1` |
| CBraMod | `patience` | `10` | `0` (off) | `finetune_trainer.py` runs all epochs and keeps the best state; it has no early-stopping branch |
| CBraMod | `select_metric` (binary corpora) | primary metric | `auroc` | `finetune_trainer.py` selects on `roc_auc > roc_auc_best` |
| CBraMod | `eval_every_steps` | `100` | `0` (per epoch) | its trainer validates once per epoch |
| LaBraM | `label_smoothing` (6 multi-class corpora) | absent / `0.0` | `0.1` | `run_class_finetuning.py` `--smoothing default=0.1`, used whenever `nb_classes != 1` |
| PACLock | `lr` | `1e-3` | `1e-4` | `Joni-z/PACLock` `configs/pacint_tuev_measured.yaml` |
| PACLock | `dropout` | `0.1` | `0.2` | same |
| PACLock | `band_pe` / `spatial_pe` | absent (silently `hz` / learned index) | `index` / `xyz` | AGENT.md:2974 names these the architecture of record |
| PACLock | `epochs` (3 small corpora) | `20` | step-budget floor | see below |

### Measured effect on group C

| Cell | Off-recipe | On-recipe | Δ |
|---|---|---|---|
| TUEV / CBraMod-scratch | 0.5367 | 0.5638 | +0.027 |
| PhysioNet-MI / CBraMod-scratch | 0.5212 | (re-running) | |
| TUAB / CBraMod-scratch | 0.8734 | (re-running) | |
| TUSZ / CBraMod-scratch | 0.4903 | (re-running) | |

Every one of these strengthens a baseline. That is the point: a win over a
mis-configured opponent is not a win.

Intermediate figures measured while the corrections were being landed one at a
time are **not** reportable and are recorded here only so they are not mistaken
for results. TUEV/CBraMod-scratch read 0.6098 with clipping and smoothing but
early stopping still on, and 0.6233 under a `patience: 51` that was a units bug
on my part -- patience counts evaluations, so with `eval_every_steps: 100` that
was one epoch of patience, and the run stopped at epoch 2 having sampled a dense
curve and picked a lucky checkpoint. The recipe CBraMod actually publishes --
fifty full epochs, validated once per epoch, best kept by AUROC -- gives 0.5638.
Training longer scores lower here, and the published recipe is still what gets
reported.

---

## Checked and left alone

**BIOT, EEGPT — no label smoothing.** Both construct a plain
`CrossEntropyLoss()` (`run_multiclass_supervised.py`, `downstream/finetune_*.py`),
so `0.0` is faithful and raising it would be *our* invention.

**LaBraM `layer_decay: 0.65`.** The argparse default is `0.9`, which looks like a
discrepancy, but `0.65` is what LaBraM's own downstream script passes. A script
the authors ship beats a default they never use.

**LaBraM label smoothing on TUAB / TUSZ / CHB-MIT.** Absent is correct: those are
binary, and `run_class_finetuning.py` takes the `nb_classes == 1 ->
BCEWithLogitsLoss` branch, which never consults `smoothing`.

**CHB-MIT focal loss (α=0.25, γ=2).** Protocol-level, not per-model: PROTOCOLS.md
appendix A fixes the loss per corpus and every model gets the same one. Probed
α=0.75 anyway on the one cell that fails there; it did not help.

**FACED normalisation (`div100`).** Matches CBraMod's own pipeline exactly —
`preprocessing_faced.py` resamples to 200 Hz and `faced_dataset.py` returns
`data/100`. FACED does arrive ~30x hotter than the other corpora (training-signal
std 27.88 against 0.10-1.02 elsewhere), but that is upstream's scale, not our
error, so the data is left as published.

---

## Two harness bugs found while doing this

**`patience` counts evaluations, not epochs.** With `eval_every_steps: 100`,
CHB-MIT runs ~49 evaluations per epoch (4941 steps), so a patience of 51 is one
epoch of patience rather than fifty. Setting `patience: 0` is what actually
disables early stopping — `train.py` reads `if patience and since_best >=
patience`.

**Checkpoint selection and reporting used the same metric.** They answer
different questions: the protocol fixes what is *reported*, each repo fixes what
it *selects on*. They diverge most where variance does. CHB-MIT's validation
split holds 150 positives in 21,184 windows, so PR-AUC swings between 0.007 and
0.5 between consecutive evaluations while AUROC barely moves; selecting and
early-stopping on that noise ended CBraMod's from-scratch runs after one epoch,
at chance. `select_metric` now separates them and defaults to the primary metric,
so every config that does not set it is bit-identical to before.

---

## PACLock's step budget

The reference recipe is "batch 32, 20 epochs", but the quantity that was
validated is the number of optimiser steps, not the number of sweeps. On TUEV at
batch 32 that is `68445/32*20 = 42,760` steps. Copying the epoch count to a
smaller corpus copies a fraction of the training: BCI-IV-2a has 2,160 windows, so
20 epochs is 1,340 steps.

The consequence was visible in the training loss, not just the test score. On
FACED it went `2.2207 -> 2.1858` across a whole run against a `ln(9) = 2.1972`
floor — the model never fit the training set. On BCI-IV-2a under the corrected
budget the loss sits on a plateau near `ln(4) = 1.386` for about twenty epochs and
then breaks through, reaching 0.66 by epoch 35. Twenty epochs ends the run inside
the plateau.

`gen_configs_d.py` therefore derives epochs from a step floor. A floor, not a
target: matching the budget exactly would cut TUAB, TUSZ and CHB-MIT from 20
epochs to 4, and this must not shorten training anywhere. Only the three small
corpora move.

| Corpus | Epochs | Steps |
|---|---|---|
| PhysioNet-MI | 20 -> 218 | 42,728 |
| BCI-IV-2a | 20 -> 638 | 42,746 |
| FACED | 20 -> 120 | 25,200 (24h partition cap) |
| all others | unchanged at 20 | already above the floor |

This is the fair reading rather than a generous one. Early stopping still decides
where training actually ends, and the large corpora early-stop long before their
twentieth epoch, so raising a ceiling they never reach changes nothing for them.

### What it did not fix

A scale hypothesis was tested and rejected: FACED arrives ~30x hotter than the
other corpora, so an input standardisation was tried on the theory that it
distorted the PAC product term. It changes the input (PhysioNet-MI's epoch-0
loss moves from 1.4074 to 1.4073) but not the loss trajectory, because the
encoder's LayerNorms already absorb the scale. The code was removed rather than
left behind as an unused option.

---

# 附录 C:被删除的预处理,以及如何重建

`/work1` 上的项目配额是 **1.9 T**,不是整个文件系统(底层是 382 T)。2026-08-18
配额到 96%(仅剩 80 G)时做过一次清理,降到 85%(293 G 可用)。

删掉预处理**不会**影响已有结果:`runs/*/seed*/result.json` 里存着完整的配置、
val 曲线和裁定,`manifest.json` 里存着每个源文件的 SHA256。删掉的只是可以从
`$PACLOCK_DATA`(511 G,仍在)重新生成的中间产物。

**这一条存在的理由**:此前 `processed_biot/tuab` 和 `processed_labram/tuab`
被以同样方式删过,但没有留记录,后来花了不少时间才弄清它们为什么不见了。

## 2026-08-18 删除清单

| 目录 | 大小 | 删除依据 |
|---|---|---|
| `processed_labram/` | 134 G | 6 个 run(LaBraM pretrained/scratch × TUAB/TUEV/TUSZ)全部 3 seed、verdict 全 OK。LaBraM 的位置编码要求 23 个单极 `-REF` 通道,CHB-MIT 只有双极导联无法还原,所以**不可能再有新格子** |
| `processed_biot/` | 70 G | 9 个 run(BIOT pretrained/scratch × TUAB/TUEV/TUSZ,加 `biot_hop50`/`biot_tok100`,加 `tuev-paclock_pilot_unfiltered`)全部 3 seed、verdict 全 OK。BIOT 在 CHB-MIT 上用 `processed/`,不依赖此目录 |
| `processed/faced_winnorm` | 2.5 G | 建了之后**从未被任何 run 引用** |
| `processed/faced_subjnorm` | 2.5 G | 每受试者归一化,实测无效(见 FINDINGS.md) |
| `processed/faced_clean` | 2.5 G | FACED 伪迹清洗,实测无效 |
| `processed/physionet_mi_winnorm` | 1.9 G | 每窗口归一化,实测无效 |
| `processed/bci_iv_2a_winnorm` | 349 M | 同上 |

删除前核对过三件事,顺序不能省:

1. 每个依赖该目录的 run 是否已达 3 seed 且 verdict 为 ok(`scripts/xlsx_gap.py`
   和逐 run 的 `result.json`);
2. 该 baseline 是否还可能有新语料(montage 限制);
3. **当时队列里每个 job 的 `data_root`** —— 删掉一个正在被读的目录会让跑了几小时
   的任务在中途崩掉。

## 重建

原始语料在 `$PACLOCK_DATA`(511 G),四套协议都可重跑:

```bash
sbatch slurm/preprocess.slurm biot   tusz      # BIOT 与 TFM-Tokenizer
sbatch slurm/preprocess.slurm labram tuab      # LaBraM:23 单极通道
sbatch slurm/preprocess.slurm frozen  <ds>     # 冻结协议 -> processed/
sbatch slurm/preprocess.slurm pac     <ds>     # PAC 协议  -> processed_pac/
```

派生副本由脚本重建,各自一条命令:`scripts/winnorm.py`、`scripts/subjnorm.py`、
`scripts/clean_faced.py`。

重建后**必须**核对 manifest 的 SHA256 与原结果一致,否则新数组和 `runs/` 里已有
的数字不可比:

```bash
sbatch slurm/run.slurm scripts.verify_processed --dataset tusz
```

## 仍然保留

`processed_pac/`(180 G)没有删。它背后有三个未达 3 seed 的格子,其中
`chbmit-paclock_pac`(n=1)是 PAC 协议敏感性分析的真实空缺。若再次需要空间,
可以只删 `chbmit` 以外的七个语料(约 137 G),把补齐那一格的能力留着。

`pretrain_runs/` 与 `pretrain_runs_60k/`(共 125 M)必须保留 —— 正在被
`p200` 三臂实验和 raw 预训练微调读取,且已在 `.gitignore` 中(见 STATUS.md)。


---

# 附录 D:同类 EEG FM 如何处理预训练/下游重叠(2026-08-21 调研)

发现自家 TUEG 切片有被试级泄漏后(`STATUS.md` §5.5),先查了领域标准,
因为"要不要修"取决于对手怎么做。**结论:没有人做被试级排除,也没有人量化过。**

| 模型 | 预训练数据 | 排除做法 | 证据 |
|---|---|---|---|
| **CBraMod** | 全量 TUEG(27,062 h / 14,987 被试) | **零排除** | 它自己发布的 `preprocessing_tueg_for_pretraining.py`(见 `vendor/cbramod/`):`iter_files()` 遍历 TUEG 根目录 → sorted → shuffle → 全量处理,无任何 session/subject 过滤;随后在 TUAB/TUEV 上评测 |
| **LaBraM** | 2,500 h+,含 TUEG | 排除的是**下游数据集**(以数据集为单位),但 TUEG 留在池中,而 TUAB/TUEV 的病人就在 TUEG 里 | 论文表述"the four downstream datasets excluded from the pre-training datasets" |
| **BIOT** | 含 TUH | 未记录 | — |
| **REVE** | 含 TUH(14,987 被试,占语料 44%) | 移除下游评测数据,但自陈分布相似性仍在 | — |

2026 年的批评论文 *What EEG Foundation Models Encode: Dataset Identity and a
Negative-Control Suite for Clinical Benchmarks*(arXiv 2607.24519)明说
"**all models are in-domain on this task**"(指 TUAB),并警告 TUAB 的结果
不应与 CHB-MIT 等 OOD 结果直接比较 —— 但它**也没有量化被试重叠比例**。

## 为什么"他们不做所以我们也不做"在本项目不成立

**他们不排是因为排了要掉数据量,我们排了一点不掉。**CBraMod 用的是全量
TUEG;我们只抽 2,000 h ≈ TUEG 的 7%。排掉 3,182 个下游被试后仍有 11,885 个
合格被试,而我们只需要 5,245 个文件 —— 干跑实测:**同样 2,000 h,文件数
一个不少,泄漏检查 0**。

而且 TUAB 自己的 train split 本来就直接在池中(占采样 15.5%),**合法的
域内信号一点没丢**;排掉的恰好只是测试病人的录音。

风险是不对称的:我们已经量出 34.8% 了。"没查过"是一种辩护,"查过、
知道了、仍然发"不是;而这个检查是五行脚本,审稿人自己能跑。

**因此建议发清洁版,并把这次测量写进论文方法节** —— 在上述批评论文已经
点名该问题的背景下,"我们的预训练切片与全部下游测试集被试不相交,而标准
TUEG 切片在 TUAB 上的重叠实测为 34.8%"是别人给不出的一句话。

**实际决定(2026-08-21)**:rung-1 采用未排除的原切片,与 CBraMod/LaBraM
口径一致 —— 理由是与对手公开数字的可比性(见 `STATUS.md` §5.5 决定)。
本附录的调研与建议保留在此,因为清洁切片已经建好(`tueg_slice_clean`),
"泄漏值多少分"这个测量随时可做。

脏切片(`processed/tueg_slice`)**保留未删**:若要量化"泄漏能把 TUAB 抬高
多少",多跑一个 60k 即可,那是论文的一节而不是成本。


---

# 附录 E:12 语料名单新增的五个语料(2026-08-19 → 22)

| 语料 | 来源 | 通道 / 窗口 | 划分 | 主指标 | 规模(train/val/test 窗口) |
|---|---|---|---|---|---|
| TUEP v2.0.1 | TUH,`01_tcp_ar` | 16 bipolar / 10 s | 被试不相交排序,按标签分层 | AUROC | 136,525 / 23,808 / 24,406 |
| TUAR | TUH,3 类伪迹 | 16 bipolar / 5 s | 同上 | κ | 20,650 / 5,046 / 1,936 |
| ADFD | OpenNeuro ds004504,Medformer 预处理版 | 19 ref / 10 s | 被试不相交,按类分层 | balanced_acc | 4,895 / 942 / 1,101 |
| APAVA | Medformer 预处理版 | 16 / 10 s | 同上 | AUROC | 484 / 58 / **46** |
| Mumtaz2016 | figshare 4244171 | 19 ref / 5 s | 被试不相交,按诊断分层 | AUROC | 4,526 / — / 1,303 |
| EEGMat | PhysioNet eegmat 1.0.0 | 19 ref / 5 s | 被试不相交(每人两类) | AUROC | 1,199 / 240 / 268 |

要点:

* **ADFD 的数组是伏特**,配置里用 `unit_scale: 1e6` 显式换算,不埋在代码里。
* **Mumtaz 只用 EC/EO 静息**,丢弃 TASK —— CBraMod 的先例;但 **划分不抄**:
  CBraMod 按文件排序切,同一被试的 EC/EO 会跨 train/test。
* **EEGMat 的标签就是文件后缀**(`_1` 静息 / `_2` 心算),每个被试两类都有,
  故无需分层。供方已做 0.5–45 Hz 滤波,所以带通取 [0.3, 45] 且不加陷波。
  训练集类别 899:300,是因为静息录 180 秒而心算 60 秒 —— 协议本身,不是错误。
* **APAVA 不适合作为评测语料**:全语料 22 个被试、588 个窗口,测试集 3 人 /
  46 窗口 / 类别 2:44。建议除名(见 `FINDINGS.md` 5.4)。

## baseline 适配:三个跨语料路径必须显式打开

新语料跑 15 个 baseline 时发现(`scripts/check_new_baselines.py`,每配置一个
子进程 —— 单进程不行,EEGPT 的 adapter 把 `vendor/eegpt` 放进 sys.path 后其
顶层 `models` 包会遮蔽 vendor/tfm 的,产生 6 个假的 ModuleNotFoundError):

1. **EEGPT** 需要 per-corpus 通道表。`TEN20` 恰好就是 19 电极 10-20 集,
   所以是 adfd/mumtaz/eegmat 的**原生**蒙太奇;TUH 语料照 tuab/tuev 的先例复用。
2. **LaBraM** 的电极索引路径要 `labram_native` 的 23 通道蒙太奇,只有 TUAB/TUEV
   有;其余语料必须 `montage_mode: positional` + `target_len`。
3. **BIOT prest16** 按 16 通道构建,19 通道语料需要 `target_len` 才走跨语料路径。


---

## 发表值锚点政策(2026-08-27 冻结)

* 锚点仅用于校准(reproduction gate:发表值须落于我们 mean±2σ),
  **永不与我们的数并排进论文表格**。
* 逐点可比:仅 TUAB、TUEV(官方切分)。CHB-MIT/IIIC/TUSZ 等自定义协议
  语料的锚点一律标 not point-comparable(collect_results.py 的
  NONCOMPARABLE 表)。
* IIIC 最终协议:SPaRCNet 官方数组 → 按 key 去重(−23,355)→ 剔专家
  平票(−5,900)→ 105,195 窗;balanced_patient_split 70/15/15,
  三 split 类分布小数点后三位一致(preprocessing/iiic.py)。
* Siena 微调配方:batch 128 / lr 1e-4 / epochs 60
  (configs/_diag/siena_paclock_duplex_b128.yaml,含完整归因注释);
  batch 32 在 326 正例下 74% 的步无正例梯度,禁用。
