# 进度(2026-08-27)

上一版写于 08-21。此后:定位收窄为阵发性临床评测并完成全面文献调研;定名
(标题 + CroFreMo);baseline × 9 语料矩阵基本铺完并两次修正(IIIC 平衡切分、
Siena 损失);**找到预训练不迁移的根因(日程只有 1.58 epoch)并在 b2 上重跑
修正日程的预训练**;论文骨架在 Overleaf 仓库里立起,Related Work 已写完。
本版全文重写;08-21 版的内容并入 `FINDINGS.md` / `CHANGELOG.md`。

---

## 文档导航

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| `STATUS.md` | 本文件 —— 现状、在跑的实验、决策规则、预算 | 先读这个 |
| `PROTOCOLS.md` | 冻结的预处理与评测协议、发表值锚点政策、baseline 配方审计 | 改预处理或复现协议之前 |
| `FINDINGS.md` | 架构搜索结论、预训练根因链、Siena/IIIC 诊断 | 想改模型之前 —— 大部分想法已经试过了 |
| `PRETRAIN.md` | 预训练方案;08-26 修正日程(必读) | 要再跑预训练时 |
| `PAPER.md` | 论文结构、ICLR 2027 硬规则、图表清单、防御策略 | 写论文时 |
| `DIRECTION.md` | 定位、定名、贡献点拆分(FAME 之后) | 讨论"我们主张什么"时 |
| `CHANGELOG.md` | 按时间的变更日志,含被否决的方案和原因 | 想知道"这个为什么是现在这样" |

---

## 1. 活动目标

1. **ICLR 2027**:摘要 9-18,正文 9-25(AoE)。标题
   *Cross-Frequency Modulation as Token Content and Pretraining Objective
   for EEG Foundation Models*,模型名 **CroFreMo**。作者:Zhizhe Zhang
   (NYU Shanghai)、Yifan Wang(Stony Brook)。
   ⚠️ 9-18 后不能加作者;须有一名作者满足 reciprocal-review 资格并在
   OpenReview 注册评审 ≥3 篇 —— Yifan 需在 9-18 前完成注册。
2. baseline × 9 语料 × 3 seed(goal 仍挂着):297 格已填,
   8 格按规则留空(其中 7 格是 baseline 自身训练失败,rule 3 扣下)。
3. `PT_v2full` 判决后,补 5 个缺失语料的 v2 微调(IIIC/TUEP/ADFD/CAUEEG/Siena)。

## 2. 九语料战绩(vs 全表最强 baseline,08-27)

| 语料 | 我们 | 最强 baseline | Δ |
|---|---|---|---|
| TUSZ (AUC-PR) | v2 0.7143 | FFCL 0.5449±0.024 | **+0.169** |
| CHB-MIT (AUC-PR) | v2 0.7134 | TFM 0.6269±0.021 | **+0.087** |
| IIIC (Kappa) | scratch 0.4868 | REVE 0.4363±0.002 | **+0.051** |
| ADFD (BAcc) | scratch 0.5617 | BIOT-scr 0.5254±0.017 | +0.036 |
| TUEP (AUROC) | scratch 0.8052 | EEGPT-scr 0.7859±0.018 | +0.019 |
| TUEV (Kappa) | scratch 0.7094 | Uni-NTFM 0.7030(自报) | +0.006(方差内) |
| TUAB (BAcc) | scratch 0.8157 | 复现最强 ST-T 0.8198 | −0.004(平;REVE 自报 0.8315) |
| CAUEEG (BAcc) | scratch 0.5254±0.012 | BIOT-scr 0.5609±0.009 | −0.036 |
| Siena (AUC-PR) | 0.1098(batch bug) | REVE 0.5181±0.096 | 修复中(SIENA_b128) |

6 胜 2 负 1 修复中,1.6M 参数对 25–69M。**以上全部不含新预训练。**

## 3. 在跑

| 作业 | 集群 | 内容 | 状态 |
|---|---|---|---|
| PT_v2full 44508503 | b2 h100 | 修正日程预训练:150k 步 × 均值 batch 197 = 15.4 epoch,band_norm_pac,+TUEG | ~50%,ETA 08-27 晚 |
| SIENA_b128 386667 | amd | Siena 修 batch(32→128, epochs 60) | 排队 |

## 4. 预训练判决(已定稿,见 FINDINGS 5.6)

结局:标题走 token-content 版(已改),预训练为分析章节。机制分语料(TUEV
优化锁死 / TUSZ 表示退化),耦合项作用 corpus-dependent,低标注例外成立。
**不再开预训练实验。** 下文为当初的预注册规则,存档:
- 09-03 patch-200 全量加载对照(TUEV):scratch@200 0.6094 vs ptF@200 0.5306(−0.079);害处不是 tokenizer 重初始化工件,5.6 判决成立。详见 FINDINGS 5.8。TUSZ 对:scratch@200 0.6140 vs ptF@200 0.6296(+0.016,方向与 5.6 一致)。CHB-MIT 对在跑。

## 4a. (存档)PT_v2full 出来后的决策规则

| 结果 | 动作 |
|---|---|
| 预训练普遍胜 scratch | 按计划写 |
| 只在阵发类胜(TUSZ/CHB/Siena) | 最好的情况:论点被证实 |
| 仍只有 TUSZ | 用最后一发:γ 插值目标,一次重训 |
| γ 后仍不行 | 停手;tokenizer 贡献 + 目标函数诚实负结果,标题砍掉 "and Pretraining Objective" |

**任何情况下不回头动架构** —— scratch 战绩已验证它,预算和时间也不允许。

## 5. 预算与集群

* **b2**:计费滞后,实际可用 ≈ 405 SU,PT_v2full 吃 ~250。Zhizhe 确认
  **SU 可以再申请** —— 但审批有周期,应尽快提。预训练全部走 b2
  (TUEG 85G 只在 b2;amd 存储紧、卡慢,不搬)。
* **amd**:节点小时 1271/2250 已用,剩 ~978。下游微调主力。
* b2 同账号有另一个项目(object-srcatt-* / color-b01-*)共享 fair-share
  与 SU 池,排队慢时先查它。

## 6. 悬而未决

* 7 个 rule-3 扣下的 baseline 格:ADFD/EEGNet 与 CAUEEG/LaBraM 是
  3-seed 全灭(疑配置),其余 5 个是个别 seed 失败 —— 待补跑。
* IIIC 按 BIOT 原协议的"保险行"(我们模型一次训练)—— Zhizhe 未拍板。
* 代码层 paclock_* → CroFreMo 重命名,等队列排空。
* 用户侧:AWS key 与 HF token 建议轮换(注入消息事件后),未确认完成。

## 6. 第三个集群:NYU Torch(2026-09-03 起)

- 仓库 `/scratch/zz5070/PACLock`(与 GitHub main 同步);环境 conda `py312`(torch 2.8+cu128,numpy 2.0.2 / scipy 1.13.1 与 amd 一致;scipy≥1.17 会让 mne 1.8 导入失败)。
- 数据:`PACLOCK_DATA=/scratch/zz5070/data/raw`,`PACLOCK_PROC=/scratch/zz5070/data`;已就绪 `processed/tuev`、`processed/tusz`(由 torch 上的原始 edf 重新预处理,split 大小与各类计数与 amd 完全一致;样本顺序不同,`imap_unordered` 所致)。其余语料需从 amd 拉或重新下载。
- 投递:`sbatch -A torch_pr_63_tandon_advanced -p h100_tandon|a100_tandon|h200_tandon slurm/torch_run.slurm <cfg> [seed]`,或 `-A torch_pr_63_general -p h200_public|l40s_public`;`--array=0-2` 跑三 seed。CPU:`-A torch_pr_63_general -p cpu_short`(≤4h,内存上限 120G)。
- 队列很深(排 1–2 小时起),适合不赶时间的批量任务;H200 上 TUEV 一个 epoch 约 8 分钟(amd MI210 约 12 分钟)。
- 进行中:`tuev_paclock_duplex` seed 0 跨集群复现(对照 amd κ 0.7094)。
