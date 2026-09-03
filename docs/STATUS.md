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
- 09-03 patch-200 全量加载对照(TUEV):scratch@200 0.6094 vs ptF@200 0.5306(−0.079);害处不是 tokenizer 重初始化工件,5.6 判决成立。详见 FINDINGS 5.8。TUSZ 对:scratch@200 0.6140 vs ptF@200 0.6296(+0.016,方向与 5.6 一致)。CHB-MIT 对:0.5486 vs 0.5448(−0.004,平)。三对齐,判决不变;b2 镜像全撤。

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
- 跨集群复现(09-03):`tuev_paclock_duplex` seed 0 在 H200 上 test κ 0.6969(amd MI210 同配置 0.7094;best val 0.572 vs 0.609),1.95 h。差 0.0125,在单 seed 噪声内(TUEV 三 seed 的 CBraMod 基线 std ≈ 0.02);torch 结果可入表,但同一格的三个 seed 应在同一集群跑。

## 7. 移植波(TP_*,CBraMod 编码器 + 我们的 duplex 前端,从零训练,3 seed;09-03 中期,8/11 落地)

| 语料 | 移植(3 seed) | CBraMod 从零(3 seed) | Δ | CBraMod 预训练(3 seed) |
|---|---|---|---|---|
| TUEV (κ) | 0.6115±0.004 | 0.5638±0.019 | **+0.048** | 0.6449±0.018 |
| ADFD (BAcc) | 0.4628±0.008 | 0.4810±0.015 | −0.018 | 0.4411±0.013 |
| CAUEEG (BAcc) | 0.4270±0.068 | 0.4266±0.067 | 0(双方各有 seed 塌到 0.333) | 0.5145±0.025 |
| Sleep-EDF (κ) | 0.6084±0.001 | 0.6545±0.003 | **−0.046** | 0.6715±0.004 |
| Siena (AUC-PR) | 0.089±0.065 | 0.199±0.050 | −0.11(两者都在正样本饥饿地板上) | 0.4243±0.052 |
| IIIC (κ) | 0.3100±0.009 | 0.3935±0.005 | **−0.083** | 0.3133±0.016 |
| ISRUC (κ) | 0.5099±0.023 | 0.6976±0.005 | **−0.188** | 0.7540±0.006 |
| TUEP (AUROC) | 0.6330±0.030 | 0.6422±0.028 | −0.009(平) | 0.6647±0.017 |

未落地:TUSZ、CHB-MIT、TUAB(仍在跑)。旧 Line-B 单 seed 曾见 CHB-MIT 0.508 vs 0.317。
中期读法:前端搬到 CBraMod 上,事件/发作类语料受益(TUEV),睡眠分期受损(Sleep-EDF −0.046)——
移植把 CBraMod 自带的 FFT 谱分支换掉、且把 8 个频带均值池化掉,谱功率信息变粗,睡眠最吃这一块。
对会议第三点的含义:"即插即用"目前只对 paroxysmal 语料成立;要么在移植里保留频带轴(不池化),
要么把结论收窄为发作/事件检测。等 6 个语料落地后定。

- 09-03 TP_tusz(401197)在 14.7 h 时主动撤销:35/50 epoch,0.42 h/epoch,剩余 15 epoch 需 6.3 h 而
  wall 只剩 5.3 h;train.py 不存中间 checkpoint、结果只在收尾时写,被 SLURM 杀掉等于全丢。
  TP_chbmit(7% 超限,赌一把)与 TP_tuab(勉强够)保留。val 轨迹(TUSZ pr_auc 0.23 vs CBraMod 从零
  test 0.48)已足够说明直接移植在 TUSZ 上输;需要正式数字时在 torch 单 seed 补(H200 约 10 h)。
  教训:50 epoch 无 patience 的长跑必须带 max_hours(ptS 配置已带 22.5 h)。

### 死活门(2026-09-03 投,amd 402075,4 卡打包,seed 0)

直接移植 8/11 落地后只赢 TUEV,判定这条线需要先过门再谈对照。修法:适配器加
`band_mode: channels`——前端的 16 行(8 fused + 8 interaction)不再均值池化,而是
每个(电极, 行)当作 CBraMod 的一个通道送进 criss-cross 编码器,编码器之后再对行
取均值,分类头形状与参数量不变(17.90M vs 17.90M)。同时跑耦合关闭对照
(`tokenizer_mode: raw`,同前端,8 行)。配置 `configs/experiments/{tuev,tusz}_cbramod_{crofremo,rawfe}_bands.yaml`。

预注册判据:TUEV 移植 ≥ 0.61(且明显高于 CBraMod 从零 0.564)**且** TUSZ 移植 ≥ CBraMod
从零 0.48 → 线活,再补耦合开/关 3 seed;否则关线,第三点改为"新颖性收回到自己的模型,
移植作为边界讨论",不再投任何 tokenizer 对照。

## 8. 预训练行定稿配方 ptS(2026-09-03 投,12 语料单 seed,三集群分跑)

不再搜索。checkpoint = duplex_v2 60k(短版;150k 的 v2_full 在所有试过的语料上都更差)。
微调 = LaBraM/CBraMod 的预训练标准配方:base lr 5e-4 + layer_decay 0.65(按深度,
frontend/band_pe=0,encoder.blocks.i=i+1,head/spatial_pe=顶),warmup 2 epoch,前 2 epoch
只训未从 checkpoint 加载的张量(patch-50 tokenizer 卷积、head、spatial_pe),之后全解冻。
不逐语料调参;batch/epochs 与从零配方相同。代码:`freeze_loaded_epochs`、
`paclock_layer_decay_param_groups`、`model._loaded_keys`、`$PACLOCK_CKPT`(commit fe43a46)。
checkpoint 走仓库 `ckpt` 分支跨集群(6.8 MB)。

| 集群 | 语料 | 任务 |
|---|---|---|
| amd | adfd caueeg tuep tuar | PTS_small 403027(4 卡打包) |
| amd | iiic siena tuab chbmit | PTS_big 403028(4 卡打包) |
| b2 | isruc sleepedf | ptS_isruc 45126830 / ptS_sleepedf 45126831(h100-80) |
| torch | tuev tusz | 16833442 / 16833443(h100_tandon) |

判读:与"最好从零"逐格比;赢/平/输如实入表,单 seed 定行后补 3 seed。
