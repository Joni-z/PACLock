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

## 1. 活动目标(2026-09-05)

`/goal 保证开启的实验正确并完成实验目的 减少时间和资源的浪费 并修改论文`。
ICLR 2027:摘要 9-18,全文 9-25。论文稿(Intro / Related Work / Method v3 / Setup / results 备注)在 MacBook 本地,
编译通过,**未推 Overleaf**(Zhizhe 看过再推)。

## 2. 论文主张(当前证据下的诚实版本;第 9 波落地 62/72 后)

1. **贡献一(经验+解释)**:1.6M 的"可学习滤波器组分频 token + 三轴注意力"模型,在 12 个临床语料上整体不输 25–69M 的
   FM;在稀有正样本的发作检测上大幅领先。架构对照证明领先来自分频与三轴两个归纳偏置,不是参数量(§6)。
2. **贡献二(机制)**:跨频耦合作为 token 内容,在痫样事件分类(TUEV)上决定性(同编码器 +0.17,三 seed);
   移植进 CBraMod 后 TUEV 赢、TUSZ 平,两处增益都可归因于耦合内容(§7)。在其他 10 个语料上耦合对我们自己的编码器
   为零(§5)。候选解释:三轴编码器有频率轴,能从分频 token 自学跨频关系,显式耦合 token 只在关系最难学的事件形态上
   有增量;无频率轴的编码器处处受益。
3. **贡献三(分析)**:预训练——目标函数消融、lr 匹配探针、标签比例、patch-200 全量对照、统一配方的预训练行(3 赢 9 输)。

已钉死:预训练不是贡献;TUEV 上耦合决定性;耦合在 TUAB/睡眠/TUEP/ADFD/CAUEEG/TUAR/Siena 为零(全三 seed)。
第 9 波已收官(FINDINGS 6.5):耦合在 TUSZ −0.03、CHB-MIT +0.03(均在 1 std 内,记为零);TUEV 逐类别增益集中在 GPED/PLED(6.4);
从零 duplex 除 TUAB 外全部三 seed(TUAB seed 1/2 待投)。

## 3. 主表战绩(我们 vs 最强已复现 baseline;三 seed 标 (3),单 seed 标 (1))

| 语料(指标) | 我们(从零 duplex) | 最强 baseline | Δ | 判 |
|---|---|---|---|---|
| TUSZ (AUC-PR) | 0.633 (1);raw 0.671±0.026 (3);预训练 0.714 | FFCL 0.545±0.024 | +0.09~+0.17 | 赢 |
| CHB-MIT (AUC-PR) | 0.713 (1);raw 0.667±0.047 (3) | TFM-pre 0.627±0.021 | +0.04~+0.09 | 赢 |
| TUEV (κ) | 0.709 (1);rot2 0.733±0.016 (3) | REVE-pre 0.685±0.032 | +0.02~+0.05 | 赢 |
| IIIC (κ) | 0.479±0.008 (3) | REVE-pre 0.436±0.002 | +0.04 | 赢 |
| TUEP (AUROC) | 0.810±0.004 (3) | EEGPT-scr 0.786±0.018 | +0.02 | 赢 |
| TUAB (BAcc) | 0.816 (1) | ST-T 0.820±0.004 | −0.004 | 平 |
| ADFD (BAcc) | 0.505±0.050 (3) | BIOT-scr 0.525±0.017 | −0.02(方差内) | 平 |
| CAUEEG (BAcc) | 0.525±0.012 (3);调参 0.558 (1) | BIOT-scr 0.561±0.009 | −0.04 / −0.003 | 平/小负 |
| Siena (AUC-PR) | 0.110 (1);ptS 0.456±0.034 (3) | REVE-pre 0.518±0.096 | −0.06(REVE 方差内) | 小负 |
| Sleep-EDF (κ) | 0.653 (1) | ContraWR 0.692±0.012 | −0.04 | 负 |
| ISRUC (κ) | 0.712 (1) | CBraMod-pre 0.754±0.006 | −0.04 | 负 |
| TUAR (κ) | 0.620±0.030 (3);调参 0.658 (1) | CBraMod-pre 0.715 (1) | −0.06~−0.10 | 负 |

**09-06 全三 seed 后:4 赢(TUSZ、CHB-MIT、IIIC、TUEP)3 平(TUEV 0.690±0.029 vs REVE 0.685、TUAB、ADFD)5 负**;
TUEV 的 duplex 三 seed 与 REVE 持平(rot2 变体 0.733 才赢),耦合对 raw 的 +0.15 不受影响。细表见 FINDINGS 6.5。

## 4. 预训练行(ptS,统一配方;FINDINGS 5.9)

checkpoint duplex_v2 60k + LaBraM/CBraMod 式微调(lr 5e-4、layer decay 0.65、warmup 2、前 2 epoch 只训未加载张量)。
12 格对从零:**3 赢**(Siena +0.35、TUSZ +0.02、Sleep-EDF +0.02)**9 输**;对 baseline 2 赢 1 平 7 输。
预训练帮的全是标签稀缺/噪声大的语料;标签充足时成为约束。主表用 ptS 行,零散预训练格进附录。
TUSZ ptS seed 1 = 0.694,seed 2 在跑。

## 5. 耦合消融账(同编码器,raw → duplex;FINDINGS 6.2)

| 语料 | raw | duplex | Δ | seed |
|---|---|---|---|---|
| TUEV | 0.536±0.034 | 0.709 / rot2 0.733±0.016 | **+0.17~+0.20** | 3 |
| IIIC | 0.466±0.008 | 0.479±0.008 | +0.013 | 3 |
| CHB-MIT | 0.667±0.047 | 0.713 | +0.05 | duplex 1(seed 1/2 amd 在跑) |
| TUSZ | 0.671±0.026 | 0.633 | −0.04 | duplex 1(seed 1/2 amd 在跑) |
| TUEP / ADFD / CAUEEG / TUAR / TUAB / Sleep-EDF / ISRUC / Siena | — | — | 0 | 3 |

tusz_type(五类发作形态)raw 0.121 vs duplex 0.132:两者近随机(被试不相交下发作类型不可学),**无结论**。

## 6. 架构对照(raw 前端,三 seed;FINDINGS 6.3)

| | 8 频带 + 三轴 | nb=1 不分频 | flat 不分轴(0.90M) |
|---|---|---|---|
| TUSZ | 0.671±0.026 | 0.558±0.041 | 0.448±0.048 |
| CHB-MIT | 0.667±0.047 | 0.245±0.068 | 0.136±0.016 |

两者都撑起发作检测,三轴更关键;CBraMod 从零(CHB-MIT 0.317±0.167)落在 nb1/flat 水平。
门控变体(fusegate / hybrid_gate)三 seed 在 TUEV 低于 duplex、在发作语料与 raw 同水平:不采用。

## 7. 移植门(tokenizer → CBraMod 编码器,频带当通道;FINDINGS 6.1)

| seed 0 | CBraMod 自带(3 seed) | 同前端耦合关 | 移植耦合开 |
|---|---|---|---|
| TUEV κ | 0.564±0.019 | 0.595 | **0.632** |
| TUSZ AUC-PR | 0.482±0.043 | 0.422 | 0.475(平) |

第三点写法:可迁移、事件任务提升、发作检测持平、增益归因于耦合内容;不宣称即插即用普遍提升。
直接移植(频带均值池化)11 语料 1 赢 2 平 7 输 1 无结果,已被修好版取代,只作附录。

## 8. 在跑 / 排队(09-06 早)

- 第 9 波 74/74 落地。amd/b2 空;torch 的 tmux ssh 会话已断(pane 掉回本地 shell),需 Zhizhe 重连。
- 待投:TUAB duplex seed 1/2(20 h 级,amd)。

## 9. 集群与预算

- **amd**(`/work1/chenyuyou/yifanwang/Zhizhe/PACLock`,mi2104x 4×MI210,强制独占,20 节点):node-hours 约 1450/2250。
  `slurm/configs_packed.slurm <cfg[:seed]>×4`、`seeds_packed.slurm`。工作簿与文档只在 amd 维护。
- **b2**(`/ocean/projects/cis260249p/qren2/Zhizhe/PACLock`,h100-80 12 SU/h):唯一有 TUEG 切片与全部预训练 checkpoint 的集群;
  processed 只有 6 语料;队列慢。scp 不通,用 `ssh b2 'cat …'`。
- **torch**(NYU,`/scratch/zz5070/PACLock`,只能经 tmux 窗口 `torch`):conda `py312`(numpy 2.0.2 / scipy 1.13.1 钉死);
  `PACLOCK_DATA=/scratch/zz5070/data/raw`、`PACLOCK_PROC=/scratch/zz5070/data`,12 语料 processed 齐(从 amd rsync,校验一致);
  `slurm/torch_run.slurm <cfg> [seed]`(h100_tandon,`-A torch_pr_63_tandon_advanced`,通常半小时起,但会整天饱和);
  `slurm/torch_cpu.slurm`(cpu_short ≤4 h,内存上限 120 G);结果每 30 min 由 `sync/push_runs.sh` 推回 amd。
  checkpoint 走仓库 `ckpt` 分支。
- 教训(已写进代码):长跑必须带 `max_hours`,且在 eval 步检查(TUAB 险过 24 h wall);stage-1 冻结期不计 patience;
  train.py 不存中间 checkpoint,被 wall 杀掉即全丢。

## 10. 悬而未决 / 待 Zhizhe

1. 看稿后推 Overleaf(Intro / RW / Method v3 / Setup / results 备注)。Intro 的贡献顺序要按 §2 倒过来,标题是否去掉
   "for EEG Foundation Models"待定。
2. b2 SU 续申;Yifan 在 9-18 前注册 reciprocal reviewer;AWS key / HF token 轮换。
3. 第 9 波尾巴落地后:重灌工作簿(消融 sheet 加 raw 三 seed、架构对照、门控),写 FINDINGS 6.x 定稿,更新 results.tex。

## 11. CF2 线:无频率轴的 CroFreMo(2026-09-07 投;Zhizhe:把它当优化任务做,目标是最好的模型,不是消融)

判断依据:同一个耦合 token 放进没有频率轴的 CBraMod 两处都正,放进自带频率轴的三轴编码器十个语料为零——
轴把 token 的信息吸收了。CF2 去掉频率子层,让 token 承载跨频信息,并在去除的同时加东西。
代码(默认关,旧配置逐位不变):`freq_mixer: none` 现允许与 duplex 同用(FreqNone 子层直接跳过);
`space_over_bands`(频带折进空间注意力,每个 patch 上 C×2nb 个 token 一起注意);`coupling_strength`
(每个频带的 |Z| 列经零初始化线性层加到 fused 行)。

阶段一(单 seed,tuev / iiic / chbmit / tusz):
| 变体 | 内容 | 参数量 |
|---|---|---|
| cf2_v0 | 无轴,行互不通信(只靠 token + 池化) | 1.23M |
| cf2_v1 | 无轴 + 频带折进空间注意力 | 1.23M |
| cf2_v2 | v1 + 耦合强度特征 | 1.23M |
| cf2_v1d192 | v1 + d_model 192 | 2.74M |
| cf2_v1raw | v1 的耦合关闭对照 | 1.22M |
对照:三轴 duplex(1.63M)。选择规则:四语料相对三轴 duplex 的平均差最大者进阶段二;耦合贡献 = 变体 − 其 raw 对照。
阶段二:胜者(+ 其 raw 对照)跑其余 8 语料,单 seed。补 seed 只在 Zhizhe 指示后。
