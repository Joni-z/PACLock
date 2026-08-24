# 文献调研归档(2026-08-24)

四条并行调研线的完整报告,支撑 `DIRECTION.md` 的每一条结论。
检索窗口:2025-11 → 2026-08(另含必要锚点)。英文为 agent 原文,未改写。

**勘误**:报告 2 引用的 arXiv:2607.24834(LRTC 失明)已因计算错误撤稿,论文中不可引用。

---

# 报告 1:EEG FM 全景(18 模型)

## Anchors

**LaBraM** (2405.18765, ICLR'24 spotlight): 4 downstream (TUAB, TUEV, SEED-V, MoBI). Baselines: SPaRCNet/ContraWR/CNN-T/FFCL/ST-T/BIOT. Tokenizer: YES separate stage — VQ "neural spectrum prediction" (8192×64), reconstructs Fourier amplitude+phase. Objective: masked patch → discrete codes (0.5). Protocol: FT headline, LP much worse (Appx K). 5.8M–369M; 2,500h. TUAB .8140/.9022; TUEV BAC .6409/κ .6637.

**EEGPT** (NeurIPS'24): 7 downstream (BCIC-2A/2B, Sleep-EDFx, KaggleERN, PhysioP300, TUAB, TUEV). Tokenizer: patch embed only. Objective: dual SSL — representation alignment (EMA momentum encoder) + masked recon. Protocol: **frozen LP + adaptive spatial filter is headline**; LP ≥ FT on BCIC-2B/KaggleERN (Table 8). 0.4–101M (25M on TUH). TUAB .7983/.8718; TUEV .6232/κ .6351.

**CBraMod** (2412.07236, ICLR'25): **12 datasets/10 tasks** (FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, SEED-VIG, MentalArithmetic, TUEV, TUAB) — became the de-facto 2026 suite. Baselines: EEGNet/EEGConformer + A-group + BIOT/LaBraM. Tokenizer: conv+FFT patch embed (no VQ); criss-cross attention; ACPE. Objective: masked raw recon (MSE, 50%). Protocol: FT. ~4M; 9,000h TUEG. TUAB .8234/.8876; TUEV BAC .7089/κ .6512 (reproductions much lower: .667 CodeBrain, .594 EEG-DINO, κ .5588 TFM-Tok, .525 BrainOmni).

**NeuroLM** (2409.00101, ICLR'25): 6 downstream (TUAB, TUEV, TUSL, SEED, HMC, Workload). Tokenizer: YES — text-aligned VQ (temporal+freq-mag recon + adversarial alignment to GPT-2 space). Objective: multi-channel autoregressive. Protocol: multi-task instruction tuning, no per-task FT. 254M–1.7B; 25,000h. TUAB .7826; TUEV BAC .4560/κ .4285.

## NeurIPS 2025

**REVE** (2510.21585): 10 tasks (PhysioNet-MI, BCIC-IV-2a, TUEV, TUAB, HMC, ISRUC, FACED, Mumtaz2016, MAT, BCIC2020-3). Tokenizer: linear patch embed + **4D Fourier positional encoding** (x,y,z,t) — topology-agnostic. Objective: masked raw recon (L1, block masking) + attention-pooled global-token recon. Protocol: FT headline (two-step probe→unfreeze); frozen REVE-Large ≈ +17% over frozen CBraMod. 12/69/408M; **60,000h, 92 datasets, 25k subjects**. TUAB .8315; TUEV BAC .6759. HF weights: brain-bzh/reve-base.

**CSBrain** (2506.23075, **spotlight**): **16 datasets/11 tasks** (adds Siena, HMC, TUSL to the CBraMod suite). Tokenizer: Cross-scale Spatiotemporal Tokenization (multi-scale conv + FFT energy + anatomical-region aggregation) + Structured Sparse Attention; no VQ. Objective: masked recon (50% temporal). Protocol: FT. 9,000h. TUAB .8172/.8957; TUEV BAC .6903. Weights: Google Drive via github yuchen2199/CSBrain.

**BrainOmni** (2505.18185): 11 datasets/9 tasks incl. MEG (AD65, MDD, PD31, TUAB, TUEV, FACED, WBCIC_SHU, PhysioNet-MI, ASD74, MEG-MMI, SomatoMotor). Tokenizer: YES — BrainTokenizer: SEANet encoder → cross-attention channel compression → 4-layer RVQ; Sensor Encoder (3D pos+orientation+type). Objective: masked discrete-token prediction (50%). Protocol: FT (frozen in appendix). 8.4/33M; 1,997h EEG + 656h MEG. TUAB .819; TUEV .622 (their LaBraM repro .588, CBraMod .525). HF weights.

**LUNA** (2510.22257): only 4 downstream (TUAB, TUAR, TUSL, SEED-V unseen montage). Baselines incl. BENDR/BrainBERT/EEGFormer/EEG2Rep/FEMBA/CEReBrO + EEG-GNN/GraphS4mer. Tokenizer: learned-query cross-attention channel unification (compute decoupled from channel count), conv+FFT patch, NeRF-style 3D PE. Objective: masked recon + query-diversity loss. Protocol: FT. 7/43/311M; 21,000h. TUAB .8157/.8957; TUAR AUROC .921; TUSL .802. 300× FLOPs reduction claim.

**NeurIPT** (2510.16548): amplitude-aware masked pretraining, progressive MoE, intra-inter lobe pooling; 8 BCI datasets; FT.

## ICLR 2026

**CodeBrain** (2506.09110): 10 datasets/8 tasks. Tokenizer: YES — **TFDual dual-codebook VQ** (temporal 4096 + frequency 4096; freq reconstructs DFT mag+phase); "quadratic token space"; interpretability claims. Objective: masked token-index prediction on **EEGSSM** (SSM+global conv). Protocol: FT. 15.17M; 9,246h. **First published leakage-controlled ablation** (retrain without TUAB/TUEV in TUEG: TUAB .8288 vs .8294 — slight degradation only). TUAB .8294/.9030; TUEV κ **.6912**/BAC .6428.

**Uni-NTFM** (2509.24222): 9 downstream (TUAB, TUEV, TUSL, SEED, TDBrain, ADFTD, BCIC-IV-2a, Workload, HMC). Tokenizer: heterogeneous projection (time conv + DFT/PSD MLP + raw) + MoE transformer; no VQ. Objective: dual-domain masked recon + load balancing. Protocol: **both LP and FT reported**, FT headline. 57M–**1.9B**; 28,000h. TUAB .8197/.8964; TUEV κ **.7030** (highest published) /BAC .6991. No public weights (anonymous code only).

**TFM-Tokenizer** (2502.16060): **4+1 datasets** (TUEV, TUAB, IIIC-Seizure, CHB-MIT + EESM23 ear-EEG OOD). Tokenizer IS the paper: single-channel time-freq motif VQ (8192), masked spectrogram recon training; **plug-in to BIOT and LaBraM gives 3–4% gains**. Protocol: tokenizer frozen, small transformer FT. **1.9M total**. TUAB .8152/.8897; TUEV multi κ .6189 (beats their CBraMod repro κ .5588 by ~11%).

**ECHO** (2509.22556, camera-ready unconfirmed): 12 datasets; decoder-centric seq2seq, autoregressive + in-context learning; multi-task no-FT protocol; TUEV κ .5085 (beats NeuroLM-XL .4285 with far smaller model).

## Others 2025–26

**EEG-DINO** (MICCAI'25; HF braindecode/eegdino): DINOv2-style hierarchical self-distillation, no reconstruction. **LP and FT equal citizens**; key controlled result: TUEV LP BAC 60.5/κ .6419 (EEG-DINO-L) vs LaBraM-B LP 34.6 and CBraMod LP 32.5 — **LP-FT gap ~6 pts for distillation vs 25–30 pts for reconstruction-pretrained**. 4.6/33/201M; 9,000h. TUAB FT .8207/.9100; TUEV FT κ .6809.

**DeeperBrain** (2601.06134, preprint, CBraMod lineage): 12 datasets (no TUAB/TUEV — deliberate). Conv patch + volume-conduction kernel + neurodynamics temporal bases. Objective: masked recon + **Neurodynamics Statistics Prediction** (19-D: band powers, PLV, cross-frequency coupling, entropy). **Frozen probing emphasized as headline**. First to benchmark vs REVE. Trained in 7h on one A5000.

**ALFEE** (2505.06291): 6 datasets; AR forecast + dual MAE + task loss; 16M–540M; 25,000h. TUAB .8239; TUEV .6525.

**LuMamba** (2603.19100, EUSIPCO'26, ETH): 5 datasets (TUAB, TUAR, TUSL, APAVA, TDBrain); LUNA queries + bidirectional Mamba; masked recon + **LeJEPA** (combo beats either alone — first LeJEPA on biosignals). **4.6M**; 21,600h. TUAB .8099.

**BrainRVQ** (2602.16951): 8 datasets; dual-domain 3-level RVQ; hierarchical autoregression + physiology-aware curriculum masking. TUEV κ .690.

**BandVQ** (2605.24921): 6 subject-independent datasets (no TUH — deliberate); **five per-band VQ-VAEs** (δθαβγ); region-based masking; 71 corpora / 357k single-channel hours.

**B[FM]²** (2606.20812): flow-matching generative, no masking, 307h only, claims 7/9 SOTA. **MTDP** (2603.04478): multi-teacher distillation from DINOv3+Chronos. **CaMBRAIN** (2605.28792): causal SSM streaming.

## 趋势(原文十条,压缩)

1. 数据集数分叉:标准套件 10–16(CBraMod 系),效率/临床论文 4–5;两篇 2026 主动弃 TUAB/TUEV(泄漏反应);CodeBrain 首发去泄漏消融。
2. 离散 tokenizer 复兴但**数字上没赢**——最好数字仍是连续 patch-embed 端到端(REVE/Uni-NTFM/CBraMod);离散赢在参数效率+可解释性。
3. FT 仍是头条;冻结成为必备次表;**冻结可用性由目标决定非规模**(EEG-DINO 对照:重建类 LP 崩 25–30 点,蒸馏/对齐类只掉 6 点)。
4. TUAB 饱和(全体 .80–.83);TUEV 是判别基准(κ .43–.70)。
5. 复现方差严重:CBraMod TUEV 自报 .7089,他人复现 .525–.667 —— 跨论文 5 个点内的比较无意义。
6. 规模无回报("scaling paradox",Compass);小模型反潮流(LuMamba 4.6M、TFM-Tok 1.9M、DeeperBrain 7 小时训练)。
7. 拓扑无关通道处理成为标配(3D 坐标 PE / learned-query)。
8. 架构多样化:SSM/MoE/seq2seq-ICL/自蒸馏/流匹配。
9. 多任务免微调线(NeuroLM→ECHO)改善中但仍落后单任务 FT 十几个点。
10. **神经科学结构注入是新兴分化点**(BandVQ 频带因子化、DeeperBrain 神经动力学统计目标、BrainRVQ 生理感知掩码课程、CSBrain 解剖区 token)。

---

# 报告 2:批评/评测文献(17 篇)

## 逐篇要点

1. **Negative-Control Suite** (2607.24519 v3): 5 frozen FMs vs handcrafted features on CHB-MIT/TUAB/ds004504/Sleep-EDF + CAUEEG external. Classical beats/matches FMs on most; CAUEEG: every FM 5.7–20.7pp below classical; REVE random-init BEATS pretrained on CAUEEG (−9.7pp). **Dataset-identity decoded at AUROC 1.000 by all 5 encoders.** BIOT on CHB-MIT disqualified (pretrained on it). Proposes **four-gate attribution protocol**: (i) beat stronger handcrafted family, (ii) paired 95% CI excluding zero, (iii) zero pretrain/eval overlap, (iv) pretrained>random AND real>permuted. "No task satisfies all four."
2. **EEG-FM-Compass** (2601.17883, Nat. Sci. Rev. 2026): 12 FMs × 13 datasets vs 7 classical ML + 9 specialists. LP penalty 5–15pp; specialists dominate SSVEP/sleep/CHB; no parameter-accuracy correlation 0.16M–1B. Must-have baselines: paradigm classical ML + EEGNet/ShallowConv floors; report LP and FT; LOSO + few-shot.
3. **Aperiodic/Low-Freq Spectral Bias** (2605.26434): ℓ2 recon dominated by 1/f aperiodic variance → β/γ under-encoded (near-undecodable at 50Hz); subject-identity κ .91–.94 vs task κ .21–.34 on BCI (reverses on sleep). Mechanistic basis for reconstruction-objective failure.
4. **What Do EEG FMs Capture** (2605.11410): closure analysis — classical features recover avg **79.3%** of FM advantage (MDD 99%, sleep 94%); only hard tasks have 21–44% residual.
5. **EEG-FM-Audit** (2605.26910): ASHA-tuned baselines gain +0.028–0.074 BA, erasing most FM margins; TUAB TS-SEFFNet 0.800 ≈ LaBraM 0.803 at 9,400× fewer params; NeuroGPT better WITHOUT pretraining on TUEV (+0.188). Mandates transparent baseline tuning.
6. **OmniEEG-Bench** (2606.00815): 54 datasets; LP: only 5/10 FMs beat EEGConformer; **pretraining diversity (ρ=−0.27) predicts rank, not hours/params**; near-chance families: ThingsEEG2, 40-class SSVEP, DEAP, workload, ErrP.
7. **Brain4FMs** (2602.11558): generative SSL > contrastive-augmentation; autoencoders > autoregressive; modality mismatch (iEEG-pretrained fails scalp); universal failures on emotion/concept decoding.
8. **Channel Adaptation** (2604.23091): CBraMod 5M top-1 on 4/5 vs models 31× larger; 24.8% of SFT runs BELOW their own frozen probe (negative transfer); FACED all near chance; only 20/50 comparisons survive FDR.
9. **The Identity Trap** (2606.06647): FT ≈ LP in 10/12 pairs; subject variance 26.8–58.8% of embedding variance vs label 0.5–6.6%; LEACE identity-erasure IMPROVES task accuracy up to +12pp; trait labels structurally confounded with identity in single-session data.
10. **Beyond Accuracy** (2605.17562): head-only probing — **every FM below supervised EEGNet**; mean pooling destroys probes (+21% switching to concatenation); zero-padding channel-dropout fragility is a padding artifact.
11. **EEG-FM-Bench** (2508.17742, ICML 2026): reconstruction gradients ≈ zero/negative correlation with classification gradients — "time-series-aware initialization rather than knowledge transfer"; generic TS-FMs competitive; multi-task FT causes negative transfer on MI.
12. **EEG-Bench clinical** (2512.08959): LDA/SVM beat all FMs on sleep (0.65–0.67 vs 0.17–0.19 — FMs at chance under channel mismatch), mTBI, schizophrenia; FMs win abnormal/epilepsy binary.
13. ~~2607.24834 LRTC blindness~~ — **WITHDRAWN, do not cite.**
14. **Multi-dim Generalization** (2605.28563): FM sample efficiency < supervised on BCIC-IV-2a at all budgets; 1-s patches miss ERP transients.
15. **TTA study** (2604.16926, MLHC 2026): gradient TTA degrades EEG performance; only T3A helps.
16. **NeuroAtlas** (2605.14698): 42 datasets, clinically grounded metrics. Epilepsy: only NeuroLM/REVE beat random init significantly. **Sleep: supervised advantage comes entirely from sequence context — at ℓ=1 supervised = TS-FM = EEG-FM.** BCI: "most EEG-FMs rely primarily on artifacts" (eye movements). TS-FMs ≈ EEG-FMs broadly.
17. **ERP Benchmark** (2601.00573): counterpoint — manual features WORST on ERP single-trial decoding; EEGConformer best overall; classical-baseline story is task-family dependent.

## 失败模式清单(18 条,择要)
数据集/站点身份泄漏(AUROC 1.000);被试身份主导;1/f 频谱偏置;TUEG 预训练/评测重叠(TUH=REVE 语料 44%;BIOT 直接训过 TUAB+CHB);未调参基线;冻结表征弱(而 FT≈LP 本身又是身份捷径信号);预训练不敌随机初始化(任务依赖);通道刚性+适配负迁移;mean-pooling 假象;MI 靠眼动伪迹;无 scaling law;种子方差(>8pp);AUROC 与临床效用脱节;预处理碎片化;情感/细粒度解码全体近随机;EEG 特异性未证明(TS-FM 打平)。

## 审稿人攻击清单(14 条)与"脱颖而出"配方
攻击:TUEG 重叠、缺手工特征/LDA 行、基线未调参、缺冻结探针(concatenation pooling)、缺身份可解码性+LEACE、缺随机初始化对照、缺外部队列、单 seed 小差距、蒙太奇失配、缺 TS-FM 对照、MI 眼动、目标函数编码何物、规模消融、closure 分析。
配方:baseline 栈(手工特征×2 档 + LDA/SVM + 调参 EEGNet/Conformer + 范式经典 ML + TS-FM + 随机初始化);双协议 + LOSO + few-shot;≥5 seed 配对 CI + FDR;负对照套件(身份探针、LEACE、标签置换、FOOOF、epoch 乱序);外部队列;振荡带可恢复性 + closure 残差;通道移除(非补零);四道门语言。**"过四道门的第一篇论文超越全部现有工作"。**

---

# 报告 3:失败机制根因

## 睡眠分期
判别单元=嵌在长时间语法中的持续谱态。SeqSleepNet(seq2seq 重构才是驱动力);IITNet(1→10 epoch = +2.48% acc/+4.90% F1,4 epoch 饱和);SleepTransformer(L=21,邻 epoch 注意力=AASM 过渡规则);L-SeqSleepNet(20–30 epoch 是领域标准;L=200 再 +0.7–1.9%;**flat 长序列反而伤 transformer**);U-Sleep(全夜卷积,15,660 人)。**CBraMod/CSBrain 的 ISRUC 协议也是单 30s epoch**——我们输它们输的是窗口内表征,上下文是全体 FM 对 sequence-SOTA(~85–88%)的共同差距。窗口内:AASM 定义=epoch 内频带占比(N3=慢波≥20%);**YASA(LightGBM+频带特征)=人类评分者一致性 86.6%,无深度网络**;ContraWR 的赢=全 epoch 谱摘要归纳偏置。SO-spindle 耦合真实(eLife 2025 元分析;Nat. Neurosci. 2023)但是**巩固变量,分期从不需要相位关系**。

## 运动想象
Blankertz 2008(经典):MI=检测弱 mu/beta ERD,单电极 SNR 低(容积传导:仿真中单电极仅~一半信号来自 3cm 内源);CSP 特征=**学习的空间线性组合的方差**;空间滤波必须在功率提取**之前**——**混合的功率≠功率的混合**;逐电极频带幅度先行=CSP 量在原理上不可恢复。S-JEPA(2403.11772)在 FM 语境下得出同一结论。黎曼线(2407.20250):全部竞争力管线在通道协方差 SPD 流形上。被试间变异>被试内(Front. Neurosci. 2023);个体 mu 峰频率变异→共享 8 频带网格跨骑个体边界。MIRepNet(2507.20254):"通用 FM 无法捕获 MI 特异神经模式"——为此专门做了 MI 限定 FM。CBraMod 0.514/CSBrain 0.566 BA(跨被试)vs 任务特定管线 ~0.70–0.80(被试内)。头皮 γ:EMG 宽带 20–300Hz、40–80Hz 处 5–10× 神经功率(Muthukumaraswamy 2013)→涉 γ 耦合 token 载肌电。

## 跨频耦合生物标志物地图
**成立处(=我们赢的地方)**:发作期 PAC=SOZ 生物标志物(Epilepsia 2025/PMC12440772);发作间期 ECoG PAC+ML 定位 SOZ(Cogn. Neurodyn. 2023);SEEG 发作期 PAC 定位致痫区(Clin. Neurophysiol. 2025);IED+HFO 联合体本质是跨频事件(Front. Hum. Neurosci. 2021;Brain 2023 fast-ripple 触发棘波);**AD:theta-gamma PAC 降低是最早期 EEG 特征之一**(PMC10017148;Brain Comm. 2024);PD beta-gamma PAC(PMC10971628)。PAC 特征单独在 CHB-MIT 达 97.5% 检出(IJMPB 2018)。**关键方法学点**:Cole & Voytek 2017、Lozano-Soldevilla 2016、Aru 2015、Gerber 2016——非正弦波形与尖锐瞬态(明确包括癫痫棘波)产生强"虚假"PAC;对生物标志物研究是混淆,**对分类器是特性**:耦合 token 天生是棘波/尖波/HFO-on-slow-wave 形态检测器。**不成立处(=我们输的地方)**:MI 中 alpha-高gamma PAC 只是镜像 ERD 时程(NeuroImage 2021),PAC 特征 MI 分类仅 ~70%(无增益);睡眠**分期**、情绪、负荷无 PAC 判别性文献。

## 伪迹(TUAR)
伪迹识别靠绝对量纲+宽带非振荡形态+跨通道拓扑(ICLabel, NeuroImage 2019);EMG 宽带(Goncharova 2003)。幅度归一化 8 频带振荡词表把宽带瞬态坍缩成"处处有功率",弃掉量纲与波形形状线索。

## 小认知/情感集
判别特征=持续频带功率拓扑(额 alpha 不对称;额中线 theta↑+后部 alpha↓,SVM 65–75%);n≈36–123 时低容量域特征模型必然占优("Attention Isn't All You Need for Emotion Recognition", 2601.22161)。

## 预训练目标
2605.26434(1/f 主导);**FAME(2608.01898):修复=逐频带标准化、等权时频目标,OmniEEG 24/41 SOTA——与我们的 8 频带架构天然兼容**;LaBraM 存在性证明(原始信号重建"不稳定且差",谱幅度+相位预测目标更好);**Laya(2603.16281):受控消融证明目标(非架构/数据)是主驱动,潜空间预测给最强冻结临床探针**;EEG-JEPA(2608.00114)、STST-JEPA(2607.06629)、S-JEPA(2403.11772);Stanford Sleep Bench(2512.09591):对比学习>掩码预测(PSG 域);**phase-swap PAC 预训练任务(2009.07664)**;成对相对相位移预训练(2511.11940)。

## 域限定 FM 先例
SleepFM(2405.17766 → **Nature Medicine 2025**,585k 小时);U-Sleep(npj Digit. Med. 2021);BrainBERT(ICLR 2023, iEEG);Brant(NeurIPS 2023)/Brant-2;EpilepsyFM(Neural Networks 193:108060);LEAD(2502.01678, AD);MIRepNet(MI)。**接受模式:限定域模型上顶刊顶会且被实际使用;通用 FM 正被 benchmark 怀疑论围剿。**

## 一致性检查
赢集(TUSZ/CHB/TUEV/TUEP/ADFD)=瞬态事件 ∪ PAC 既证生物标志物,且 PAC 估计量对尖锐瞬态的敏感性是特性;输集=持续谱态+空间协方差+小样本;TUAB 混合型→持平。无一例外。

---

# 报告 4:tokenizer 先例与定位地图

## tokenizer-as-contribution 先例
TFM-Tokenizer(模板,详见报告 1):tokenizer 即论文;4+1 数据集;插入 BIOT(+3–4%)与 LaBraM(数据稀缺单集设置下 CHB-MIT AUC-PR +147%);消融=流移除/掩码策略/码本 256–8192/PE 移除;token 质量分析(类唯一性、同类检索 60%、谱熵频率感知、PLED 形态可视化);1.4M 全栈。CodeBrain TFDual(时/频双码本,dual 全胜消融;频码↔σ 纺锤,时码↔K 复合波)。BrainOmni RVQ×4+Sensor Encoder。LaBraM VQ(相位只是重建目标)。NeuroLM 文本对齐 VQ。BrainRVQ(双域 RVQ+重要性引导课程)。BandVQ(五频带独立 VQ-VAE——最强显式频带先验;无耦合)。NeuroRVQ(2510.13068)——**最接近相位感知:尊重相位圆拓扑的 phase-aware loss**;仍无耦合。EpilepsyFM(癫痫专用码本;期刊)。
**模式**:全部离散、全部两阶段(先训后冻)、全部靠{流消融+码本消融+插入+可解释性}证明贡献。**没有任何已发表 tokenizer 把 CFC/PAC/相位结构作为一等 token 内容。**

## 冻结惯例
EEGPT=冻结旗手;多数 FT 头条;REVE/Uni-NTFM/DeeperBrain/EEG-DINO 双报告;怀疑论文献("Are EEG FMs Worth It?" ICLR 2026 等)确立冻结探针=诚信测试;**已发表 FM 在冻结下普遍难看→冻结赢=高杠杆稀缺证据**。

## 定位地图(2026-08)
**拥挤**:规模;any-setup/拓扑无关;架构本身;通用时频 VQ;"神经科学 grounding"话术。
**空缺**:①耦合/相位作为 token 结构(无人);②顶会级 scalp-EEG 临床/阵发限定 FM(无人;癫痫在期刊、痴呆在 preprint,无人统一发作+痫样+痴呆);③小模型冻结探针赢;④临床边缘部署实数。

## 小模型先例
TFM-Tok 1.4M 上 ICLR;LUNA 效率共同头条上 NeurIPS;FEMBA-on-Edge/TinyMyo(MCU 部署);多篇 benchmark 使"小而强"成为常识→**小本身不是贡献,小+冻结赢或小+部署实数才是**。

## 三个定位模板(供组合)
T1 耦合感知 tokenizer(插入式,TFM 血统;4–5 数据集+插入 ≥2 骨干+重消融+token 可解释性+1 个 OOD)——先例最强、证据账单最小。
T2 阵发性临床 FM(域内广度换域间广度;负对照套件抢先;事件级临床指标)——空位但账单大。
T3 诚实评测冠军(1.6M 冻结特征打 30–400M;统计检验;效率表;边缘部署)——与领域可信度压力最对齐,风险=冻结赢不干脆。
**最佳组合:T1 为脊柱,吸收 T2 的临床评测故事与 T3 的冻结+效率附录——三者互容,各答一类审稿人。**
