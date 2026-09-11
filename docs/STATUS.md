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

## 1. 现状一句话(2026-09-09)

CF1 与 CF2 两条线的实验目标已完成。**定稿模型 = CF2 v1d192**(无频率轴 + 频带折进空间注意力 + d_model 192,2.74M);
**耦合归因在定稿配置下四格全正**(CHB-MIT +0.136、TUEV +0.107、TUSZ +0.063、IIIC +0.027)——会后指出的"核心设计没有全局分量"已翻过来。
CF1(加法移植进 CBraMod)可归因但净收益只有 TUEV,退为支撑段。**主表全部单 seed**(Zhizhe 规矩);补 seed 待指示。
ICLR 2027:摘要 9-18,全文 9-25。论文按新主线在 MacBook 本地重写中(未推 Overleaf)。

## 2. 定稿模型对老模型与最强 baseline(单 seed;老模型与 baseline 三 seed)

| 语料 | 指标 | v1d192 | 耦合关 | 三轴 duplex(老) | 最强 baseline | Δ vs baseline |
|---|---|---|---|---|---|---|
| TUSZ | AUC-PR | **0.698** | 0.635 | 0.639±0.029 | FFCL 0.545±0.024 | +0.153 |
| CHB-MIT | AUC-PR | **0.716** | 0.580 | 0.698±0.045 | TFM-Tokenizer 0.627±0.021 | +0.089 |
| TUEP | AUROC | 0.799 | — | 0.810±0.003 | EEGPT-scr 0.786±0.018 | +0.013 |
| IIIC | κ | 0.447 | 0.420 | 0.479±0.008 | REVE-pre 0.436±0.002 | +0.011 |
| TUEV | κ | 0.680 | 0.573 | 0.690±0.029 | REVE-pre 0.685±0.032 | −0.005 |
| TUAB | BAcc | 0.820 | — | 0.816(1) | ST-Transformer 0.820±0.004 | 0.000 |
| Sleep-EDF | κ | 0.639 | — | 0.642±0.011 | ContraWR 0.692±0.012 | −0.053 |
| CAUEEG | BAcc | 0.511 | — | 0.525±0.012 | BIOT-scr 0.561±0.009 | −0.050 |
| ISRUC | κ | 0.698 | — | 0.702±0.007 | CBraMod-pre 0.754±0.006 | −0.056 |
| ADFD | BAcc | 0.461 | — | 0.505±0.050 | BIOT-scr 0.525±0.017 | −0.064 |
| TUAR | κ | 0.600 | — | 0.620±0.030 | CBraMod-pre 0.715(1) | −0.115 |
| Siena | AUC-PR | 0.201 | — | 0.170±0.070 | REVE-pre 0.518±0.096 | −0.317 |
2 大赢、2 小赢、2 平、6 输——与老模型同一格局,发作检测领先更大。TUAB 口径为 Balanced Acc(代码 primary 是 AUROC,比较时须换列)。

## 3. 设计对照(CF2 家族,单 seed)
| 语料 | v0d192 不折叠 | v0d192+强度 | **v1d192 定稿** | v1d256 | v3 折叠+强度+d192 |
|---|---|---|---|---|---|
| TUSZ | 0.612 | 0.643 | **0.698** | 0.631 | 0.671 |
| CHB-MIT | 0.630 | 0.513 | **0.716** | 0.608 | 0.671 |
| TUEV | 0.690 | 0.721 | 0.680 | 0.679 | 0.661 |
| IIIC | 0.458 | 0.448 | 0.447 | 0.465 | 0.427 |
| TUAR | 0.624 | 0.626 | 0.600 | 0.657 | 0.545 |
| ADFD | 0.498 | 0.467 | 0.461 | 0.454 | 0.361 |
| Siena | 0.239 | 0.398 | 0.201 | 0.269 | — |
| Sleep-EDF | 0.610 | 0.614 | 0.639 | 0.653 | 0.633 |
| CAUEEG | 0.538 | 0.541 | 0.511 | 0.475 | 0.541 |
读法:折叠给发作两格 +0.06~+0.09、伤小语料 0.02~0.04;强度特征帮 TUEV/Siena/CAUEEG、在 CHB-MIT 崩;d256 救小语料、丢发作两格;
全叠(v3)在小语料崩。没有变体处处最好,按"≥ 老模型格数 + 发作检测优先"定 v1d192。耦合关闭对照(§13)在 d192/d256 两档同向。

## 4. CF1 结论(§14):可归因;净收益 TUEV +0.077 与 **CHB-MIT +0.148**(09-09 21:33 落地:自带 0.317±0.167 / raw 0.341 / 耦合 0.465,耦合对 raw +0.124);官方权重版失败(TUEV −0.022、TUAR −0.127)。Sleep-EDF 两臂在 b2 排队。

## 5. 机制句(全文主线)
**编码器缺跨频通路时,耦合 token 承重;不缺时,冗余。** 三轴(有频率轴):十格为零;CBraMod(有 rfft 谱分支):TUSZ 负、TUEV 正;
无轴 CroFreMo(无任何跨频通路):四格全正。三种编码器 × 耦合开关是 Results 的核心表。

## 6. 在跑 / 集群
- b2:chbmit_cbramod_add_cpl(≈20h)、sleepedf 两臂排队。amd、torch 空。
- torch:tmux `torch` 窗口;直连需 Zhizhe 认证;`~/.ssh/config` 已加 ControlMaster(下次 Zhizhe 在窗口里 ssh 后可复用 8h);`push_runs.sh` 循环随会话死,重连后重启。
- 仓库:`runs/**/result.json` 全部跟踪(1629+);`*.npz` 已忽略;torch/b2 的本地未跟踪结果备份在 `runs_backup/`。

## 7. 待 Zhizhe
补 seed 指令(建议:TUSZ/CHB-MIT/TUEV/IIIC/TUEP/TUAB 各 +2 seed,四格耦合关闭 +2 seed,约 20 任务、amd 一天半);看稿推 Overleaf;标题。

## 15. CF1 新宿主移植(2026-09-10 投递;预测事先写下)

**设计**:替换式(宿主的 patch 嵌入换成我们的 tokenizer,编码器与头不动;每个(电极,行)当一个宿主"电极",位置 = 电极位置 + 可学行偏置;
行在编码器后均值池化,头形状与原生相同),只报"宿主从零 → 宿主 + 我们的 tokenizer(开耦合)",单 seed。归因不在此表做(消融节已有)。
依据:CBraMod / LaBraM / TFM-Tokenizer 的惯例——主表一行全模型,消融另起一节;TFM 插进 BIOT/LaBraM 也是单臂。

**矩阵**(单 seed;native 三 seed 已有者标 *):
| 宿主 | 跨频通路 | TUEV | IIIC | CHB-MIT | 预测 |
|---|---|---|---|---|---|
| LaBraM 5.86M(从零) | 无 | 0.372* → **0.522**(+0.150,09-10 04:40 命中) | 0.406* → 0.420(+0.014,≈2σ,弱正,09-10 11:01) | 0.360* → 0.377(+0.017,在 LaBraM ±0.050 噪声内,22h 时限停在 ep 11) | 三格正(CHB-MIT 因宿主训不好) |
| REVE 69.5M(从零) | 无 | native 0.303 → **0.544**(+0.241,09-10 06:01 命中) | native 0.299 → **0.377**(+0.078,09-10 15:42 命中) | — | 两格正(69M 从零训不起来,预训练版 0.685/0.436) |
| CBraMod(已完成) | rFFT 分支 | 0.564 → 0.632/0.641 | 0.393 → 0.396 | 0.317 → 0.465 | 已得:形态类正、TUSZ 负 |
判据:换头 − native 超过 native 行的 seed 标准差。不跑 TUSZ/TUAB/睡眠/认知(机制预测零或负;TUSZ 边界已由 CBraMod 给出)。

**实现**:`labram_paclockfe_adapter.py`(d 200、patch 200 与我们网格对齐;rel_pos_bias 关,偏差已记)、`reve_paclockfe_adapter.py`
(前端 patch 180 = REVE 步长,P = h 网格对齐;d 512);`slurm/smoke_gpu.slurm` GPU smoke 通过(REVE 换头 batch 16 峰值 20.7 GB)。
**投递**:amd — CF1_labram2(IIIC、CHB-MIT,batch 16×4)、CF1_reve(4 个:两 native + 两换头,batch 16×8)、LaBraM TUEV 由 auto-launcher
在 `processed_labram/tuev` 重建完成后投(数据曾丢失,已重建)。
**教训**:配置注释里写了 `\\n` 字面量把 `name:` 吞进注释(REVE 首投 KeyError),已修;LaBraM batch 64 OOM → 16×4。

## 16. CBraMod 重预训练对(TFM 设置,2026-09-10 搭建;Zhizhe 拍板)

**为什么**:TFM-Tokenizer 的宿主实验不是"从零"也不是"挂发布权重",是**用新 tokenizer 把宿主重新预训练一遍再微调**
(BIOT "replaced raw EEG inputs with token embeddings while following the original training protocol";LaBraM "substituted its
neural tokenizer with ours during masked EEG modeling",预训练数据就是四个下游集)。我们挂发布权重失败的原因(编码器绑定自己的
token 分布)在这个设置下不存在。

**设计**(一对,同一循环、同一池、同一目标,唯一变量是 tokenizer):
- `native`:vendored CBraMod 原样(PatchEmbedding + mask_encoding + 位置卷积 + criss-cross + proj_out)。
- `crofremo`:同一编码器/位置卷积/proj_out,PatchEmbedding 换成我们的前端(8 行/电极,行当通道);被掩的 (电极, patch) 在
  原始信号上先置零(滤波器组感受野不泄露)再把整格的行换成可学 mask token;行在编码器后均值池化再 proj_out。
- 目标 / 掩码 / 优化器 = `vendor/cbramod/pretrain_trainer.py`:MSE on masked cells、Bernoulli 0.5、AdamW 5e-4 / wd 5e-2、余弦到 1e-5、clip 1。
- 池:`tueg_slice`(706,817 × 10 s,1,963 h,5,224 被试,已剔除与 TUSZ 共享的 session;16 双极通道 @200 Hz),CBraMod 自家也是 TUEG。
- 微调:`experiments/*_cbramod_pretrained` 配方(lr 1e-4、multi_lr、50 epoch),`pretrained_path` 指向我们的 checkpoint;
  `configs/cf1/{tuev,iiic,chbmit,tusz}_cbramod_{ptn,ptc}.yaml`(ptc 在 10 s 语料 batch 16×4)。
- 代码:`models/foundation/cbramod_pretrain.py`、`training/pretrain_cbramod.py`、两个 `configs/pretrain/cbramod_*_tueg.yaml`;
  `build_cbramod` / `build_cbramod_paclockfe` 加 `pretrained_path`;amd GPU smoke 通过(两模型前向反向、checkpoint 往返加载严格匹配)。
- 算力:MI210 上 crofremo batch 128 一步 25 s、native 2.9 s;b2 H100 上 300 步探针在跑,按探针把 steps 定到单卡 ≤ 14 h;
  两个预训练并行跑在 b2,微调 8 个单 seed 回 amd。**预测**:ptc > ptn 在 TUEV / IIIC / CHB-MIT;TUSZ 是边界。

**算力调度(09-10 17:30)**:b2 的 h100-80 队列前有 2,500+ 任务,L40S 孪生也在排(`~/twin_pre.sh` 撤后起者);
另在 torch(h100_tandon,空)投同一对预训练的 **TFM 同款池版本**:池 = TUAB/TUEV/TUSZ/CHB-MIT 的训练集(TFM 就是四个下游集合起来预训练),
`configs/pretrain/cbramod_{native,crofremo}_ds4.yaml`,同样 12k 步。两套池各自成对比较;先出哪套用哪套,另一套作补充。

**进展(09-10 19:30)**:amd 上四语料池 12k 步对已完成(原生 0.125 s/步、带前端 4 卡 DP 0.33 s/步——smoke 的 25 s 是首次内核编译的假象);
8 个微调(`*_cbramod_{ptn,ptc}_ds4`)已起,checkpoint 严格加载(step 12000)。60k 步对在跑(FT 由 `slurm/auto_ft.sh ds4_60k` 自动投,
按日志"pretraining done"判完成——登录节点 python 无 torch,首版用 torch.load 判的启动器不会触发,已换)。b2 上 TUEG 池对在 L40S 上跑
(孪生守护已撤 h100 版)。三套池 × 两种 tokenizer,微调各 4 语料。

**暂停(09-10 21:40,Zhizhe):宿主重预训练这一块整体再考虑,先不跑。** 已撤:amd 上 60k 带前端预训练(42k/60k 时撤)、40-epoch 对(315k 步,刚起 9 分钟)、
b2 上 TUEG 池带前端版(8k/12k 时撤);自动投递与孪生守护全部停。保留:已完成的 checkpoint(四语料池 12k 两个、60k 原生、TUEG 原生),
以及正在跑的 12k 对的 8 个微调(FT_ds4_a/b,不撤,给第一个数据点)。已得的早期数字:原生 tokenizer 经我们 12k 步预训练后
TUEV 0.493 / IIIC 0.291,低于从零(0.564 / 0.393)——12k 步(1.5 遍池子)是有害的半熟先验;CBraMod 官方是 27k 小时 × 40 epoch。
待议问题:预算怎么定(按 epoch 对齐 = 315k 步,带前端约 29 小时)、池用哪个、这一层在论文里值不值。

## 17. 论文(2026-09-11):审稿意见的第一类(写作)修订已推 Overleaf

0503c83 + 022f59d,18 页、0 未定义引用;清单见 `paper_drafts/v2026-09-11/NOTE.md`。以后每次改动经 `push_overleaf.sh`(编译门控)推送。
第二类(图表)与第三类(实验:同 token 数幅度-only、自身相位、surrogate、合成 PAC)待做。
第二轮(09-11):审稿人第二次意见的写作/逻辑项全部处理,Overleaf 296050d;清单 `paper_drafts/v2026-09-11b/NOTE.md`。
所有表格数字改为从 `results/cells_2026-09-11.json` 生成(样本标准差),附录全矩阵由 `gen_matrix` 从 runs 重建。

**12k 对的微调结果(09-11 01:00,单 seed;四语料池 12k 步预训练,同一微调配方)**
| 语料 | 原生 tokenizer(ptn) | 我们的 tokenizer(ptc) | 配对差 | 参照:从零 / 官方预训练 |
|---|---|---|---|---|
| TUEV κ | 0.493 | **0.654** | **+0.161** | 0.564 / 0.645 |
| IIIC κ | 0.291 | 0.357 | +0.066 | 0.393 / 0.313 |
| CHB-MIT | 0.256 | 跑 | | 0.317 / 0.233 |
| TUSZ | 0.302 | 跑 | | 0.482 / 0.434 |
读法:同一个 12k 步预算下,原生 tokenizer 的预训练是有害的半熟先验(三格全低于从零),而换成我们的 tokenizer 后 TUEV 达到 0.654,
高于从零(0.564)也高于 CBraMod 官方 27k 小时预训练(0.645)。这是"同预算下我们的 token 让宿主预训练学到更多"的第一格证据(单 seed)。
第三版(09-11 晚):正文只讲一个模型——三轴全部退入附录"早期设计"一节;新增定稿模型的 TUEV 逐类别表(每类都涨,
GPED 0.59→0.83、PLED 0.50→0.58、SPSW 0.07→0.20,macro-F1 0.48→0.58);Overleaf 9bfa2b3;`paper_drafts/v2026-09-11c/NOTE.md`。

## 18. 归因对照:逐频带解析特征、无跨频对齐(2026-09-11 投递)

审稿意见第 4 条的最小版本(领域标准是"关键部件对一两个合理替代",见 CBraMod/LaBraM/TFM 的消融规模):在 duplex 与 waveform-only 之间加一臂——
`pac_token_mode: own`:每个频带的交互 token 用**自己的**相位特征,不做跨频对齐(h_j = a_j ⊙ p_j/|p_j|),行数、门、编码器、配方、seed 全同于定稿。
验证(GPU smoke,`smoke/smoke_own.py`):扰动低四个频带的相位,`own` 下高四个频带的交互 token 变化 0.000e+00,`measured` 下 1.97——干预精确。
配置 `configs/cf2/{tuev,iiic,tusz,chbmit}_cf2_v1d192own.yaml`,amd 一个节点,单 seed。
**事先写下的读法**:own ≈ duplex → 收益来自解析信号特征(幅度+相位),跨频对齐不是原因,摘要归因改写为"解析信号前端";
own ≈ waveform-only < duplex → 跨频对齐本身带来增益;介于两者之间 → 两者各占一部分,按差值报。token 数对照引用 CBraMod 加行实验(同八行,波形 vs 交互)。
**09-11 05:30,Zhizhe:重预训练这条线在他考虑完之前一切停止——12k 步明显不成立,要跑就跑完整预算。** 已撤 CHB-MIT/TUSZ 两个未完成的
微调(FT_ds4_b);amd/b2 上该线无任何任务、无自动投递。已落地的 TUEV/IIIC 两格仅作记录。保留这条线时不得再投任何任务。

## 19. 两条预训练线(2026-09-11,Zhizhe 拍板:要提升——好的语料不掉、差的往上拉;宿主按 CBraMod 原始设定跑,一次跑对)

**发现并修正**:定稿模型的 patch 是 **50 采样 = 0.25 s,PAC 窗口也是 0.25 s**(cf2 配置继承自 _diag duplex),论文里写的"1 s throughout"是错的;
CBraMod/LaBraM 移植用 1 s、REVE 用 0.9 s。论文已改。旧预训练 checkpoint 是 1 s 网格,所以 ptS/ptR 加载时跳过 3 个 tokenizer 张量。

**A. 我们的模型**:`aux_target: raw_patch` + `aux_mask_mode: cell`(build.py `_raw_patch_loss`)——CBraMod 式:按 (电极, patch) 格掩 0.5,
先把原始信号在被掩格置零再过前端(滤波器感受野不泄露),整格所有行换 mask token,编码后对格内行取均值预测该格 50 个原始采样。
GPU smoke:损失有限;泄露检验——只改被掩格的原始采样,被掩格的预测变化 0.000e+00。配置 `configs/pretrain/axisfree_rawpatch.yaml`:
架构 = 定稿 v1d192(patch 50),池 = 12 语料训练集 + 干净 TUEG 切片,AdamW 5e-4 / 0.05,余弦,clip 1,步数待探针,编号存 checkpoint。
微调:`checkpoint:` + 从零配方(ptR 式);验收标准 = 好的不掉、差的上拉,逐语料对从零比。

**B. 宿主 CBraMod,原始设定**:`configs/pretrain/cbramod_{native,crofremo}_tuegc.yaml`——掩码 0.5、AdamW 5e-4 / 0.05、余弦到 1e-5 无 warmup、
clip 1、batch 128、**40 epoch = 219,000 步**,池 = `tueg_slice_clean`(700,126 × 10 s,1,945 h,下游被试与 TUSZ 会话已剔除)。
差异如实记:他们 30 s × 19 通道,我们 10 s × 16 通道;数据 1,945 h 对 27,000 h。每 20k 步存编号 checkpoint(预算曲线)。
微调 `configs/cf1/*_cbramod_{ptn,ptc}_tuegc.yaml`。

**算力**:切片正经 Mac 中转从 b2 流向 amd(89.6 GB,`stream_slice.sh`),amd /work1 有 133 TB 空闲;到位后 native 单卡(≈8 h)、
crofremo 4 卡 DP(≈20 h,续跑链)、我们的模型探针后定步数。全部在 amd,不用 b2 的 SU。

## 20. 补 seed 到 3(2026-09-11 Zhizhe 指示:只补进论文表格的实验)

清单 `results/seed_plan_2026-09-11.json`(127 个缺失的 (实验, seed)):定稿模型 12 语料 ×2、消融(waveform-only / d256 / d256-raw / own)4 语料、
CBraMod 加行 5 语料 ×2 臂、CBraMod 替换 tuev/tusz、LaBraM 换头 3 语料、REVE 从零与换头 tuev/iiic、TUAB 三轴、TUAR 全部 baseline 单 seed 格。
不补:设计变体表(思路性消融)、CBraMod 官方权重版、d128 阶段一变体。
**amd**:113 个任务、29 个打包节点已投(SEED_00..28),收集器 `wait_set.sh` 盯 `results/seed_amd_targets.txt`。
**torch 待投**(MI210 上跑不进 24 h 的 10 个:CBraMod add_raw/add_cpl × tusz/chbmit × seed 1,2 + tusz crofremo_bands × seed 1,2):
`results/seed_torch_pending.txt`,等 torch 会话重连后用 `torch_run.slurm <cfg> <seed>` 投。
**b2 投递(09-11 08:10)**:CBraMod 原始设定对已在 b2 L40S 上起——native(约 0.2 s/步,219k 步 ≈ 12 h,一次续跑保险)、crofremo(0.92 s/步 ≈ 56 h,
两次续跑链;续跑路径已用探针验证"resumed from step 300")。L40S 计费约 1 SU/GPU·h,合计 ≈ 70 SU。我们模型的预训练探针在 L40S 上 OOM(batch 96),
batch 改为 24/48(0.25 s 网格每 10 s 窗口 40 patch × 256 token),探针重投中,出每步耗时后定步数。
**我们模型的预训练已投(09-11 09:30,b2 L40S)**:探针 0.6 s/步(batch 24,26.7 GiB);池 1.9M 窗口(6 语料 + 干净切片),一个 epoch ≈ 63k 步 ≈ 10.5 h,
40 epoch 不可能(17 天);投 **150k 步 ≈ 2.4 个池 epoch ≈ 25 h**,续跑链一次,每 25k 步存编号 checkpoint 做预算曲线。逐语料损失尺度差异大
(CHB-MIT 逐窗归一化 ≈ 0.25;TUH 语料原始 /100 有重尾窗口,单次 5–20),与 CBraMod 一样用原始 MSE,如实记。
**ptR Siena 落地**:预训练起点 + 从零配方 = 0.422(从零 0.170±0.086,ptS 0.456±0.034)——Siena 上先验的收益不依赖保守配方。

**归因对照首格(09-11 09:47)——TUEV:waveform-only 0.573 / own(逐频带解析特征,无跨频对齐)0.578 / duplex 0.680。**
own ≈ waveform-only,duplex 高 0.10 → 按事先写下的读法:TUEV 上的增益来自跨频对齐本身,不是解析信号特征。单 seed;IIIC/TUSZ/CHB-MIT 在跑。
**ptR TUEV(10:07)**:预训练起点 + 从零配方 = 0.636(从零 0.690±0.036,ptS 0.642)——换配方没救回来,差在起点不在配方;Siena 0.422 则说明标签稀缺处先验有效。
**TUSZ(12:47)**:waveform-only 0.635 / own 0.578 / duplex 0.698——own 甚至低于 waveform-only,duplex 高 0.12;同向。IIIC/CHB-MIT 在跑。
**IIIC(15:08)**:waveform-only 0.420 / own 0.430 / duplex 0.447——同向,量小(IIIC 一贯如此)。CHB-MIT 在跑。
**CBraMod 原生 tuegc 预训练完成(219k 步,b2 15:59)**,checkpoint 已流到 amd,4 个 ptn_tuegc 微调投 amd。
**论文(09-11 17:30,Overleaf 75fc07e)**:归因对照三臂表进正文(`tab:control`),摘要/引言/方法/结论的归因从"交互行整体"收紧到"跨频对齐本身";
`paper_drafts/v2026-09-11e/NOTE.md`。待办:补 seed 落地后重跑 `scripts/gen_tables.py` 重生成所有表,并把 caption/setup 里"单 seed"的说法改掉。
**CBraMod 原生 tokenizer、原始设定预训练(40 epoch,1,945 h 切片)后微调(09-11 19:49,单 seed)**:TUEV 0.500、IIIC 0.324
(12k 步版 0.493 / 0.291;从零 0.564 / 0.393;官方 27k h 权重 0.645 / 0.313)。40 epoch 也没让原生 CBraMod 在 TUEV 追平从零——
在 1,945 h 的数据量上,原生目标的先验本身就弱;官方的 27k h 才是它起作用的原因。带前端的配对项 47k/219k,约两天后出,届时对着这两格读。
CHB-MIT / TUSZ 的 ptn_tuegc 微调在跑。

## 21. 补 seed 中途汇总(2026-09-11 21:10,87/127 落地;括号内 seed 数)

定稿模型三 seed:TUSZ 0.680±0.048、CHB-MIT 0.669±0.041、TUEV 0.655±0.037(2)、IIIC 0.455±0.018、TUEP 0.799±0.005、Sleep-EDF 0.633±0.012、
CAUEEG 0.518±0.029、ISRUC 0.697±0.015、ADFD 0.456±0.073、TUAR 0.591±0.012、Siena 0.220±0.030、TUAB 0.820(1)。
**单 seed 时的两个头条缩水**:CHB-MIT 0.716 → 0.669(seed 0 是高的;对 TFM 0.627 仍 +0.04 但在噪声内);TUEV 0.680 → 0.655(对 REVE-pre 0.685 变成小输)。
TUSZ 对 FFCL 仍 +0.135,稳。
对照臂三 seed:TUSZ raw 0.635 / own 0.611 / duplex 0.680(对齐 +0.045,own 低于 raw);TUEV 0.554(2) / 0.590 / 0.655(2)(own 拿回约三分之一);
IIIC 0.434 / 0.439 / 0.455(小);CHB-MIT raw 0.633±0.076(2;seed 1 = 0.687!)/ own 0.585(1) / duplex 0.669 → **CHB-MIT 的 +0.136 是 seed 运气,三 seed 约 +0.04、在噪声内**。
换头三 seed:CBraMod TUEV scratch 0.564 / +raw 0.584 / +cpl 0.634(归因 +0.05、净 +0.07,稳);LaBraM TUEV 0.372 → 0.502、REVE 0.319 → 0.556(稳);
LaBraM IIIC 0.406 → 0.401(2)——变零。
结论:稳的是 TUSZ 的领先、TUEV 上三个宿主的换头收益、CBraMod 的归因;缩水的是 CHB-MIT 头条和 TUEV 持平;论文里的数字和措辞要按三 seed 重写。
**CHB-MIT own(23:29,seed 0)**:0.651——raw 0.633±0.076(2)/ own 0.651 / duplex 0.669±0.041,三臂全在噪声内,CHB-MIT 上分不出;四语料对照臂 seed 0 全齐。
**ptn_tuegc CHB-MIT / TUSZ(09-12 00:10,单 seed)**:0.440 / 0.491(从零 0.317±0.205 / 0.482±0.052;官方权重 0.233 / 0.434)——原生 tokenizer 的 40 epoch 预训练在发作两格帮(CHB-MIT +0.12),在 TUEV/IIIC 伤;四格齐。带前端配对项 66k/219k。
