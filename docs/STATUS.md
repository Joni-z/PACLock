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

## 1. 现状一句话(2026-09-07 晚)

Zhizhe 9-06/07 的判断:三轴 duplex 那款模型"基本死了"——耦合 token 在它里面十个语料为零,核心设计没有全局分量。
方向重置为两部分:**部分一** 耦合 token 补进没有频率轴的 FM 编码器(CF1,加法移植);**部分二** 为耦合 token 设计的
无频率轴小模型(CF2),目标是一个 SOTA 模型,不是消融。**硬规则:所有我们这边的实验单 seed;补 seed 只在 Zhizhe 指示后。**
ICLR 2027:摘要 9-18,全文 9-25。

## 2. 两条线的当前状态

### 部分一:CF1 加法移植(STATUS §12 的设计;FINDINGS 6.1 的旧移植作废)
CBraMod 自带 patch embedding / 编码器 / 头原样保留,我们的前端每电极加 8 行作额外通道。三臂同参数量:add_raw(8 行波形,对照)、
add_cpl(8 行耦合交互 token)、add_cplmean(读出取均值)。门:TUEV、TUSZ 单 seed;对照 CBraMod 自带 0.564±0.019 / 0.482±0.043。
判据 add_cpl > add_raw 且 ≥ 自带 → 铺其余 9 语料。

**门的判决(09-08 20:30,单 seed;CBraMod 自带为三 seed):**
| | 自带 tokenizer | add_raw(加 8 行波形) | add_cpl(加 8 行耦合) | add_cplmean |
|---|---|---|---|---|
| TUEV κ | 0.564±0.019 | 0.596 | **0.641** | 0.642 |
| TUSZ AUC-PR | 0.482±0.043 | 0.476 | **0.384** | 0.379 |
TUEV 过:耦合对 raw +0.045、对自带 +0.077。TUSZ 不过:耦合比 raw 低 0.092、比自带低 0.098,两种读出一致(0.384/0.379),不是单次噪声。
读法:加法移植下 CBraMod 已有自己的 rfft 谱分支(每 patch 的频带能量),状态类标签不缺跨频信息,多出的 8 倍 token 只带来稀释;
事件形态标签(TUEV)才需要显式耦合。与我们自己两代编码器的规律一致(耦合在 TUEV 决定性、TUSZ 中性或有害)。
**铺开首格(09-09,torch)**:IIIC add_raw 0.3815 / add_cpl 0.3962(CBraMod 自带 0.393±0.005)——耦合对 raw +0.015,但只追平自带。
至此 CF1 三格:TUEV 耦合 +0.045(对自带 +0.077,唯一实打实的赢)、IIIC +0.015(追平自带)、TUSZ −0.092(负)。
形态类正、状态类负的规律与两代自研编码器一致;CHB-MIT / Sleep-EDF 在 b2 跑。

**处置**:不按原计划铺 9 个语料;只补 4 个语料定范围——IIIC、TUEP(torch)与 CHB-MIT、Sleep-EDF(b2),各 add_raw + add_cpl,
共 8 个单 seed(`add_cplmean` 两格与 `add_cpl` 相同,已从铺开中去掉,省一半)。若 IIIC 与 TUEV 同向,部分一的主张写成
"耦合 token 提升 FM 编码器在痫样事件/形态任务上的表现",TUSZ/睡眠作为负结果如实报;若 IIIC 也负,部分一只剩 TUEV 一格,退为附录。
状态:torch 先起(tuev add_raw、add_cpl),b2 孪生已由 `twin_watch.sh` 撤。
**首格(09-07 20:24)**:tuev add_raw κ 0.5956(CBraMod 自带 0.564±0.019)——8 行 raw 额外通道本身就 +0.03;add_cpl 在跑,
它必须再高于 0.596 才算耦合有归因。
**TUSZ 半边(09-08 14:41,b2)**:add_raw 0.4762(自带 0.482±0.043,平——TUSZ 上加 raw 通道没有 TUEV 那种红利);add_cpl / add_cplmean 在 torch 跑。
**TUEV 半边过门(22:24)**:add_cpl κ **0.6407** > add_raw 0.5956 > 自带 0.564±0.019——耦合归因 +0.045,对自带 +0.077。TUSZ 半边在排。

### 部分二:CF2 无轴 CroFreMo,定稿候选 v3
v3 = 去掉频率注意力子层 + 频带折进空间注意力(`space_over_bands`)+ 耦合强度特征(`coupling_strength`)+ d_model 192(2.74M)。
阶段一(单 seed,括号为三轴 duplex 三 seed):
| | 耦合关 v1raw | v0 无轴 | v1 折叠 | v2 折叠+强度 | v1+d192 |
|---|---|---|---|---|---|
| TUEV (0.690) | 0.546 | 0.658 | 0.648 | 0.657 | 0.680 |
| IIIC (0.479) | 0.453 | 0.435 | 0.455 | 0.481 | 0.447 |
| TUSZ (0.639) | 0.534 | 0.644 | 0.612 | 0.591 | torch 跑 |
| CHB-MIT (0.699) | 0.661 | 0.569 | 0.647 | 0.628 | torch 排 |
无轴家族里耦合 token 承重(TUEV +0.10~0.13、TUSZ +0.06~0.11、IIIC +0.03 经强度特征),整体与三轴持平。
状态:v3 在 amd 跑全部 12 个语料(4 节点;同节点还带 v0d192);网格里的 v0cs 已撤,阶段一的 amd 节点已撤(d192 留 torch 孪生)。
决策规则:v3 四语料 ≥ 三轴 duplex → 定稿,按 Zhizhe 指示补 seed;否则只允许再换一次。
v3 首个落地:ADFD 0.361(三轴 duplex 三 seed 0.505±0.050;val 曲线两者同水平 0.44 vs 0.43,test 差是该语料的被试级方差——
duplex 三个 seed 是 0.562/0.512/0.441),单 seed 不能判,记录待议。

### CF2 进展(09-07 16:30;单 seed;括号为三轴 duplex 三 seed)
| | v3 折叠+强度+d192 | v0d192 不折叠+d192 | v1d192 折叠+d192 | v0 | 备注 |
|---|---|---|---|---|---|
| TUEV (0.690) | 0.661 | **0.690** | 0.680 | 0.658 | 不折叠 ≥ 折叠 |
| TUSZ (0.639) | 0.671 | 跑 | **0.698** | 0.644 | 无轴家族普遍 ≥ 三轴 |
| IIIC (0.479) | 跑 | 跑 | 0.447 | 0.435 | 阶段一 v2(折叠+强度)0.481 |
| CHB-MIT (0.699) | 跑 | 跑 | torch 排 | 0.569 | |
| 小语料 v3 | TUAR 0.545 (0.620) / TUEP 0.803 (0.810) / Sleep-EDF 0.633 (0.642) / ADFD 0.361 (0.505, 方差大) | | | | 折叠在小语料疑似过拟合 |
倾向:候选换成 v0d192(+ 耦合强度待定),等 IIIC/CHB-MIT/TUSZ 的 v0d192 落地后用掉那一次换的机会。
18:30 补:v0d192 落地 TUSZ 0.612(0.639;v1d192 0.698、v3 0.671)、TUEP 0.812(0.810)、TUAR 0.624(0.620;v3 0.545)、
ADFD 0.498(0.505;v3 0.361)、Siena 0.239(0.170)、Sleep-EDF 0.610(0.642;v3 0.633)。v3 CAUEEG 0.541(0.525)。不折叠在小语料上把 v3 丢掉的分全部拿回,TUSZ 上单 seed 略低于折叠版
(该语料 std≈0.03)。v0d192 已提前铺到其余语料;v0d192+强度在四决策语料跑。
22:30 补:**TUEV v0d192cs 0.7207**(老 0.690,v0d192 0.690)——耦合强度特征在 TUEV +0.03,超过 REVE 0.685;IIIC v0d192 0.458 / v3 0.427(老 0.479);
CHB-MIT v0d192 0.630(老 0.699);ISRUC v3 0.677(老 0.702)。cs 版已提前铺到其余 8 语料(CF2_cs_a/b)。
09-08 04:30 补:CHB-MIT v1d192(折叠+d192)**0.716**(老 0.699)、v3 0.671、v0d192 0.630;IIIC v0d192cs 0.448(老 0.479;阶段一 v2 折叠+强度 d128 0.481);
Siena v0d192cs **0.398**(老 0.170);Sleep-EDF v0d192cs 0.614(0.642);ISRUC v0d192 0.696(0.702)。
两个家族:不折叠(v0d192 / +cs)在小语料、TUEV、Siena、CAUEEG ≥ 老模型,但 TUSZ/CHB-MIT 低于老模型;折叠+d192(v1d192)在 TUSZ +0.06、CHB-MIT +0.02,
小语料未知(v3 = 折叠+强度+d192 在小语料崩,不知是折叠还是强度所致)。已投 v1d192 到其余 8 语料(CF2_v1d192_a/b)以定夺。
CF1:TUEV add_cplmean 0.6419(≈ add_cpl 0.6407);TUSZ 三臂在 b2/torch 跑。
09-08 06:30 决策表(单 seed vs 老模型三 seed;✓ ≥ 老,≈ 差 ≤0.01):
| | v0d192 不折叠 | v0d192cs +强度 | v1d192 折叠+d192 |
|---|---|---|---|
| TUEV 0.690 | 0.690 ≈ | **0.721 ✓** | 0.680 ≈ |
| TUSZ 0.639 | 0.612 | 跑 | **0.698 ✓** |
| CHB-MIT 0.699 | 0.630 | 0.513 | **0.716 ✓** |
| IIIC 0.479 | 0.458 | 0.448 | 0.447 |
| Siena 0.170 | 0.239 ✓ | **0.398 ✓** | 0.201 ✓ |
| TUEP 0.810 | 0.812 ≈ | 0.812 ≈ | 跑 |
| TUAR 0.620 | 0.624 ≈ | 0.626 ≈ | 0.600 |
| ADFD 0.505 | 0.498 ≈ | 0.467 | 0.461 |
| CAUEEG 0.525 | 0.538 ✓ | 0.541 ✓ | 跑 |
| Sleep-EDF 0.642 | 0.610 | 0.614 | 跑 |
| ISRUC 0.702 | 0.696 ≈ | 跑 | 跑 |
| TUAB | 跑 | 跑 | 跑 |
读法:折叠对发作两格是 +0.06/+0.02 的稳定优势,在小语料上小负 0.02–0.04;不折叠稳但发作两格输;强度特征在 TUEV/Siena/CAUEEG 帮、CHB-MIT 单 seed 崩(0.513)。
等 v1d192 的 TUEP/CAUEEG/Sleep-EDF/TUAB/ISRUC 落地后按"≥ 老模型的格数"定,一次定。
**09-08 12:00 定稿:CF2 最终模型 = v1d192(无频率轴 + 频带折进空间注意力 + d_model 192,2.74M;无耦合强度特征)。**
依据(单 seed vs 老三轴 duplex):TUSZ 0.698 ✓(+0.06)、CHB-MIT 0.716 ✓(+0.02)、Siena 0.201 ✓、TUEV 0.680 ≈、TUEP 0.799 ≈、Sleep-EDF 0.639 ≈、
IIIC 0.447(−0.03)、TUAR 0.600(−0.02)、CAUEEG 0.511(−0.01)、ADFD 0.461(−0.04);TUAB/ISRUC 在跑。对 baseline:TUSZ +0.15、CHB-MIT +0.09、
IIIC +0.01、TUEP +0.01 赢,TUEV 平,其余同老模型输。耦合归因(同族 v1 vs v1raw):TUEV +0.10、TUSZ +0.08、CHB-MIT −0.01、IIIC 0。
不选 v0d192(发作两格输 0.03/0.07)、不选强度版(CHB-MIT 0.513)。ISRUC v1d192 0.698(老 0.702 ≈)。TUAB 落地后进主表;不补 seed。
Zhizhe 09-08 指示:跑 d256(v1d256,4.85M),12 语料单 seed 在 amd(CF2_d256_a/b/c);若 ≥ v1d192 则定稿升级。CF1:torch 起了 tuev add_raw,b2 孪生已撤。

## 3. 已钉死的事实(供写作)
- 三轴 duplex 主表:4 赢(TUSZ、CHB-MIT、IIIC、TUEP)3 平 5 负(FINDINGS 6.5);耦合在其中只有 TUEV 决定性(+0.15,逐类别集中在 GPED/PLED)。
- 架构对照:分频与三轴各自必要(CHB-MIT 0.667 → nb1 0.245 / flat 0.136)。
- 预训练:分析章节,3 赢 9 输,机制两种;不再重跑。CF2 若定稿,其预训练行为空(现有 checkpoint 只能部分迁移),届时决定是否在 b2 重训。
- 旧移植(替换式)作废:同前端耦合关闭比 CBraMod 自带低 0.06,耦合只是填坑。

## 4. 在跑 / 排队(09-07 晚)
- amd:CF2v3_small(siena tuep tuar adfd)、CF2v3_big(caueeg sleepedf tuab isruc)、CF2b_tuev_iiic、CF2b_chb_tusz(v3 + v0d192)。
- torch:tusz_cf2_v1d192 跑、chbmit_cf2_v1d192 排;CF1 六门排。
- b2:CF1 六门排(vendor/cbramod 已补)。
- 收集器:amd `wait_set.sh`(cf2.list,含 CF1)、b2 `wait_b2.sh`、`twin_watch.sh`。

## 5. 集群与规矩
- amd:20 节点强制独占,`configs_packed.slurm <cfg[:seed]>×4`;节点空转即撤。b2:h100-80,余额不用管,数据只有 6 语料,vendor 只有 cbramod。
- torch:tmux `torch` 窗口经 vpnbox VPN;VPN 会话约三天到期,断了只能 Zhizhe 重新认证(9-06 openconnect 表单登录被拒,另行重连);
  重连后 `git pull` 并重启 `sync/push_runs.sh`。账号只有 general / tandon_advanced(无 priority 关联)。
- 长跑带 max_hours(eval 步检查);stage-1 冻结期不计 patience;train.py 不存中间 checkpoint。

## 6. 论文
完整稿在 MacBook 本地(Intro / RW / Method v3 / Setup / Results / Conclusion / 附录全表),编译 14 页 0 错误,未推 Overleaf。
主张按重置改写待 CF1/CF2 落地:部分一 = 标题主张(token content for FMs),部分二 = 无轴小模型。`paper_drafts/` 有备份。

## 7. 待 Zhizhe
看稿推 Overleaf;标题;补 seed 指令;SU 续申;Yifan reviewer 注册;密钥轮换。
