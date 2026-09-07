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
判据 add_cpl > add_raw 且 ≥ 自带 → 铺其余 9 语料。状态:6 个门任务在 torch(h100_tandon)与 b2(h100-80)孪生排队,`twin_watch.sh` 撤后起者。

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
