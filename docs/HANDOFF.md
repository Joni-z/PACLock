# 交接文档 — CroFreMo(原 PACLock),2026-09-12

给接手的对话。读完这一份就能接着干。仓库 `git@github.com:Joni-z/PACLock.git`,三个集群的工作目录都是它的 checkout。
**详细历史看 `docs/STATUS.md`(§1–§24,按时间追加)和 `docs/FINDINGS.md`;本文件只讲"现在是什么状态、下一步做什么"。**

---

## 1. 三个集群怎么连

### amd —— AMD HPC Fund,主力,不计费
```
ssh amd            # 已配好免密;工作目录 /work1/chenyuyou/yifanwang/Zhizhe/PACLock
```
* SLURM 强制 `--exclusive`,一个节点 4 张 MI210。**永远用打包脚本**,一个节点塞 4 个配置:
  ```
  SEED=0 sbatch -J NAME slurm/configs_packed.slurm cfg1.yaml cfg2.yaml cfg3.yaml cfg4.yaml
  # 也支持 cfg.yaml:seed 的写法,一个节点跑不同 seed
  ```
* 单个 GPU 的冒烟测试:`sbatch slurm/smoke_gpu.slurm smoke/xxx.py`(**绝不在登录节点跑计算**)。
* 预训练用整节点:`sbatch -J NAME slurm/amd_pretrain.slurm <module> [args]`,加 `--dp` 可用 4 卡 DataParallel。
* 提交必须带 `-p mi2104x`(打包脚本里已写死);登录节点的 python 没有 torch。
* 20 个节点,`sinfo -p mi2104x` 看空闲。

### b2 —— PSC Bridges-2,**通过 NSF ACCESS 分配,计 SU,账号与别人共用**
```
expect ~/.claude/jobs/18b2571f/tmp/b2_master.exp >/dev/null 2>&1   # 先做这一步!否则 Permission denied
ssh b2             # 工作目录 /ocean/projects/cis260249p/qren2/Zhizhe/PACLock
```
* **连接经常掉**,每次用前先跑上面那行 expect 脚本重建 ControlMaster。
* **scp 被禁**,传文件只能 `ssh b2 'cat 远端文件' > 本地文件`,或反向 `ssh b2 'cat > 远端文件' < 本地文件`。
* 提交:`sbatch --gpus=l40s-48:1 -t 30:00:00 -J NAME slurm/run_b2.slurm <module> [args]`。
  GPU 类型写 `--gpus=类型:数量`(不是 `--gres`)。h100-80 队列常年积压两千多个任务,**用 l40s-48**。
* 余额:`projects` 查看,现在 561 / 2,189 SU。墙钟上限 30 小时,长任务必须挂 `--dependency=afterany:<jobid>` 的续跑链(`--resume`)。
* 这台机器的 python 是 3.6,`yaml.safe_dump(..., sort_keys=...)` 会报错;改配置去 amd 上改完推 git,b2 上 `git fetch && git reset --hard origin/main`。
* **只有这台有 TUEG 预训练数据**(`../processed/tueg_slice_clean`,90 GB)。经本机中转到 amd 实测 0.35 MB/s,不可行,所以要 TUEG 的预训练只能在 b2 跑。
* b2 上只有 6 个下游语料(tuab/tuev/tusz/chbmit/sleepedf/isruc)+ 切片,其余 6 个临床语料不在。

### torch —— NYU,不计费,H100/H200
```
ssh torch          # 直连能不能用取决于 ControlMaster 是否还在
tmux capture-pane -pt torch    # Zhizhe 维护的登录会话在这个 tmux 窗口
~/.claude/jobs/18b2571f/tmp/tx.sh '<命令>' <超时秒>   # 通过那个窗口执行命令
```
* 直连需要密码/Duo,**我们登不了**;`~/.ssh/config` 里给 torch 配了 ControlMaster,只要 Zhizhe 在 tmux 窗口里 `ssh torch` 过一次,之后 8 小时内 `ssh torch` 可直连。断了就请 Zhizhe 重连。
* 提交:`sbatch -p h200_public -A torch_pr_63_general -J NAME slurm/torch_run.slurm <cfg> <seed>`。
  **不要用 `h100_tandon`**:那个分区的 QOS(`QOSMaxGRESPerUser`)会把任务永久挂起,即使一个都没在跑。通用账号的 `h200_public` 正常。
* 通用模块入口:`sbatch -J NAME slurm/torch_module.slurm <module> [args]`。
* 结果回传:`/scratch/zz5070/sync/push_runs.sh` 把 `runs/**/result.json` 和 npz rsync 到 amd(`--ignore-existing`)。会话一断这个循环就死,重连后要重启。

### 通用坑
* **`git pull` 在 torch/b2 上经常被未跟踪的 `result.json` 挡住**:先 `git status --porcelain -uall | grep '^??' | grep result.json` 把它们挪到备份目录再 pull(注意要 `-uall`,否则只列目录)。
* 本机(mac)上 `timeout` 命令不存在;前台 `sleep` 被禁,等待用远端 sleep 或后台脚本。
* 提交 git 的署名要求见 CLAUDE.md / 会话提示。

---

## 2. 论文

* 源码在 **MacBook**:`ssh mbp`(密码在 `~/.claude/jobs/18b2571f/tmp/mbp_master.exp`),目录 `/Users/zzz/Desktop/figure/ICLR2027-Paper`,**git 远程就是 Overleaf**。
* **推送一律用 `./push_overleaf.sh "commit message"`**:它先编译、检查无未定义引用,再提交、rebase、推送;编译不过不推。
* 当前 Overleaf 版本 `75fc07e`,20 页(正文超 9 页限制,待裁)。
* 所有表格由 **`scripts/gen_tables.py`** 从 `runs/` 生成(CBraMod 风格:数据集做列组、三个指标、方法分组、最优加粗),生成物在 `results/tables/*.tex`。**数字变了就重跑这个脚本,不要手改表。**
* 论文源码备份在仓库 `paper_drafts/v2026-09-11{,b,c,d,e}/`,每个目录有 NOTE.md 说明那一版改了什么。
* 题目:*Cross-Frequency Coupling as Token Content: Encoder-Dependent Benefits for Clinical EEG*。
* ICLR 2027:摘要 9-18,全文 9-25。

---

## 3. 项目讲什么

现有 EEG 基础模型的 token 只描述单个频带或一段宽带波形,**没有一种 token 表示两个频带之间的关系**。我们把相位-幅度耦合做成 token 内容:每个电极分成可学频带,逐 patch 测耦合统计量 Z,生成"交互 token";三条性质——相位参考不变、幅度模长保持(带 ε)、门控可关。然后用一个小模型(1.6–2.7M)在 12 个临床语料上对 25–69M 的基础模型。

**Zhizhe 的要求(2026-09-12 明确)**:要的是一个**三 seed 下全面能打的定稿模型**,不是 insight 类文章。"表现好的数据集不许掉,表现差的往上拉"。

---

## 4. 现在的主结果(三 seed,括号内为 seed 数)

| 语料 | 指标 | 最强 baseline | A: v1d192(无轴+duplex) | B: rot2(纯耦合+三轴) | C: 三轴 duplex |
|---|---|---|---|---|---|
| TUSZ | AUC-PR | FFCL 0.545±0.030(3) | 0.680±0.048(3) | 0.688(1) | 0.639±0.035(3) |
| CHBMIT | AUC-PR | TFM-pre 0.627±0.025(3) | 0.669±0.041(3) | 0.510(1) | 0.698±0.055(3) |
| TUEV | κ | REVE-pre 0.685±0.039(3) | 0.655±0.037(2) | 0.733±0.016(3) | 0.690±0.036(3) |
| IIIC | κ | REVE-pre 0.436±0.003(3) | 0.455±0.018(3) | -- | 0.479±0.009(3) |
| TUEP | AUROC | EEGPT-scr 0.786±0.021(3) | 0.799±0.005(3) | 0.799±0.029(3) | 0.810±0.004(3) |
| TUAB | BAcc | ST-T 0.820±0.004(3) | 0.818±0.013(3) | 0.799(1) | 0.813±0.006(3) |
| SLEEPEDF | κ | ContraWR 0.692±0.015(3) | 0.633±0.012(3) | 0.638±0.016(3) | 0.642±0.014(3) |
| ISRUC | κ | CBraMod-pre 0.754±0.008(3) | 0.697±0.015(3) | 0.710(1) | 0.702±0.009(3) |
| CAUEEG | BAcc | BIOT-scr 0.561±0.012(3) | 0.518±0.029(3) | -- | 0.525±0.015(3) |
| ADFD | BAcc | BIOT-scr 0.525±0.021(3) | 0.456±0.073(3) | 0.484±0.053(3) | 0.505±0.061(3) |
| TUAR | κ | CBraMod-pre 0.698±0.015(3) | 0.591±0.012(3) | 0.584±0.023(3) | 0.620±0.037(3) |
| SIENA | AUC-PR | REVE-pre 0.518±0.117(3) | 0.220±0.030(3) | 0.314±0.074(3) | 0.170±0.086(3) |

**读法**:只有 TUSZ 是稳赢(+0.135,远超方差)。CHB-MIT、IIIC、TUEP 的领先都在噪声里。Sleep-EDF / ISRUC / CAUEEG / ADFD 落后 0.04–0.07,TUAR 落后 0.11,Siena 落后 0.20+。
**没有任何一个现成模型在 12 格上全面领先**,这是当前的核心矛盾。

### tokenizer 模式的逐语料表现(所有历史实验求平均)

| 语料 | 纯耦合 | duplex | 纯波形 | hybrid/fused |
|---|---|---|---|---|
| TUSZ | 0.632(8) | 0.630(21) | 0.583(16) | 0.638(12) |
| CHBMIT | 0.513(6) | 0.612(20) | 0.451(14) | 0.663(11) |
| TUEV | 0.681(23) | 0.660(19) | 0.551(9) | 0.625(9) |
| IIIC | -- | 0.453(18) | 0.443(10) | -- |
| TUEP | 0.799(3) | 0.805(10) | 0.810(3) | -- |
| TUAB | 0.803(4) | 0.810(10) | 0.805(3) | -- |
| SLEEPEDF | 0.635(9) | 0.636(12) | 0.650(3) | -- |
| ISRUC | 0.697(10) | 0.695(11) | 0.701(3) | -- |
| CAUEEG | -- | 0.525(13) | 0.517(3) | -- |
| ADFD | 0.484(3) | 0.458(14) | 0.501(3) | -- |
| TUAR | 0.577(5) | 0.614(16) | 0.625(4) | -- |
| SIENA | 0.314(3) | 0.220(11) | 0.171(3) | -- |

纯耦合 vs duplex 逐语料是 **5:5**。纯耦合赢 TUEV/TUSZ/Siena/ADFD/ISRUC,duplex 赢 CHB-MIT/TUAR/TUEP/TUAB/Sleep-EDF。
**机制解释(Zhizhe 指出,已证实)**:纯耦合的 token `h_j = a_j ⊙ u_j/|u_j|` 里,`u_j` 只由**比 j 慢**的频带相位合成,频带自己的相位进不了自己的 token,所以**丢失频带内信息**;在需要波形形状的语料(CHB-MIT、TUAR)就崩。duplex 靠多加一行波形 token 补回来,代价是行数翻倍、在 TUEV 上被稀释。

---

## 5. 做过的实验(按线索)

### 5.1 模型结构线
| 家族 | 内容 | 结论 |
|---|---|---|
| `paclock_*` | 三轴编码器(时间/空间/频率三个注意力子层)的各种 tokenizer:raw / pac_interaction / hybrid / fused / duplex;频带数 8/16;stem 深浅 | duplex 与纯耦合各擅胜场;raw 最差;fused(单行融合)最差之一(0.55) |
| `cf2_v*` | 去掉频率子层的"无轴"家族:v0 不折叠、v1 折叠(频带并入空间注意力)、+耦合强度特征、d128/192/256 | 定稿曾选 **v1d192**(折叠+d192);补 seed 后 TUEV 0.655、CHB-MIT 0.669,不如预期 |
| `paclock_rot2` | **纯耦合 tokenizer + 三轴**,1.62M | TUEV **0.733±0.016** 是全项目最高,但 CHB-MIT 0.510 |
| `crofremo_n*`(在跑) | 按因子分析组合:纯耦合+rotation+频率注意力+16/24 带+d128/192+正则,5 臂 × 4 语料 | 待出 |
| `crofremo_s*`(待投) | **自耦合**:见 §6 | 待出 |

### 5.2 归因对照(论文核心)
同一编码器、同一配方,只换额外行装什么:
* 纯波形 vs duplex:TUEV 0.554→0.655、TUSZ 0.635→0.680、CHB-MIT 0.633→0.669、IIIC 0.434→0.455。
* **`pac_token_mode: own`**(每个频带只用自己的相位、不做跨频对齐)——用来分离"解析特征"和"跨频对齐":TUEV 0.590、TUSZ 0.611、IIIC 0.439、CHB-MIT 0.651。**结论:解析特征本身几乎不涨,跨频对齐才是增益来源**(TUSZ 上 own 甚至低于纯波形)。前端做过干预验证(扰动慢频带相位,own 模式下快频带 token 变化为 0)。
* 早期三轴的 12 语料耦合开关表、TUEV 逐类别表都在 `docs/FINDINGS.md`。

### 5.3 换头(把 tokenizer 装进别人的编码器)
* **加法式**(保留宿主 tokenizer,每电极加 8 行;对照是加 8 行波形):CBraMod 从零上 TUEV 0.564→0.584(波形)→**0.634**(耦合);TUEP、IIIC、CHB-MIT 同向,**TUSZ 负**(0.482→0.384)。
* **替换式**(换掉宿主的 patch 嵌入):LaBraM 从零 TUEV 0.372→**0.502**;REVE 从零 0.319→**0.556**;IIIC、CHB-MIT 小正。
* **挂官方预训练权重:失败**(CBraMod TUEV 0.645→0.623、TUAR 0.715→0.588)。原因:预训练编码器绑定自己的 token 分布。
* 这些实验用的 tokenizer **本来就是 `pac_interaction`(纯耦合)**,与 duplex 的选择无关。

### 5.4 预训练
* **我们的模型**:旧目标(掩码频带幅度+耦合重建,3800h 池,60k 步)3 赢 9 输;新目标(**CBraMod 式原始波形 patch 重建**,TUEG 切片+6 语料,150k 步)刚跑完并微调 12 语料——首批 Siena +0.12、ADFD 持平、TUEV 持平、Sleep-EDF −0.01、TUEP −0.04、**TUAR −0.09**,仍是"弱的拉上来、强的掉下去",**不满足 Zhizhe 的验收标准**。
* **微调配方闭环检验**(ptR = 预训练起点 + 从零配方):TUEV 0.636(从零 0.655)、Siena 0.422(从零 0.220)。**差在起点不在配方。**
* **宿主重预训练(TFM 设置)**:用 CBraMod 原始设定(40 epoch = 219k 步、batch 128、掩码 0.5)在干净 TUEG 切片上预训练两份——原生 tokenizer 已完成,微调 TUEV 0.500 / IIIC 0.324 / CHB-MIT 0.440 / TUSZ 0.491(发作两格帮、事件两格伤);**配我们 tokenizer 的那份在 b2 上跑到约 130k/219k,还要约一天**,出来才是同预算配对比较。小预算(12k 步)那对的配对差距很大:TUEV 0.493 → **0.654**。

### 5.5 正则(2026-09-12 新发现)
验证曲线显示模型在**第 2–13 个 epoch 就到顶然后一路掉**,是**过拟合不是欠训练**(我一度误判为欠训练,投了 60-epoch 实验,已撤)。
我们的权重衰减是 **1e-5**,CBraMod/LaBraM 是 **0.05**;而且**从未用过数据增强**(代码里有 `RandomAugment`,支持 flip/jitter/时间遮蔽/通道遮蔽/频率遮蔽)。
`reg` 家族四臂(r1 wd0.05 / r2 +dropout0.35 / r3 wd0.2 / r4 +增强)正在 7 个过拟合语料上跑 seed 0。

---

## 6. 正在进行的工作与下一步

### 在跑
| 位置 | 任务 | 说明 |
|---|---|---|
| amd | `ROT2_*` | 纯耦合 tokenizer 补 seed,12 语料 × 3 seed,**CHB-MIT 那三个 seed 是决定性的** |
| amd | `REG_*` / `REG4_*` | 正则四臂 × 7 语料 × seed 0 |
| amd | `DSN_*` | 因子分析设计的 5 臂 × 4 语料 |
| amd | `AXFPT_*` | 我们模型新预训练的 12 语料微调 |
| amd | `smoke_gpu`(排队)+ `slurm/launch_after_smoke.sh`(已挂) | **自耦合**的冒烟测试;通过后自动投 `SELF_*` 四臂 |
| torch | `seed_*`(2 跑 6 排,h200_public) | CBraMod 在 TUSZ/CHB-MIT 上的重 seed |
| b2 | `pre_cbr_crofremo_tuegc_r` | 宿主重预训练(我们的 tokenizer),约 130k/219k |

### 本机监视器(后台 shell,掉了要重挂)
* `~/.claude/jobs/18b2571f/tmp/wait_set.sh 60 <list>` —— 轮询 amd 的 `~/wave9.list`,结果落地就报;当前 251 个目标。
* `~/.claude/jobs/18b2571f/tmp/wait_b2pre.sh` —— 轮询 b2 的预训练完成。
* 其他工具:`tx.sh`(经 tmux 对 torch 发命令)、`b2_master.exp` / `mbp_master.exp`(重建连接)。

### 最重要的未决:定稿模型
**自耦合(coupling_self)** 是当前最有希望的设计,已实现未验证:
* 现在 `u_j = Σ_{i<j} α_ij e^{-i∠Z_ij} p̃_i`(严格下三角,排除对角线)。
* 改为 `Σ_{i≤j}`,把 **Z_jj**(频带自身相位与自身包络的耦合 = 波形形状/非正弦性,Cole & Voytek 2017)放进来。
* 好处:①补回纯耦合丢失的频带内信息(正是 CHB-MIT 崩的原因);②**行数不变**,没有 duplex 的稀释;③所有频带都满足相位参考不变性,论文里命题 2 的 `j>0` 特例可以取消。
* 代码:`frontend/triaxial.py` 的 `coupling_self` 开关(掩码 `diagonal=0`),`build.py` 已转发。四臂配置在 `configs/selfcoup/`(s1 纯耦合+自耦合、s2 duplex+自耦合、s3 16 带、s4 +正则)。
* **验收**:s1 在 CHB-MIT 上能否从 0.51 拉到 0.61 以上,同时 TUEV 保住 0.73 那一档。

### 决策流程(Zhizhe 要求:一次跑对,不要单 seed 拍板)
1. 自耦合四臂 + 因子设计五臂 + 正则四臂 → 在 4 个决策语料上选出**结构 + 正则**的组合;
2. 胜出组合 → 12 语料 × 3 seed 做最终主表(约 36 个任务、一天);
3. 同时给它重做两个归因臂(纯波形、own),约 8 个任务;
4. 更新 `scripts/gen_tables.py` 重生成所有表,改掉论文里"单 seed"的措辞,推 Overleaf。

### 历史教训(别重犯)
* **不要用单 seed 淘汰模型**:纯耦合曾因 CHB-MIT 单 seed 0.510 被淘汰,而那个语料的 seed 方差高达 ±0.055~±0.205。
* **不要只用一个语料做因子分析**:我曾只用 TUEV 得出"纯耦合最好",按 12 语料重算是 5:5。
* **投任务前先跑 GPU 冒烟测试**,并且要跑两步以上再计时(第一步包含内核编译,曾把 0.33 s/步误读成 25 s/步)。
* Zhizhe 的硬规矩:**没有明确指示不跑 3 seed**;该撤的任务立刻撤;统筹三个集群不要空转。
