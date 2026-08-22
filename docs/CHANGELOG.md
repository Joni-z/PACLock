# 协议实现记录

记录实现过程中遇到的、与冻结协议相关的判断。**冻结参数一个都没有改动**;
这里记的是协议文字未覆盖到的边界情况,以及依据文献通行做法所做的选择。

---

## 2026-08-04 — TUAB:按类切分导致的 train↔val 受试者重叠

**现象.** `preprocessing/tuab.py` 的泄漏守卫报错:3 个受试者
(`aaaaaoiz`、`aaaaaopm`、`aaaaaoru`)同时出现在 train 和 val。

**成因.** 协议规定「normal 与 abnormal **分别**按 subject ID 排序后前 80%/后 20%」。
实测 TUAB 官方 train 里有 **54 个受试者同时拥有 normal 和 abnormal 记录**
(同一病人不同次就诊结论不同,临床常见)。两类各自独立切分时,这些人可能
normal 记录落 train、abnormal 记录落 val。54 人中有 3 人跨越了 80/20 切分点。

**文献做法.** 两个官方实现都是按类独立切分,因此都有同样的重叠:

* BIOT `datasets/TUAB/process.py` — 对 abnormal / normal 各自
  `np.random.shuffle` 后取前 80%
* CBraMod `preprocessing/preprocessing_tuab.py` — 同样按类各自 80/20

我们要对齐的所有已发表 TUAB 数字都是这样产生的。

**决定.** **保持协议不变,采用通行做法。** A 组存在的意义就是校准 pipeline
是否能对上已发表值,若在此处改用更严格的划分,反而会引入一个与文献不可比的
差异,把「pipeline 是否正确」和「划分是否不同」两个因素混在一起。

**处理方式.**

* 重叠**不修复**,但**完整记录**在 `processed/tuab/manifest.json` 的
  `qc.subject_overlap`(含受试者 ID)与 `qc.n_overlapping_subjects`
* `Manifest.check_disjoint(strict=False)` 仅对 TUAB 使用;其余数据集保持
  致命错误,泄漏就是 bug
* 预处理时打印警告

**影响范围.** 仅 train↔val。官方 eval(=test)与 train **零受试者重叠**
(已实测确认),所以报告的 test 指标不受影响;重叠只可能让 val 略微乐观,
从而轻微影响 best-checkpoint 的选择。

**论文中需注明**:TUAB 的 train/val 划分沿用 BIOT/CBraMod 的按类划分惯例,
其中 3 名受试者跨 train/val;test 为官方 eval,受试者与 train 不重叠。

---

## 2026-08-04 — A 组:改用 BIOT 官方实现,参数量对齐

**现象.** 自己重写的 A 组五个模型能训练,但参数量与 xlsx 列出的值对不上
(如 FFCL 0.70M vs 2.4M)。A 组的唯一作用是校准 pipeline 能否复现已发表值,
架构不对则校准无意义,且 B/C 组结论会继承这个错误。

**处理.** 删除自己的重写,改为 **vendor BIOT 官方实现**
(`ycq091044/BIOT`,`model/`)到 `paclock_bench/models/baselines/biot_official/`,
本地只保留一层薄适配。重写版产出的结果已移入 `runs_invalidated/` 作废。

**重写版的两处结构性错误**(说明为何必须换掉,不只是超参不同):

* ContraWR / CNN-Transformer / FFCL 官方是在 **STFT 频谱**上跑 **2D** ResBlock,
  我写成了在原始波形上跑 1D 卷积
* FFCL 的 LSTM 输入是 `shorten()` 的交错重排,我写成了简单的等间隔下采样

**超参来源.** 取自作者的训练脚本 `run_binary_supervised.py` /
`run_multiclass_supervised.py`,**不是**模型文件的构造函数默认值——两者不一致,
而脚本才是产出已发表数字的那个:

| 项 | 值 | 备注 |
|---|---|---|
| `token_size` | 200 | 200 Hz 语料;Sleep-EDF 是 100 Hz,配置里必须覆盖 |
| `hop_length` | 100 | ⇒ `steps = hop_length // 5 = 20` |
| ContraWR / FFCL | `fft = token_size` | |
| CNN-Transformer | `fft = sampling_rate` | 与上面两个**不同**,脚本确实如此 |
| **ST-Transformer** | **`depth = 4`** | 模型文件默认是 3;这一项就让它从 2.64M 变到 3.43M |
| SPaRCNet | `block_layers=4, growth_rate=16, bn_size=16` | 与文件默认相同 |

**参考配置.** xlsx 每个模型只给一个参数量,**是按 TUEV 的 5 秒窗口
(16ch × 1000 @ 200Hz)测的**。FFCL 是长度相关的,正好钉死了这一点:
T=1000 时实测 2.416M vs 表列 2.40M;T=2000 时是 2.465M。

**结果.** 5 个中 4 个吻合到 ≤2.1%:ContraWR 1.7%、CNN-Transformer 1.4%、
FFCL 0.7%、ST-Transformer 2.1%。

**遗留:SPaRCNet 对不上,不予修正.** 用作者脚本的原始超参实测 0.992M(TUEV)/
1.142M(TUAB),xlsx 列 0.79M,差 25.5%。已试遍窗口长度与
`block_layers`/`growth_rate`/`bn_size` 的各种组合,**没有任何一组能复现 0.79M**。
判定为已发表数字自身的不一致,**保留官方架构不动**。

理由:为了让参数量对上而去编造超参,等于把校准基线换成一个与已发表模型
不同的模型 —— 这正是这项检查要防的事。真正有约束力的校准门是**指标比对**
(A 组能否对上 TFM-Tokenizer 的已发表值),不是参数量;参数量只是架构是否正确的
廉价代理。已记入 `light_supervised.py` 的 `KNOWN_PARAM_DISCREPANCIES`,
`tests/test_baseline_params.py` 会持续打印。

---

## 2026-08-04 — A 组:改用 BIOT 官方训练配方

**现象.** 首轮 TUEV 结果里 SPaRCNet 只有 kappa 0.276,发表值 0.423,差 0.148;
同时 CNN-Transformer 与 FFCL 虽然"通过"复现门,但靠的是 ±0.07–0.08 的大方差,
说服力很弱。

**根因.** 我自己定的训练配方与 BIOT 的**每一项都不同**:

| 项 | BIOT 官方 | 首轮用的 |
|---|---|---|
| 优化器 | Adam | AdamW |
| **学习率** | **1e-3** | 1e-4(低 10 倍) |
| scheduler | 无 | cosine |
| batch size | 512 | 64 |
| epochs | 100 | 30 |
| grad clip | 无 | 1.0 |

学习率低 10 倍、epoch 少 3 倍,足以让 SPaRCNet 欠拟合。

**处理.** 全部改为 BIOT 的配方(`run_binary_supervised.py` /
`run_multiclass_supervised.py`)。训练循环新增 `optimizer` 配置项,因为 AdamW 的
解耦权重衰减与 Adam 在 wd=1e-5 下并不等价。首轮结果移入 `runs_wrong_recipe/`。

**唯一保留的偏离:epoch 上限.** BIOT 跑 100 epoch,在 TUAB/TUSZ/CHB-MIT
(30 万+ 窗口)上远超模型峰值,也超出节点时限。实际停点由主指标的 early stopping
决定,上限只用来约束作业时长。

**顺带修正:STFT 帧数规则.** ContraWR / CNN-Transformer / FFCL 把频谱池化 4 次
(共 256×)并假定塌缩到 1×1。BIOT 的默认 hop 在 5–10 秒窗口下没问题,但
Sleep-EDF 和 ISRUC 是 **30 秒 epoch**,默认会产生约 601 帧,末端剩 3 帧导致
分类器矩阵乘失败。现按数据自动放宽 hop 使帧数 ≤256,**5–10 秒的数据集保持
BIOT 原始设置不变**。`tests/test_all_configs_forward.py` 覆盖全部 40 个组合。

---

## 2026-08-04 — 观察:官方配方下 val 曲线剧烈震荡(不修正)

跑 A 组时发现 TUEV 上 CNN-Transformer 的验证曲线逐轮大幅跳动,例如 seed 1 的
kappa 序列:`0.242 → 0.431 → 0.416 → 0.058 → 0.048 → 0.346 → ...`,seed 0 同样
在 0.05–0.41 之间反复。两个 seed 都如此,不是个别 seed 崩溃。

**成因**:BIOT 的配方是 lr=1e-3、Adam、**无 scheduler**。在这个学习率下不收敛到
平稳点是预期行为。

**不做修正.** 加 scheduler 或降 lr 都会偏离官方配方,而 A 组的全部意义是复现官方
数字。BIOT 自己也是取验证集最优 checkpoint,我们做法一致。

**两个后果,报告时必须说明**:

1. **best-checkpoint 是在噪声曲线上选的**,存在向上的选择偏差。峰值
   (seed1 0.431 / seed0 0.410)确实落在发表值 0.3815 附近,但这部分是
   "在震荡序列里取最大值"的结果,不能解读为模型稳定达到该水平。
2. **seed 间标准差偏大**(首轮观察到 ±0.07–0.08)。复现门用 mean±2std 判定,
   区间被噪声撑宽后很容易"通过"。**因此判定为 reproduced 的说服力弱于
   std 小的行**,附录里应同时给出各 seed 的峰值与曲线,而不是只报均值。

早停 patience 也因此带有随机性(seed 1 在 epoch 18 停,峰值在 epoch 2)。

---

## 2026-08-05 — 矩阵变更:SEED-V 替换为 FACED

**决定.** 用户改用 FACED 替换 SEED-V。SEED-V 的配置与预处理脚本保留但标记
`DEPRECATED`,不再生成实验配置。

**FACED 协议来源.** xlsx **初版**(2026-08-04 之前)本就包含完整的 FACED sheet,
后来被改版移除。本次直接采用该版本的冻结协议,未做任何重新推导。

**两者差异很大,是重写而非改名:**

| 项 | FACED | SEED-V |
|---|---|---|
| 类别数 | **9** | 5 |
| 通道数 | 32 | 62 |
| 窗口 | 30 s trial 切 3×**10 s** | **1 s** |
| 样本形状 | 32×2000 | 62×200 |
| 划分 | **subject-disjoint**(S000–079/080–099/100–122) | trial-disjoint(同一人跨 split) |
| 受试者 | 123 | 16 |

FACED 是 subject-disjoint,因此 `Manifest.check_disjoint()` **要调用**;
SEED-V 当初是刻意不调用的。

**一个必须注意的执行要求.** 协议冻结项写明「使用**官方发布的 pre-processed**
`.pkl`」,官方预处理已包含 0.05–47 Hz 滤波、坏导插值、**ICA 去眼动**、
common-average 重参考、每视频取末 30 s。因此 `preprocessing/faced.py`
**只做 250→200 Hz 重采样和切窗,不再做任何滤波** —— 二次滤波会悄悄改变所有模型
看到的输入,而官方的 ICA 步骤我们也无法 bit-identical 复刻。

用户手上的百度网盘 zip 有 52 GB,而官方 pre-processed 版本约 3–4 GB,
**高度怀疑那是 raw 版本**,已提示确认。若确为 raw,不能直接使用。

**传输方案.** 实测本机上行仅约 0.4 MB/s,52 GB 需约 37 小时,不可行。
集群可直连 Synapse(HTTP 200),改为在集群侧用 `synapseclient` 下载官方
pre-processed 版本,走 Slurm 作业,凭据用 Personal Access Token(可限权可撤销),
不使用百度 BDUSS(等同整个账号登录态)。

---

## 2026-08-05 — ISRUC 预处理连挂两次:通道命名与二维标签

### 第一次:`--jobs` 参数缺失

`slurm/preprocess.slurm` 统一给所有数据集传 `--jobs`,但只有用多进程的
TUAB/TUEV/TUSZ/CHB-MIT 定义了该参数。

**为什么拖到现在才暴露**:Sleep-EDF 和 PhysioNet-MI 当初是在登录节点直接跑的
(即那次违规),没走 Slurm 脚本,因此没经过这条路径。**绕过标准路径就等于绕过了
它的检验** —— 这是那次违规的一个副作用。

**处理**:9 个预处理脚本接口统一,全部接受 `--jobs`;单进程的注明「仅为接口一致」。
顺带把 ISRUC 真正并行化(100 个 subject 各自读 EDF + 滤波),并按 subject 号
重排序,保证 Pool 乱序返回不影响输出的确定性。

### 第二次:通道命名两套 + 二维标签

**通道命名(数据现实,非 bug)**:ISRUC 受试者分两批,一批用耳参考拼写
`F3-A2`,另一批用乳突拼写 `F3-M2`。**A1/A2 与 M1/M2 是同一组参考电极**
(左右耳/乳突),因此两种拼写指的是同一导联,协议的通道表对两者都成立。
`pick_channels` 现在把 `A1≡M1`、`A2≡M2` 归一后再匹配。

**为什么烟测没抓到**:`tests/test_readers.py` 只测了 subject 2,而它恰好是 A 拼写。
测试已改为**遍历到覆盖两种拼写各一个**为止 —— 单样本烟测在"数据集内部存在
变体"时是不够的。

**二维标签(我的 bug)**:ISRUC 标签形状是 `(n_seq, 20)`,`common.save_split`
里的 `np.bincount(y)` 只接受一维,抛 `object too deep for desired array`。
已改为 `y.ravel()`。这条路径其他数据集都是一维标签,所以只有 ISRUC 触发。

---

## 2026-08-06 — B 组:五个模型的实现与两个不可用项

### 已完整复现(严格照各自原仓库)

| 模型 | 权重 | 预处理来源 | 状态 |
|---|---|---|---|
| BIOT | `EEG-PREST-16-channels.ckpt`,零 missing/unexpected | `datasets/TUAB/process.py` | ✅ |
| CBraMod | HuggingFace `pretrained_weights.pth` | 其 `preprocessing_tuab.py` **即本工作簿的冻结协议** | ✅ |
| LaBraM | `labram-base.pth`,224 张量(204 block) | `dataset_maker/make_TUAB.py` | ✅ |

四套预处理互不相同,每一项都可追溯到上游具体文件:

| | BIOT | LaBraM | TFM | CBraMod/我们 |
|---|---|---|---|---|
| 带通 | 无 | 0.1–75 | 0.1–75 | 0.3–75 |
| notch | 无 | 50 Hz | 50 Hz | 60 Hz |
| 通道 | 16 双极 | **23 单极 -REF** | 16 双极 | 16 双极 |
| 优化器 | Adam 1e-3 | AdamW + layer decay 0.65 | AdamW 1e-3 | AdamW + multi_lr |
| 归一化 | q95(loader) | ÷100(loader) | q95(loader) | ÷100(预处理) |

### 关键验证:BIOT 精确命中发表值

TUAB AUROC:预训练 **0.8730 ± 0.0026** vs BIOT 自报 **0.8730**(差 0.0000);
from-scratch 0.8694 vs 0.8691(差 0.0003)。

**这条结果同时解答了 A 组的遗留疑问。** A 组所有可比单元格都比发表值高
0.02–0.045,当时推测源于预处理差异但未验证。现在用 BIOT 自己的预处理跑
同一套训练/评测代码,能精确复现发表值 —— 说明训练循环、best-checkpoint 选择、
指标实现**没有系统性问题**,A 组的偏高确实来自预处理(CBraMod 协议的带通+notch
vs BIOT 的不滤波)。B 组本身充当了原计划的预处理对照实验。

### 两个不可用项

**TFM-Tokenizer:上游代码与发布权重版本不一致。**
`get_tfm_tokenizer_2x2x8`(超参 8192/64,与 finetune 脚本默认值一致)构造出的
`TFM_VQVAE2_deep` 期望 `decoder.0/decoder.2`,而发布权重含
`freq_pos_embed`、`temporal_pos_embed`、`temporal_pos_embed_decoding`、
单层 `decoder` —— 全仓库 grep 不到任何包含 `freq_pos_embed` 的代码。
HuggingFace 上是同一批文件。判定为上游自身问题,非本仓库适配错误。
预处理、适配层、配置均已就绪,上游修复后可直接跑。

**EEGPT:权重不可自动获取且形状不匹配。**
权重在 figshare 需浏览器交互;更根本的是它要求 **58 通道 / 256 Hz / 4 秒**,
而 TUAB 的单极通道总共只有 23 个,凑不齐 58。判定为在本工作簿的数据上不可行。

### 修正的错误

* **TFM 采样率**:曾照抄 `dataset_configs.yaml` 的 `sampling_rate: 256`,但那是
  语料原始采样率;模型输入是 **200**(`--resampling_rate` 默认值、
  `get_stft_torch` 的 `n_fft=resampling_rate`、tokenizer 的 `n_freq=100`
  三处佐证)。按 256 会产生 2560 点窗口,预训练 tokenizer 无法消费。
* **变体冲突**:`collect_results` 以 `(dataset, model)` 为键,导致预训练与
  from-scratch 两行同名 `biot`。改用 run name 的变体后缀。
* **两处 pyhealth 依赖**:LaBraM 与 TFM 的 `utils.py` 均在模块顶层为 metrics
  引入 pyhealth。分别用 AST 解析常量、内联函数解决,不拖入无关重依赖。

### LaBraM 在 CHB-MIT 上不可行(montage 限制)

LaBraM 的位置编码按电极身份索引,要求 23 个**单极** `-REF` 通道
(`make_TUAB.py` 的 `chOrder_standard`)。CHB-MIT 的原始 EDF **只提供双极导联**
(`FP1-F7`、`F7-T7`…),单极信号无法从中还原 —— 双极是两个单极之差,
一组差值不足以反解出各自的绝对电位。

因此 B 组的实际可行矩阵是:

|  | TUAB | TUEV | CHB-MIT |
|---|---|---|---|
| BIOT | ✅ | ✅ | ✅(其 CHBMITLoader 用同一 16 双极 montage) |
| CBraMod | ✅ | ✅ | ✅ |
| LaBraM | ✅ | ✅ | ⛔ 需单极,CHB-MIT 无 |
| TFM | ⛔ 上游权重与代码版本不一致 | ⛔ | ⛔ |
| EEGPT | ⛔ 需 58 通道 | ⛔ | ⛔ |

合计 8 个 B 组单元格(3+3+2),每个 3 seeds,另有同样数量的 from-scratch 对照。

---

# 2026-08-06 → 2026-08-18

上一条记录停在 A 组完成。这一段覆盖 B/C/D 组、预训练、以及两处进入模型的架构
改动。逐条按「做了什么 / 为什么 / 证据」记录,被否掉的同样记录,因为重走一遍
的成本比读一段文字高得多。

## 架构改动(两处,均零新增参数)

### `interaction_mode: rotation`(新增,未设为默认)

`token = a_j · aligned_phase_j / |aligned_phase_j|`。

**为什么**:`product` 模式下 `|aligned_phase|` 在各 patch 间的变异系数约 0.62,
而它是耦合估计量的副产物、不是特征;在 BCI-IV-2a 上四个类的耦合统计量完全一致
(mean |Z| 0.0026–0.0031,preferred phase 一致性处于 surrogate 零水平),
于是频带功率被乘上一个不含标签信息的每-patch 随机增益。PAC 的信息在
`aligned_phase` 的**方向**里,不在模长里。

**证据**:TUSZ +0.100(n=1)、TUEV +0.025(n=3)、PhysioNet-MI +0.024(n=1)、
BCI +0.012(n=3);TUAR −0.021(n=1)、Sleep-EDF −0.001(n=1)。

**验证**(`scripts/verify_rotation.py`):`product`/`concat` 与改动前逐位一致;
`|h| = |a|` 误差 2.4e-7;相位扭转仍改变 token;**规范不变性首次实测通过**
(`p_i → e^{iδ_i}p_i, Z_ij → e^{iδ_i}Z_ij` 下 1.. 频带不动),`product` 与
`rotation` 都成立 —— 这条主张 docstring 里写了很久但从未被测过。

### `head: spatial`(新增,未设为默认)+ 一个会让实验作废的 bug 修复

保留电极身份,只池化频带与时间轴。

**为什么**:`mean`/`band`/`attn` 都把电极轴平均掉,而运动想象的判别量是对侧
感觉运动区的 mu/beta 去同步 —— 一个空间对比。旁证:BCI 上最好的 PACLock 变体
一直是 `raw_headattn`(用学出来的权重池化而非均匀平均)。

**证据**:PhysioNet-MI +0.084、BCI +0.056(均 n=1);FACED +0.004(无效)。
与 rotation 叠加:BCI 0.4344,高于任一单项。

**Bug**:投影层原本在 `forward()` 里懒构建,而 optimizer 在此之前已经捕获
`model.parameters()` —— 该层会带着随机初始化训练全程,并被记录为「试过没用」。
改为按 `cfg['n_channels']` 在 `__init__` 中构建;`scripts/verify_head.py` 断言
它在首次 forward 前已在 `parameters()` 中、能收到梯度、能被优化器更新。

## 等级三诊断:排除掉的解释

`PhysioNet-MI / FACED / BCI-IV-2a` 落后 baseline 0.2–0.4。以下每条都曾被当作
原因,每条都被实测否掉:

* **「PAC 在 band-power 任务上退化到随机」—— 混淆,已在 `FINDINGS.md` 撤回。**
  0.259 那个数字来自 `processed_pac` 预处理(0.5 Hz 高通、无 notch);同一模型
  配置在 `processed/` 上是 0.3588。约 0.10 属于预处理,只有约 0.06 属于 tokenizer。
* **样本量** —— TUEV 分层降采样到 2160 窗口(BCI 规模)仍得 kappa 0.6523,
  仍领先 SPaRCNet +0.161;32 倍数据缩减只掉 0.055。为此给
  `build_dataloaders` 加了 `train_subsample` / `train_subsample_seed`
  (分层抽样,子集种子与模型种子解耦,`scripts/check_subsample.py` 验证)。
* **分类头参数爆炸** —— 九个语料 `n_params` 恒为 1.60–1.63 M。
* **recipe** —— 已跑过的 `patch200`/`lr3e5` 2×2 最多 +0.011。
* **幅度信息不可及** —— `scripts/probe_readability.py` 用岭回归从单个未训练
  token 回归其自身 log 频带功率:rotation 二次可读 R²=0.907、concat 线性可读
  0.853,都**高于**在 BCI 上真正获胜的 raw(0.469 / 0.031)。信息一直都在。
* **耦合显著性门控** —— 见下。

## 被否决的设计:`coupling_gate: significance`(已完全回退)

`w_ij = relu(1 − 1/λ_ij)`,`λ_ij = |Z_ij|² / E_null|Z_ij|²`,耦合不显著时让频带
退回自己的相位。`scripts/pac_null_calib.py` 用循环移位 surrogate(同时保留两侧
边缘分布与两侧自相关)测零水平,两条结论杀死了它:

1. **解析零水平错了 23–350 倍,而且方向是反的。** 我按包络自相关推出
   `L_eff = (bw_i + bw_j)·T`,预言有效自由度**远小于**名义 200;实测是 174–10121,
   多数**大于** 200。漏掉的主导项是:`A~_j(t)·exp(i φ_i(t))` 是**振荡积分**,
   相位以 f_i 高速旋转,抵消比随机游走快得多,所以有效自由度随源频率**上升**
   (45 Hz 源频带达 ~10⁴)。包络自相关真实存在但是二阶效应。
2. **显著性区分不开语料。** 用校准后的零水平,`frac(w>0)` 在 TUEV 是
   0.358–0.370、在 BCI 是 0.356–0.370,且 TUEV 内部类间平坦。运动想象里的跨频
   耦合统计上真实存在,只是**不具判别性** —— 门控恰好保留了没用的那部分耦合。
   **显著性 ≠ 判别性**,任何不看标签的耦合统计门控都区分不开这两种情况。

它还有一个真实缺陷,被自带的 `λ→∞` 等价性检查抓到:用 `w` 取代 `|Z|` 作混合
权重后,`λ→∞` 给出**均匀**权重而非幅度权重,所以它从来不是无条件 tokenizer
的严格推广。

## 预训练

* 预训练全部在 b2,checkpoint 传回 AMD 微调。正式 60k checkpoint 在
  `pretrain_runs_60k/`;`pretrain_runs/` 下同名的是 6000 步早期试跑,仅被
  `*_ft_*` 配方实验引用,**矩阵行未受影响**(一度误报为混淆,系
  `scripts/ckpt_steps.py` 用 `basename(dirname())` 剥掉父目录所致,已修)。
* **排除消融**:预训练池剔除 TUSZ/CHB-MIT 后,收益仍保留 66% / 69%。
* **raw vs pac 预训练**(同 60k、同 d256、每语料 `patch_len` 一致):raw 在三个
  等级三语料上分别 +0.049 / +0.036 / +0.005。TUSZ/CHB-MIT 的格子此前从未跑过。
* **已知配方问题**:预训练 `patch_len=200` 与 TUSZ/CHB-MIT/BCI 微调的
  `patch_len=50` 形状不符,tokenizer 权重加载时被跳过 —— 这些语料上的预训练
  只迁移了 encoder。

## 新增语料

* **TUAR**(伪迹事件形态,3 类):用来检验 PAC 的 TUEV 优势是否可泛化。
  **答案是否定的**:pac 0.5780 vs raw 0.6289。`bckg` 类 0 事件不是解析 bug ——
  41 个带 bckg 区间的标注文件全部没有配套 EDF(352 csv vs 310 edf);
  `chew` 只出现在 212 个受试者中的 25 个,验证集 0 窗口,故降为 3 类。
* **TUSL**:全语料仅 300 个事件(每类 100),在花训练算力前放弃。

## 工具

新增 `scripts/`:`verify_rotation.py`、`verify_head.py`、`pac_null_calib.py`、
`pac_noise_diag.py`、`pac_noise_diag2.py`、`probe_readability.py`、
`check_subsample.py`、`ckpt_steps.py`、`status_snapshot.py`。

`scripts/normalize_xlsx.py` 的 `range(1, 8)` 硬编码改为 `ws.max_column` ——
这个脚本唯一的职责就是统一格式,却漏掉了新增的两列 delta。

## 文档

* `docs/STATUS.md` 重写(上一版停在 08-06,内容已与事实不符)。
* `docs/FINDINGS.md` 追加 tokenizer 一节,并**撤回**其中一个已发布的错误
  结论(见上「混淆」条)。
* 删除 `docs/FACED_操作指南.md`:一次性操作手册,任务 8/12 完成;其中的协议
  事实(官方 `.pkl` 版本、split 边界 S000–079/080–099/100–122、每人 84 窗口、
  类别分布 3,3,3,3,4,3,3,3,3)已由 `docs/PROTOCOLS.md` §8 更完整地记录。


---

# 2026-08-19 → 08-21:结构收敛波与 12 数据集数据齐备

## 架构(全部有 verify 脚本,全部零初始化等价旧形态)

* `tokenizer_mode: fused`(行内融合,blend/gated 两式)与
  `tokenizer_mode: duplex`(nb 融合行 + nb 门控交互行,init ≡ hybrid+gate)。
* `raw_stem: deep`(3 层 conv 残差精炼,末层零初始化)、
  `learned_montage`(W=I+Δ,语料私有)、`n_bands: 16`、
  头 `meanspatial` / `gated_meanspatial` / `flatten`。
* `scripts/verify_duplex.py` 26/26;快照链 `_triaxial_prev*.py` 至 prev5。

## 判决(详见 FINDINGS.md 第四部分)

* 骨干定稿方向:**duplex + rotation + nb8**,唯一三个等级一语料全超
  baseline 的网格(TUEV 0.7094 / TUSZ 0.6328 / CHB 0.7130,均单 seed)。
* **旗舰全组件堆叠被证伪**:BCI 旗舰 0.3661(预期 0.53–0.56);
  gated_meanspatial 弃用(零初始化保底在训练中不成立);深 stem 伤癫痫
  (CHB −0.023 / TUSZ −0.044),降为语料条件项。
* 头按任务族微调期选择(CBraMod 每数据集一头的先例),不属于骨干。

## 数据 / 语料

* 新增 TUEP(136,525 训练窗)、ADFD、APAVA(Medformer npy 版,ADFD 伏特
  单位换算)loader 与配置;metrics 注册 tuar/tuep/adfd/apava。
* 新增 Mumtaz2016 与 EEGMat loader(commit 4ca79f3):CBraMod 协议、
  被试不相交划分;montage.py 增 `_MONO_19` 共享 19 电极表。
* EEGMat 首次下载截断(63M/158.8M),devel 分区续传修复。
* 12 数据集名单进 PAPER.md;FACED/TUSL 除名,BCI/PMI 转候补。

## 集群 / 运维

* devel 分区:0.5 h 墙钟上限;**无 pytorch/2.7.1 模块**(rocm/6.3.1 缺失),
  只能跑纯 CPU/网络任务(下载、解压)。
* 集群内节点间 ssh 需要 home 的 `authorized_keys` 含自己的公钥(08-21 已配;
  且只允许进有本人作业的节点;等价方式 `srun --jobid=<id> --overlap --pty bash`)。
* 登录节点 `/tmp` 有他人 `inspect.py` 遮蔽标准库(与 b2 同款坑)——脚本放
  repo 或私有目录,不放共享 `/tmp`。
* 冗余的 squeue 监视 shell 清理:每波提交只留一个监视循环。


---

# 2026-08-21 → 22:rung-1 预训练、75 个 baseline 单元格、十二语料判决

## 预训练

* 配对行掩码实现(commit 7f12853):掩码打在 nb 个物理频带上,同时遮住融合行
  与交互行;`scripts/verify_duplex_pretrain.py` 16/16,含 encoder 入口 hook
  的泄漏测试。旧模式的 aux loss 与快照逐位相同。
* `pt_duplex_base` 44058728 完成:60k 步,2:05:06,约 25 SU(h100)。
  **发现 h100 计费 12/小时,l40s 24/小时** —— 此前预训练都跑在贵一倍的卡上。
* `scripts/verify_duplex_transfer.py`:152 张量载入 / 5 排除 / 0 形状丢弃。
  发现 duplex 比 pac_interaction 多一个 patch_len 相关张量
  (`frontend.tokenizer`),线 A 的排除清单照抄会漏。
* **结果为负**:下游 4/5 变差。见 FINDINGS 第五部分。

## Baseline

* 五个新语料 × 15 个 baseline = 75 个单元格,全部落地,**共 18 node-hours**。
* 三个跨语料适配问题修复(EEGPT 通道表 / LaBraM positional / BIOT target_len),
  见 PROTOCOLS 附录 E。
* `scripts/add_slate_sheets.py` 建五张 sheet;`fill_xlsx.py` 注册五个数据集
  与两个 duplex 行标签。

## 数据

* Mumtaz2016 与 EEGMat 预处理完成 —— 十二语料的数据侧全部齐备。
* 干净 TUEG 切片(被试级排除)建好并保留,但**按决定不使用**:rung-1 用与
  CBraMod/LaBraM 同口径的原切片。

## 运维

* 停掉 `FL_long`(22h50m 只产一个已无意义的旗舰格)。教训:一个 24 小时的
  单目的任务 ≈ 五个语料全部 baseline 的成本。
* 预算口径厘清:AMD 是 node-hours(整节点),b2 是 SU(单卡);
  b2 单位价值约为 AMD 的 35 倍,baseline 不该搬去 b2。
