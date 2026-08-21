

---

# 第四部分:结构收敛波(2026-08-19 → 08-21)

目标(Zhizhe,08-19):用单 seed 消融确定最有前景做预训练的**结构**;12 个
下游数据集不允许输十几个点。四波实验,全部 `configs/_diag/`、全部单 seed、
全部 mi2104x 四卡打包。

## 4.1 波一:fused(同行内融合)—— 行分离是 TUEV 的硬需求

`fused` 在 raw 投影里加零初始化 β 的 PAC 混合(`blend`)或内容门(`gated`),
网格尺寸不变。四语料判决:

| 语料 | fuse | fusegate | raw (3) | 此前家族最好 |
|---|---|---|---|---|
| CHB-MIT | 0.7194 | **0.7441** | 0.6672 | hyb_gate 0.7513 |
| TUEV | 0.5493 | 0.5879 | 0.5359 | hybrid 0.6951 |
| TUSZ | 0.5643 | **0.6950** | 0.6710 | hybrid 0.6299 |
| BCI-IV-2a | 0.4483 | 0.4082 | 0.4192 | hybrid 0.3912 |

结论:癫痫语料要 fusegate(行内门控融合),但 **TUEV 需要行分离的交互
token**(fused 比 hybrid 低 0.11 —— 把交互混进 raw 行会抹掉事件形态判别所需
的那部分)。没有一个融合模式全局最优。

## 4.2 波二:duplex —— 无短板的网格

`duplex` = nb 行融合混合(β 零初始化)+ nb 行门控交互(α 初始 1),初始化
逐位等于 hybrid+gate(`scripts/verify_duplex.py`,26/26)。

| 语料 | duplex | 最强外部 baseline | delta |
|---|---|---|---|
| TUEV | 0.7094 | 0.6519 (TFM) | +0.058 |
| TUSZ | 0.6328 | 0.5449 (FFCL) | +0.088 |
| CHB-MIT | 0.7130 | 0.6269 (TFM) | +0.086 |

不是任何单格的冠军(fusegate 在 TUSZ/CHB 各高 0.03~0.06),但**唯一在三个
等级一语料上同时超 baseline** 的网格。骨干选它:预训练骨干要的是无短板,
单格冠军可以做微调期条件项。

## 4.3 波三:H1–H4 单因子 —— 每语料一个约束,不叠加

假设来源:A 组小模型(SPaRCNet 等)在 MI/情绪上赢我们的共性是深 conv stem +
早期通道混合;我们是全场唯一的单线性投影入口。

| 假设 | 实现 | 帮助 | 伤害 | 判决 |
|---|---|---|---|---|
| H1 深 stem | 3 层 conv/GELU 残差精炼 raw 投影,末层零初始化(初始逐位=线性) | TUEV +0.020、FACED +0.034、PMI +0.015 | **CHB −0.023、TUSZ −0.044**(旗舰波实测) | 语料条件项;癫痫路径禁用 |
| H2 学习式蒙太奇 | 语料私有 W=I+Δ 通道混合,Δ 零初始化,骨干外 | PMI +0.049 | 其余无效 | 仅 PMI |
| H3 nb16 | 频带分辨率翻倍(个体 mu 峰变异) | BCI +0.031 → 0.5583;TUEV(pac)安全 0.7223 | 与其他组件组合互毁(见旗舰) | 语料条件项 |
| H4 flatten 头 | 只池化频带轴,保留 C×P×D(线索锁定轨迹) | FACED +0.072 → 0.2344 | 伤 BCI/PMI | 语料条件项 |

组合波(c_*):**不叠加**。faced flat+stem 0.2432 ≈ 两者较好者;bci
nb16+stem/nb16+stem+mont 均不超 nb16 单用;pmi nb16+mont 不超 mont 单用。
每个语料只有一个 binding constraint,解掉之后其余组件是噪声或负担。

小语料调参波(过拟合方向:aug/wd/patience):PMI 0.4840→0.5048、FACED
0.1622→0.1801、BCI 无改善。**调参不是出路**——十几个点的缺口是结构性的。

## 4.4 波四:旗舰证伪 —— 零初始化保底不等于可叠加

旗舰 = duplex + nb16 + stem + montage + gated_meanspatial,每个组件单独
有效或"零初始化保底"。预注册预期 BCI 0.53–0.56。实测:

| 臂 | 实测 | 对照 | 差 |
|---|---|---|---|
| BCI 旗舰全家桶 | **0.3661** | nb16+spatial 0.5583 | **−0.19** |
| BCI nb16+gated_ms | 0.4174 | nb16+spatial 0.5583 | −0.14 |
| TUSZ fusegate+gated_ms | 0.6595 | fusegate+mean 0.6950 | −0.036 |
| CHB fusegate+stem | 0.7210 | fusegate 0.7441 | −0.023 |
| TUSZ fusegate+stem | 0.6506 | fusegate 0.6950 | −0.044 |

三个教训:

1. **零初始化只保第 0 步。**γ=0 的 gated_meanspatial 在 init 逐位等于 mean
   头(verify 过),但训练中门会打开并且打开得有害 —— "最坏情况=已证安全形态"
   只对初始化成立,不对训练终点成立。
2. **深 stem 不是全局无害**,此前"帮助或不伤"的结论来自没测癫痫语料。
3. **组件相互作用在小数据上是破坏性的**(BCI 全家桶比最差单组件还低)。

## 4.5 判决:骨干 vs 微调期条件项

* **骨干(预训练、迁移)**:duplex + rotation + nb8 + 线性 tokenizer +
  三轴 encoder。
* **微调期条件项(不迁移,按任务族)**:头(mean / spatial / flatten)、
  stem、montage、nb16(注意 nb 改变骨干网格,实际不可微调期切换 ——
  它要么进骨干要么放弃;当前证据下放弃,BCI 的 +0.031 记为机会成本)。
* 头按语料选的先例:CBraMod 仓库为每个下游数据集单独一个
  `model_for_*.py`。预训练交付物是骨干,头本来就是微调期部件。

线 A(patch200 三臂)收尾:预训练目标与 PAC tokenizer 配对 —— pac 预训练帮
TUSZ/CHB,raw 预训练反而伤;tokenizer 迁移的净贡献为正。线 B(CBraMod 移植)
收尾:移植臂建立了"前端可移植"的证据,第四臂(CBraMod 预训练 encoder +
我们的 tokenizer)仍缺,记为论文期待办。
