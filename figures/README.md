# figures/ — 汇报与论文用示意图(TikZ 源码)

每个 `.tex` 都是独立的 `standalone` 文档,只依赖标准 TikZ 库 + amsmath/amssymb。

## 编译

本仓库所在的两个集群都没有 LaTeX。两种方式:

```bash
pdflatex tokenizer.tex        # 本地有 TeX Live 时
```

或把单个 `.tex` 文件整个粘进 **Overleaf**(新建空白项目替换 main.tex 即可),
直接得到裁好边的 PDF,拖进 slides 就能用。

> 注意:这些源码在提交时**没有经过编译验证**(两边集群都装不了 LaTeX)。
> TikZ 已按保守写法书写,但首次编译如遇报错,多半是节点间距/重叠这类
> 视觉问题而非语法问题——微调 `above right=.. and ..` 里的距离即可。

## 文件

| 文件 | 内容 | 用在哪 |
|---|---|---|
| `tokenizer.tex` | 完整 tokenizer 结构图:sinc 滤波器组 → 双路(raw 保相位 / Hilbert→耦合估计→规范不变对齐→rotation 交互)→ 可学门 → 混合 token 网格 (C×2n_b×P) → 三轴 encoder。公式与机制注记齐全 | 方法页主图 |
| `modes.tex` | 三种模式对比条:raw / constitutive PAC / hybrid,各自的胜负语料与数字 | 动机页 / 对比页 |

## 图中数字的出处

图内标注的数字(TUEV 0.733、TUSZ 0.671 等)均来自
`results/PACLock_baseline_matrix_filled.xlsx` 的 `_消融` sheet,与
`docs/STATUS.md` 的表一致;hybrid 的"validation running"以
`docs/STATUS.md` 当日状态为准。改数字请两处同步。
