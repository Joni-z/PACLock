# 进度(2026-08-06)

## A 组:已完成 ✅

**135/135 run**(9 数据集 × 5 模型 × 3 seeds),45/45 单元格达成 3 seeds。

产物:`results/PACLock_baseline_matrix_filled.xlsx`
- 41 个单元格已填(`mean±std`)
- 4 个按硬规则 3 留空(全是 CNN-Transformer)
- 新增 `FACED` sheet(工作簿原无,协议取自初版)
- 新增 `_A组填写记录` sheet:各 seed 明细、留空原因、离散警告
- `SEED-V` sheet 标为停用
- **原工作簿零改动**

### 各数据集最佳(A 组)

| 数据集 | 最好模型 | 主指标 |
|---|---|---|
| TUAB | ST-Transformer | AUROC 0.8972 |
| ISRUC | ContraWR | kappa 0.7520 |
| Sleep-EDF | ContraWR | kappa 0.6916 |
| BCI-IV-2a | SPaRCNet | BAcc 0.6440 |
| PhysioNet-MI | ST-Transformer | BAcc 0.5938 |
| TUSZ | FFCL | PR-AUC 0.5449 |
| CHB-MIT | FFCL | PR-AUC 0.5341 |
| TUEV | ST-Transformer | kappa 0.5006 |
| FACED | ST-Transformer | BAcc 0.3197 |

## 遗留问题(交给后续)

1. **CNN-Transformer 在小数据集上系统性塌陷** —— FACED/BCI-IV-2a/PhysioNet-MI
   上都恰好落在随机水平,4 个留空单元格全是它。它是唯一用
   `fft=sampling_rate`(其余 STFT 模型用 `token_size`)且参数量最大(3.2M)的模型。
2. **6 个单元格 seed 离散 20–43%** —— BIOT 配方(lr=1e-3 无 scheduler)固有的
   不稳定,改配方就不是复现。论文附录应给各 seed 值而非只报均值。
3. **A 组指标普遍高于发表值**(TUAB +0.02~0.045,方差极小)—— 推测因协议冻结
   CBraMod 预处理(带通+notch)而发表值出自 BIOT 预处理(仅重采样+q95)。
   **未验证**,建议做一次同模型同 split 的预处理对照。
4. **SPaRCNet 参数量** 0.99M vs xlsx 列 0.79M,用官方超参无法复现该值,已记录。

## 下一步:B / C / D 组

* **B 组** 官方预训练权重 —— 必须用各自 repo 的预处理 + 归一化 + finetune recipe,
  需要各自独立的 loader,不能走我们的 pipeline。硬规则 1 的复现门只适用于本组。
* **C 组** 同 pipeline from-scratch —— 对称超参、参数量对齐。
* **D 组** PACLock from-scratch 完整体 —— 模型代码已 vendored 并通过 31/31 等价性测试。

## 集群规范

所有计算走 Slurm;登录节点只用 `squeue`/`sbatch`/`sacct` 和代码同步。
详见 README「集群使用规则」。
