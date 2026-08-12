# 作废:batch size 过大导致训练不足

这三个数据集用 BIOT 的 batch_size=512 跑,但它们的训练集只有 2k-10k 窗口,
每轮仅 4-20 次参数更新。实测(同 seed、仅改 batch):

  BCI-IV-2a  sparcnet  512 -> 0.4996   64 -> 0.6478   (+0.148)
  PhysioNet  sparcnet  512 -> 0.5733   64 -> 0.6000   (+0.027)
  FACED      contrawr  512 -> 0.1111   64 -> 0.1758   (FACED 在 512 下完全没学)

已改为按数据集规模选 batch(保证每轮 >=100 步)后重跑。作废于 2026-08-06。
