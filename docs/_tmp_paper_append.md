

---

# 2026-08-21 更新:名单数据侧齐备,锁定条件客观化

08-20 提案的 12 个:TUAB、TUEV、CHB-MIT、TUSZ、ISRUC、Sleep-EDF、TUEP、
TUAR、ADFD、APAVA、Mumtaz2016、EEGMat(MentalArithmetic)。

* **数据全部在手**:TUEP/ADFD/APAVA 已处理;Mumtaz(810M,figshare API)与
  EEGMat(158.8M,首次下载截断已修复)loader 落地
  (`preprocessing/{mumtaz,eegmat}.py`,commit 4ca79f3),预处理任务 381007。
* **协议对齐 CBraMod、划分不抄**:两语料按 CBraMod 预处理(5 s 窗、19 通道
  10-20、200 Hz),但 CBraMod 的 Mumtaz 划分按文件排序切,同一被试的 EC/EO
  会跨 train/test(泄漏);我们用被试不相交排序划分(PROTOCOLS 标准),
  论文里这一句差异必须写明,数字对比时要预期我们的口径更严、数字偏低。
* **锁定条件**:7 个无实测语料的 duplex scratch 单 seed
  (DUP_slate_long/short)全部回数、无输十几个点的格子。崩了从候补池换:
  BCI/PMI(预训练翻盘才进)、TUSL、SHU-MI、SEED-VIG。
* 新语料的 A 组 baseline(硬规则约束)是进正表前的独立阶段,费用未排。
