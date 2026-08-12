# FACED 数据处理与上传指南

在**已经下载好 FACED 压缩包的那台电脑**上操作。

---

## 为什么这么做

压缩包 52 GB,直接传到集群按实测上行速度(约 0.4 MB/s)需要 **37 小时**,不现实。

但预处理后的产物只有 **2.46 GB**(10,332 个窗口 × 32 通道 × 2000 采样点),
上传约 **1.8 小时**。所以:**在本地预处理,只上传结果**。

预处理是确定性的,而且 manifest 会记录每个源文件的 SHA256,所以在哪台机器上跑
不影响结果的可复现性。

---

## 前提检查

那台电脑需要:

```bash
# 1. 能连集群(如果不能,看文末「没有集群访问权限怎么办」)
ssh amd 'echo ok'

# 2. Python 依赖
python3 -c "import numpy, scipy, yaml; print('依赖 OK')"
# 缺的话:pip3 install numpy scipy pyyaml
```

磁盘至少留 **120 GB**(52 GB 压缩包 + 解压后的量 + 输出)。

---

## 第 1 步:拿代码

```bash
rsync -az amd:/work1/chenyuyou/yifanwang/Zhizhe/PACLock/ ~/PACLock/
cd ~/PACLock
```

只有代码,几 MB。

---

## 第 2 步:先查压缩包里是什么 ⚠️ 最关键

**不要先解压。** 先花几秒看清楚里面是什么:

```bash
bash scripts/faced_local_prep.sh inspect ~/Downloads/FACED.zip
#                                        ^^^^^^^^^^^^^^^^^^^^ 换成你的实际路径
```

输出会给出明确判断,分两种情况:

### 情况 A:找到 `.pkl` 文件 ✅

```
found 123 .pkl files
-> looks like the PRE-PROCESSED release. This is what the protocol wants.
```

这是协议要求的官方预处理版本,**继续第 3 步**。

### 情况 B:没有 `.pkl` ⛔

```
no .pkl found -> this is probably the RAW release.
-> STOP and check with Zhizhe.
```

**停下来,把 inspect 的完整输出发给 Zhizhe(助手),先别解压。**

原因:评测协议冻结的是**官方预处理版本**,它已经做过 0.05–47 Hz 滤波、坏导联插值、
**ICA 去眼动**、common-average 重参考。这些步骤我们自己复刻不可能做到逐位一致,
用 raw 数据等于偏离冻结协议,需要明确决策并记入变更日志。

**无论哪种情况,都把 inspect 的输出发一份过来。**

---

## 第 3 步:解压 + 预处理

仅在**情况 A** 时执行。

```bash
# 解压(52 GB,需要时间)
unzip ~/Downloads/FACED.zip -d ~/faced_raw

# 预处理:250 Hz -> 200 Hz,每个 30 秒 trial 切成 3 个 10 秒窗口
bash scripts/faced_local_prep.sh run ~/faced_raw ~/faced_out
```

### 正常输出长这样

```
  sub000 [train] (84, 32, 2000)
  sub001 [train] (84, 32, 2000)
  ...
  sub122 [test] (84, 32, 2000)
[train] (6720, 32, 2000) -> ~/faced_out  labels=[720, 720, 720, 720, 960, ...]
[val]   (1680, 32, 2000) -> ~/faced_out
[test]  (1932, 32, 2000) -> ~/faced_out
[manifest] ~/faced_out/manifest.json
```

### 自检要点

| 检查项 | 应该是 |
|---|---|
| 每个 subject 的窗口数 | **84**(28 视频 × 3 窗口) |
| 样本形状 | `32 × 2000` |
| subject 划分 | sub000–079 → train,sub080–099 → val,sub100–122 → test |
| 每人的类别分布 | 9,9,9,9,**12**,9,9,9,9(neutral 有 4 个视频,所以是 12) |
| 总大小 | 约 2.46 GB |

如果某个 subject 被 `EXCLUDED`,脚本会打印原因并继续 —— 少量排除是正常的,
会记录在 manifest 里;但**如果大批被排除,把输出发过来**。

---

## 第 4 步:上传

```bash
bash scripts/faced_local_prep.sh upload ~/faced_out
```

- 约 **1.8 小时**
- **中断了就重跑同一条命令**,会自动续传(用的是 `rsync --partial`)
- 传完会自动核对集群上的 manifest 并打印各 split 的窗口数

看到类似这样就成功了:

```
splits: {'train': 6720, 'val': 1680, 'test': 1932}
qc: 123 subjects
```

**把这个输出发过来**,我就可以提交训练作业了。

---

## 故障排查

**`ssh amd` 连不上**
→ 那台电脑没配置集群访问。看文末替代方案。

**`unzip` 报磁盘空间不足**
→ 需要约 120 GB 余量。可以只解压需要的部分:
```bash
unzip -l ~/Downloads/FACED.zip | grep -i processed | head   # 先看有哪些目录
unzip ~/Downloads/FACED.zip 'Processed*/*' -d ~/faced_raw   # 只解压这部分
```

**`ModuleNotFoundError: No module named 'numpy'`**
→ `pip3 install numpy scipy pyyaml`

**预处理报 `no .pkl under ...`**
→ 解压出来的目录结构可能多套了一层,试试:
```bash
find ~/faced_raw -name '*.pkl' | head -3      # 看 .pkl 实际在哪
bash scripts/faced_local_prep.sh run <上面那个真实目录> ~/faced_out
```

**上传中断**
→ 直接重跑第 4 步的命令,不会从头开始。

---

## 没有集群访问权限怎么办

在那台电脑上完成第 1–3 步是不行的(第 1 步就需要 ssh)。改成:

1. **把代码从这台电脑拷过去**(U 盘 / AirDrop):
   这台电脑上先执行
   ```bash
   rsync -az amd:/work1/chenyuyou/yifanwang/Zhizhe/PACLock/ ~/PACLock/
   tar czf ~/Desktop/PACLock.tgz -C ~ PACLock
   ```
   把 `~/Desktop/PACLock.tgz` 拷到那台电脑,解压。

2. 在那台电脑上执行**第 2、3 步**(inspect 和 run,都不需要联网)。

3. 把 `~/faced_out`(约 2.46 GB)拷回这台电脑,然后在这台电脑上执行:
   ```bash
   rsync -avP --partial ~/faced_out/ amd:/work1/chenyuyou/yifanwang/Zhizhe/processed/faced/
   ```

---

## 完成之后

我会在集群上提交 5 个模型 × 3 seeds 的训练作业,FACED 就是评测矩阵里
最后一个数据集了。

有任何一步输出看着不对,直接把终端输出贴过来。
