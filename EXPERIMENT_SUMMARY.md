# 实验进度总结（2026-11-22）

> 面向交接与复现：记录两个阶段的协议、已跑结果、产物位置、待办与已知风险。
> 所有数字均来自实际运行的 `results/*.json`，不是估算。

---

## 1. 项目结构

| 阶段 | 目录 | 研究内容 | 状态 |
|---|---|---|---|
| 第一阶段 | `vae_bdm/` | 冻结 49k 全量 VAE 上的静态潜空间数据蒸馏（v2 协议） | 已完成，README 最新 |
| 第二阶段 | `vae_streaming/` | 严格因果流式 + Online VAE + 双层记忆 + 跨版本对齐 + Streaming DM | 代码完成，主结果已出 |

- 特征维度：单图/组级统一为 `16×8×8 = 1024`。
- 阶段二不覆盖 `vae_bdm/`、`checkpoints/` 及阶段一结果。
- 两阶段所有产物分别写入各自目录；`results/`、`checkpoints/`、`figures/`、`logs/` 不入 Git。

### Git 状态

- 远程：`https://github.com/passer-by-a/-Data-Distillation-for-Edge-Computing.git`，分支 `main`。
- 最新提交：`e6f0eed`（阶段二 README 同步）。
- 上一提交：`90f8afd`（双标签协议 + 80 轮配置 + 标签协议测试）。
- 未入库：结果、权重、图、日志。

---

## 2. 第一阶段（`vae_bdm/`）

### 2.1 设置

- 冻结 VAE（49,000 张全量训练），静态潜空间蒸馏。
- CIFAR-10 / CIFAR-100，压缩比 10:1（每类 500 → 50）。
- 3 seeds，报告 mean±std。

### 2.2 结果（v2 协议）

**CIFAR-10**

| 方法 | 精度 |
|---|---|
| Full data | 90.19±0.19 |
| Random | 89.90 |
| K-means | 90.09 |
| K-means + Moment | 90.05 |
| Batch DM | 90.06±0.06 |

**CIFAR-100**

| 方法 | 精度 |
|---|---|
| Full data | 61.56 |
| Random | 58.30 |
| K-means | 59.23 |
| K-means + Moment | 60.85 |
| Batch DM | 59.61±0.25 |

### 2.3 结论

- CIFAR-100 上 DM 比 Random 高 1.3 点、比 K-means 高 0.4 点，追回约 6.7 点的压缩损失。
- CIFAR-10 上各方法接近 50:1 无损，存在明显天花板效应。
- 另有压缩比扫描与 INT8 部署表，见 `vae_bdm/README.md`。

---

## 3. 第二阶段协议（`vae_streaming/`）

### 3.1 数据流

- Base：9,800 张（20%，类平衡；CIFAR-10 各类 967–995 张）。
- 流式：39,200 张（49,000 − 9,800）。
- 主口径：80 轮 × 490 张。
- 其他分组口径：5 轮（7,840/轮）、40 轮（980/轮，内部 392 细分轮再合并）、392 轮（100/轮，每类每轮 10 张）。
- 严格因果：`CausalStream.round(t)` 访问 `t+1` 及以后直接抛错；Base 与所有轮次索引互斥。

### 3.2 两种标签协议（`stream.label_mode`）

| 协议 | 说明 | 用途 |
|---|---|---|
| `supervised_stream` | Base 与流式数据都带真实标签，CE 对当前窗口和 Base Replay 生效 | **当前主口径** |
| `unlabeled_after_base` | 仅 Base 9,800 张有真实标签；流式返回 `UnlabeledBatch`（无标签字段） | 对照实验 |

无标签协议下的标签来源：

- 固定 Initial Base Teacher（深拷贝 + `eval()` + 冻结，version 恒为 0，全程不更新）。
- 伪标签阈值 0.9；低于阈值不进入 CE。
- 来源固定 `PSEUDO_SOURCE=initial_teacher_pseudo`，不与真实标签混用。
- Replay 只保存 Base 真实有标签样本（容量 1,000）。
- 审计：`tests/test_label_protocol.py` 6 项，覆盖标签密封、Teacher 冻结、伪标签来源。

### 3.3 方法

| 方法 | 说明 |
|---|---|
| `dm`（Plain DM） | 单个冻结随机 MLP 的均值匹配 |
| `dm_multi`（Multi-MLP DM） | 3 个不同种子冻结随机 MLP，损失取平均 |
| `online_kmeans` | 固定预算流式质心 / coreset |
| `online_moment` | Chan 累计统计（count/mean/M2） |
| `online_reservoir` | 类平衡 Reservoir 原始样本 + 当前 Encoder 重编码 |
| `online_reservoir_latent` | 诊断：类平衡 Reservoir 只存潜变量，跨版本仅 alignment，不重编码 |
| `dm_nodistill` | 诊断：保留 DM 代表集 / 对齐 / 云端训练，`updates_per_class=0` |

### 3.4 预算（80 轮主口径）

| 项目 | 值 |
|---|---|
| 代表集 | 980（50:1） |
| Raw Replay | 1,000 |
| 在线更新 | 每 2 轮一次，共 40 次 × 150 步 = 6,000 步 |
| DM 更新 | 80 次（每类每次 20 步） |
| 通信 | Full Sync + INT8，累计 82.43 MB |
| 云端训练 | 固定预算，不做基于独立验证集的逐轮选优（`select_best_on_val=false`） |

### 3.5 可复现性锚点

- 80 轮监督：`stream_sha1=b75d2a809942de30`，Base `base_indices_sha1=9d36421552fc6fdf`
  （三 seed 共用同一数据流：`--seeds` 只覆盖 `vae.seed`，`split_seed`/`shuffle_seed` 固定 42）。
- 80 轮无标签：`stream_sha1=a914f9bb4998e17e`。
- 同 seed 内各方法逐位相同的 Encoder 轨迹：
  - sup80 末轮：seed42 `c99628fc00f2373f`、seed43 `f918ab02e872a88b`、seed44 `ac877cc0033185f7`
    （每方法 40 条 SHA1 全部一致）。
  - u2 末轮 `a450a63a284d62fd`。
  - r392cmp：seed42 `de208c5776faa5e9`、seed43 `2ff0ae1c07fb8367`、seed44 `49bc9511bd4c1291`。
- 无标签 Teacher 指纹：`7343521e454f4f08`。

---

## 4. 第二阶段结果

### 4.1 80 轮监督五方法（**当前主表**，3 seeds 42/43/44）

| 方法 | r0 | 最终 | 在线平均 | 最终 F1 | 平均 F1 | INT8 | 方差比 | DM 耗时 |
|---|---|---|---|---|---|---|---|---|
| Plain DM | 76.23±0.26 | **78.65±0.28** | 75.33±0.97 | 78.48±0.36 | 75.26±0.98 | 78.64±0.30 | 2.35±0.16 | 8.4 s |
| Multi-MLP DM | 76.23±0.26 | 78.32±0.66 | 75.53±0.88 | 77.99±0.66 | 75.50±0.83 | 78.32±0.62 | 4.41±0.18 | 15.2 s |
| K-means | 76.23±0.21 | 76.50±0.38 | 74.56±0.93 | 75.98±0.39 | 74.30±0.99 | 76.12±0.47 | 0.03±0.01 | 4.2 s |
| Moment | 76.23±0.26 | 77.11±1.26 | 74.73±1.32 | 76.78±1.33 | 74.46±1.31 | 77.15±1.25 | 1.00±0.00 | 1.0 s |
| Reservoir | 76.23±0.26 | **78.69±1.08** | **76.31±0.52** | 78.53±1.19 | 76.19±0.54 | 78.72±1.04 | 7.12±0.25 | 1.8 s |

逐 seed 最终精度：

- DM：78.44 / 78.54 / 78.97
- Multi-MLP：77.56 / 78.67 / 78.73
- K-means：75.97 / 76.81 / 76.73
- Moment：75.73 / 77.43 / 78.18
- Reservoir：77.45 / 79.38 / 79.24

- 墙钟 258–281 s；五方法共享同一 Initial Base VAE 权重（同 seed 内逐位相同）。
- Reservoir 的初始代表集已改为与其他方法相同的 KMeans 初始化（tag `sameinit80` 重跑），
  因此 r0 与其余方法逐 seed 一致；旧值为 r0 76.25±0.47、最终 78.65±1.50、在线平均 76.11±0.47。

观察：

1. Plain DM 与 Reservoir 的最终均值几乎并列（78.65 vs 78.69），但稳定性差距明显：DM 标准差仅 0.28，Reservoir 为 1.08。
2. Reservoir 在 seed 43/44 领先（79.38 / 79.24），seed 42 为 77.45；单 seed 42 仍会低估 Reservoir。
3. Multi-MLP 比 Plain DM 低 0.33（78.32±0.66），单 seed 时的 0.9 点差距缩到噪声量级。
4. K-means 均值最低（76.50±0.38）；Moment 为 77.11±1.26，单 seed 的 75.73 是三个 seed 中最差的一次。
5. DM 相对 K-means 高 2.15、相对 Moment 高 1.54；Reservoir 在线平均最高（76.31）。

> K-means 行已按计数初始化修复后重跑（tag `fixinit80`），旧值为 76.73±0.54；
> 仅潜变量 Reservoir 见 4.1b，同样为重跑值。

### 4.1b 诊断消融（3 seeds，tag `diag80`）

目的：判断 Reservoir 的优势是否来自逐轮精确重编码，以及 DM 的特征优化本身是否有效。
两个诊断都保留 80 轮主口径的 Online VAE、代表集预算、`diagonal_affine` 对齐与云端训练。

| 变体 | 最终 | 在线平均 | 最终 F1 | INT8 | 方差比 |
|---|---|---|---|---|---|
| Plain DM（主表） | 78.65±0.28 | 75.33±0.97 | 78.48±0.36 | 78.64±0.30 | 2.35±0.16 |
| DM w/o Optimization | 75.95±0.13 | 74.39±1.00 | 75.34±0.15 | 74.66±0.45 | 0.03±0.01 |
| Reservoir（重编码） | 78.65±1.50 | 76.11±0.47 | 78.51±1.45 | 78.68±1.48 | 7.12±0.25 |
| Reservoir（仅潜变量） | **79.76±0.51** | **76.04±0.35** | **79.53±0.62** | **79.76±0.53** | 0.99±0.02 |

逐 seed 最终精度：

- DM w/o Optimization：75.99 / 76.05 / 75.80
- Reservoir（仅潜变量）：79.13 / 79.76 / 80.38

结论：

1. **Reservoir 的优势不来自精确重编码**。只存潜变量、跨版本仅做 alignment，反而比
   重编码高 1.11（79.76 vs 78.65），且 std 从 1.50 降到 0.51、方差比从 7.12 降到 0.99。
   逐轮用当前 Encoder 重编码历史原图不但没有收益，还增加了表示漂移。
2. **DM 的特征优化是真实贡献**。同样的记忆 / 对齐 / 云端训练下，关掉优化掉到
   75.95（低于自身 r0 76.23），DM 优化带来 +2.70。
3. 但“仅潜变量 Reservoir”无需任何蒸馏目标即达到 79.76，比 DM 高 1.11。说明当前
   DM 目标没有充分利用这套记忆与对齐框架，Reservoir 的随机替换策略更强。
4. 两诊断均 80 轮完整、`stream_sha1=b75d2a809942de30`，与主表同数据流可直接对比。
   尚未在 392 轮、无标签与 shift 场景验证，不能直接替换主表结论。

### 4.2 80 轮无标签协议（对照，seed 42）

| 方法 | r0 | 最终 | 在线平均 | 最终 F1 | 平均 F1 | INT8 |
|---|---|---|---|---|---|---|
| Multi-MLP DM | 76.27 | 60.64 | 63.78 | 60.19 | 63.59 | 60.61 |
| Reservoir | 75.91 | **65.70** | **67.24** | **65.40** | **66.94** | 65.65 |
| K-means | 76.27 | 58.99 | 63.65 | 58.14 | 62.91 | 58.46 |
| Moment | 76.27 | 57.58 | 62.05 | 53.96 | 59.95 | 57.62 |

- 伪标签接受 20,625 / 39,200（52.6%），类别分布严重偏斜：
  `[2466, 2878, 1839, 31, 2346, 968, 2450, 2475, 2432, 2740]`（第 3 类仅 31 条）。
- 偏斜来自 Teacher 置信度偏置，不是数据流按类别分桶（数据流不按类别构造）。
- 结论：固定 Initial Base Teacher 不足以支撑 80 轮无标签流，该协议只作对照，不进主表。

### 4.3 40 轮折中口径（seed 42）

| 方法 | r0 | 最终 | 在线平均 | 最终 F1 | INT8 | 方差比 |
|---|---|---|---|---|---|---|
| Plain DM | 75.97 | **77.58** | 74.86 | 77.51 | 77.52 | 1.84 |
| Multi-MLP DM | 75.97 | 77.51 | **75.27** | **77.52** | 77.48 | 3.58 |
| Residual DM（`lambda_latent=1.0`） | 75.97 | 74.08 | 72.11 | 73.81 | 74.01 | 0.63 |

- 40 组 × 980 张，每组一次在线更新（40 × 150 步），累计同步 41.72 MB。
- 排序与 80 轮一致；两种 DM 差距在噪声内；Residual DM 明显掉点，只作结构对照。

### 4.4 392 轮长流（seed 42/43/44）

| 方法 | r0 | 最终 | 在线平均 | 最终 F1 | 平均 F1 | INT8 | 方差比 |
|---|---|---|---|---|---|---|---|
| online_kmeans | 76.87±0.69 | 77.47±2.25 | 74.48±0.29 | 77.19±2.40 | 74.15±0.37 | 76.59±2.29 | 0.04 |
| online_moment | 76.87±0.69 | 77.51±0.89 | 75.46±0.30 | 77.16±0.94 | 75.16±0.34 | 77.47±0.85 | 1.00 |
| online_reservoir | 76.20±0.19 | 78.33±1.03 | **75.75±0.12** | 78.24±1.10 | 75.62±0.08 | 78.31±1.06 | 8.27 |
| dm | 76.87±0.69 | **79.14±1.37** | 75.72±0.25 | **79.13±1.31** | **75.67±0.22** | **79.14±1.38** | 2.91 |

逐 seed 最终精度：

- DM：78.09 / 78.25 / 81.08
- Reservoir：76.95 / 78.60 / 79.43
- Moment：76.64 / 77.17 / 78.73
- K-means：74.30 / 78.81 / 79.31

协议：392 轮 × 每类 10 张，每 10 轮更新一次，末轮强制 flush；`n_train_used=49,000`，`vae_pending_leftover=0`；累计通信 399.94 MB。

观察：

1. DM 最好，Reservoir 次之，Moment ≈ K-means。
2. K-means 方差最大，seed42 明显掉点（74.30），必须三 seed 同时报告。
3. 与 80 轮口径排序一致（DM > Reservoir > Moment ≈ K-means）。
4. 旧 `r392p2`（单 seed、改造前、DM 用 `reencode`）只作历史记录，不与 `r392cmp` 混用。

### 4.5 5 轮旧结果（历史，保留）

| 数据集 | DM | Moment | K-means |
|---|---|---|---|
| CIFAR-100 | 41.68±0.57 | 39.46±0.52 | 36.84±0.84 |
| CIFAR-10 | 78.12±0.50 | 76.74±0.56 | 76.23±1.46 |

---

## 5. 产物位置

### 文档

- `vae_streaming/README.md`：阶段二完整说明（协议、方法、全部结果表）。
- `vae_bdm/README.md`：阶段一完整说明。
- 本文档：跨阶段交接总结。

### 配置

- `vae_streaming/configs/cifar10_supervised_multi80_u2.yaml`：80 轮监督主配置。
- `vae_streaming/configs/cifar10_unlabeled_multi80_u2.yaml`：80 轮无标签配置。
- `vae_streaming/configs/cifar10.yaml`：5 / 40 / 392 轮共用基础配置。

### 结果与图

- 结果 JSON：`vae_streaming/results/cifar10/iid/<method>_{sup80,u2,r40cmp42,r392cmp}_seed*.json`
- 图：
  - `figures/sup80_five_methods_seed42.png`
  - `figures/u2_four_methods_seed42.png`
  - `figures/dm_multi_u2_seed42_curve.png`
  - `figures/cifar10_r392p2_*_iid.png`

### 测试

- 共 34 项：28 项流式协议 + 6 项标签协议。
- 命令：`python -m pytest vae_streaming/tests -q`

### 常用命令

```bash
# 80 轮监督五方法（主口径）
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10_supervised_multi80_u2.yaml \
  --methods dm,dm_multi,online_kmeans,online_moment,online_reservoir \
  --seeds 42,43,44 --tag sup80

# 80 轮无标签四方法
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10_unlabeled_multi80_u2.yaml \
  --methods dm_multi,online_kmeans,online_moment,online_reservoir \
  --seeds 42 --label-mode unlabeled_after_base \
  --pseudo-threshold 0.9 --tag u2

# 392 轮四方法三 seeds
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10.yaml \
  --methods online_kmeans,online_moment,online_reservoir,dm \
  --seeds 42,43,44 --rounds 392 --class-quota 10 \
  --update-every 10 --tag r392cmp

# 40 轮折中口径
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10.yaml \
  --methods dm,dm_multi --seeds 42 --rounds 40 --class-quota 10 \
  --group-rounds 392 --update-every 1 --tag r40cmp42
```

---

## 6. 待办

| 优先级 | 任务 | 说明 |
|---|---|---|
| ~~高~~ | ~~80 轮扩 seed 43/44~~ | 已完成（3 seeds 主表已更新）；均值上 DM 与 Reservoir 并列，DM 方差更小 |
| ~~高~~ | ~~Reservoir 是否依赖重编码 / DM 是否真在蒸馏~~ | 已完成（4.1b）：仅潜变量 Reservoir 79.76，重编码无收益；DM 优化贡献 +2.70 |
| 高 | K-means 392 轮 seed42 掉点归因 | 74.30 vs 78.81 / 79.31，需定位 coreset / 质心更新 |
| 暂停 | CIFAR-100 80 轮迁移 | 预算 980 / 9800、`update_every=2/8/16` 均低于初始（DM 33.3–35.7）；当前 80 轮协议在 100 类上不可用，需重新设计在线更新调度后再启动 |
| 高 | 仅潜变量 Reservoir 扩到 392 轮 / shift / 无标签 | 80 轮已领先 DM 1.11，需确认是否稳定成立 |
| 中 | Reservoir 80 轮高方差归因 | 重编码版 77.45 / 79.38 / 79.24（std 1.08）；仅潜变量版 std 降到 0.51 |
| 中 | Moment 跨 seed 波动归因 | 75.73 / 77.43 / 78.18（std 1.26），排查累计统计与对角仿射对齐误差 |
| 中 | 40 轮扩 seed | 目前仅 seed 42 |
| 低 | 无标签协议改进 | 换 Teacher 或按类别校准阈值，不能只调总阈值 |
| 挂起 | 392 轮 lr 归因消融 | 用户暂未要求 |

---

## 7. 已知风险与口径提醒

1. **20% Base 是共同瓶颈**：初始 VAE 只见过 9,800 张图，弱于阶段一 49k 全量 VAE。增加轮数只能改善增量维护，不能补齐编码器上限。
2. **80 轮主表已扩到三 seed**，但三 seed 共用同一数据流（`split_seed`/`shuffle_seed` 固定 42），seed 方差只覆盖 VAE 初始化与 DM 随机特征，不含数据流划分方差。
3. **无标签协议不能替代主表**：伪标签类别严重偏斜，总体精度受 Teacher 偏差主导。
4. **不同分组口径不可混用**：5 / 40 / 80 / 392 轮的数据流与更新频率不同；`r392p2` 与 `r392cmp` 的 DM 口径也不同。
5. **不使用测试/验证标签逐轮挑模型**：云端固定训练预算（`select_best_on_val=false`）。
6. **对齐统计量是估计值**：跨版本转换标记 `estimated=True`，不能当作真实历史分布；不可靠时回退整集同步，成本计入通信。
7. **仅潜变量 Reservoir 目前只是 80 轮诊断结论**：79.76 未在 392 轮 / shift / 无标签验证，暂不能直接替代主表或据此宣称新方法最优；它也没有蒸馏目标，属于“记忆 + 对齐”基线。
8. **CIFAR-100 两套口径都需重跑**：(a) 80 轮协议在 100 类上失效，所有方法低于初始，暂不可用；(b) 5 轮旧结果（DM 41.68）由旧版代码生成，当前代码同配置只有 38.31±0.76，且旧配置隐式按 val 选最优轮次。两者都不能与 CIFAR-10 主表混用。
