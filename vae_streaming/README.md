# 第二阶段：Memory-aware Streaming Latent Dataset Distillation

> 严格因果数据流下的 **Online VAE + 双层记忆 + 跨版本特征对齐 + Streaming DM**
> 本阶段不实现自适应蒸馏强度、学习式上传调度、Class-Incremental 与 Group-VAE。

---

## 1. 范围与与第一阶段的关系

第一阶段（`vae_bdm/`）在**冻结**的 49k 全量训练 VAE 上做静态潜空间蒸馏。
本阶段（`vae_streaming/`）研究：当数据逐轮到达、VAE Encoder 持续更新时，
如何维护一个可增量同步的 1024 维代表集，并保证历史特征不被错误地混用。

- 不覆盖 `vae_bdm/`、`group_vae/`、`checkpoints/`、`vae_bdm/results/` 的任何产物。
- 本阶段所有产物写入 `vae_streaming/{results,checkpoints,figures}`。
- 单图/组级特征仍为 `16×8×8 = 1024` 维。

---

## 2. 因果数据流协议

从 `data.load_split` 的 49,000 / 1,000 / 10,000 划分出发：

```yaml
stream:
  initial_fraction: 0.20     # Initial Base Set = 9,801 张（类平衡）
  num_rounds: 5              # 第一批最小实验；正式协议为 20
  scenario: iid              # iid / label_shift / covariate_shift
```

- 每轮大小约 7,840 张（49,000 − 9,801 = 39,199，按 5 轮均分）。
- `CausalStream.round(t)` 有访问守卫：访问 `t+1` 及以后直接抛错。
- Base 与所有轮次索引互斥；`assert_no_leakage(t, idx)` 可审计。
- 测试集只用于评估；不进入 VAE 训练、统计记忆、代表集或 DM。
- Label Shift：所有类在 Base 中均出现；每轮强调一个类别块（权重 3:1），只改变比例。
- Covariate Shift：亮度 / 对比度 / 高斯模糊 / 颜色风格按轮次增强；测试域用同一确定性变换，
  但测试图像不参与任何更新。

**VAE 预训练泄漏**：主实验的初始 VAE 只使用 Initial Base Set 训练（30 epoch），
不使用第一阶段 49k 预训练权重。第一阶段权重只作为非因果离线参考。

### 2.1 冻结特征归一化

逐通道 z-score 统计量只在 Initial Base Set 上拟合一次并冻结。
这样边缘与云端共享同一个固定线性变换，编码器更新不会破坏两侧一致性，
也避免用未来数据重估归一化。所有版本的特征、统计量、代表集都在该空间内。

---

## 3. 模块

| 模块 | 文件 | 说明 |
|---|---|---|
| A. Online VAE | `models/online_vae.py` | 只更新 Encoder/Decoder/分类头；固定 steps；可选 replay + stability |
| B. Raw Replay | `memory/replay_buffer.py` | 类平衡 Reservoir，容量 1,000；不上传 |
| C. Streaming Stats | `memory/running_statistics.py` | 每类 count / mean / M2；版本锁定 |
| D. Feature Alignment | `models/feature_alignment.py` | 对角仿射 `z_new ≈ a⊙z_old+b`，锚点拟合/留出检验 |
| E. Representative Memory | `memory/representative_memory.py` | 固定预算 980，稳定 ID / 版本 / 轮次 |
| F. Cloud Incremental | `cloud/incremental_trainer.py` | 保留模型与优化器状态；可切换每轮 scratch |

### 3.1 Online VAE

```
L = MSE(recon, x) + β·KL + λ_c·CE + λ_s·||E_t(x_R) − sg(E_{t−1}(x_R))||²
```

- 每轮固定 `steps_per_round=150`、batch 128；replay 样本占 batch 的 25%。
- `E_{t−1}` 为 eval 模式深拷贝快照，不接收梯度。
- 不同方法使用相同的图像访问预算；无 replay 方法仍跑同样步数，只改变采样来源。

### 3.2 Streaming Statistics

Chan 并行合并：

```
n = n_a + n_b,  δ = μ_b − μ_a
μ = μ_a + δ·n_b/n
M2 = M2_a + M2_b + δ²·n_a·n_b/n
```

- `update()` 要求传入版本等于当前统计版本，否则抛错；
  跨版本必须先 `align_to(version, alignment)`。
- 对齐后 `estimated=True`，表示这是版本转换估计值，不是真实历史图像的重编码结果。
- 只接收当前轮新特征；Replay 回放不会重复计数（接口层面隔离）。

### 3.3 跨版本对齐

- 锚点来自 Replay Memory（历史图像），前一半拟合、后一半留出检验。
- 逐维 `a = Cov(z_old, z_new)/Var(z_old)`，`b = μ_new − a⊙μ_old`。
- `a` 截断到 `[1/3, 3]`；记录 fit / held-out MSE、相对误差、类别中心误差。
- 统计量转换：`μ ← a⊙μ+b`，`v ← a²⊙v`（近似，明确标记 estimated）。
- 对齐不可靠时，正确性回退是整集完整同步；成本计入累计通信。

### 3.4 Streaming DM

```
L_SDM = λ_m·||μ_S − μ_target||² + λ_v·||v_S − v_target||²
        + λ_a·||S − S_aligned_prev||²
```

- 目标统计量来自**当前版本**的统计记忆（已含本轮新样本）。
- 逐类独立 `nn.Parameter` + 独立 AdamW；每类固定 `updates_per_class=20`。
- 全量同步模式优化全部代表；固定预算 patch 模式只优化本轮允许更新的位置，
  其余位置严格保持对齐后的旧值。

### 3.5 通信

- INT8 使用第一阶段的非对称 uint8 协议（`vae_bdm.quant.Quantizer`）。
- **Full Sync**：每轮上传全部 980×1024 特征。
- **Patch Sync**：每轮上传 `Q = round(fraction×M)` 个特征；
  10% 时 Q≈98，位置按类别均衡确定性轮换。
- Patch 消息附带 1024 维 `scale` / `bias` 全局映射；云端先转换全部旧特征，再替换补丁。
- 消息含 `sync_id`，云端重复应用同一消息不会修改状态。
- 同时报告理论字节与 `np.savez` 真实序列化文件大小。

---

## 4. 运行

```bash
# 单元测试（22 项，含完整流水线断点续跑）
python -m pytest vae_streaming/tests/test_streaming_protocol.py -q

# CIFAR-100 IID 主实验（单方法）
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar100.yaml \
  --methods online_replay_align_dm --seeds 42

# 全方法对照（IID）
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar100.yaml \
  --methods frozen_stream_dm,online_naive_dm,online_replay_dm,online_replay_align_dm,online_kmeans,online_herding,online_moment,online_reservoir,full_method \
  --seeds 42,43,44

# 分布变化场景
python -m vae_streaming.experiments.run_label_shift \
  --config vae_streaming/configs/cifar100.yaml --seeds 42
python -m vae_streaming.experiments.run_covariate_shift \
  --config vae_streaming/configs/cifar100.yaml --seeds 42

# 消融（含 stability）
python -m vae_streaming.experiments.run_ablations \
  --config vae_streaming/configs/ablation.yaml \
  --names no_stability,with_stability

# 断点续跑
python -m vae_streaming.experiments.run_iid ... --stop-after 2
python -m vae_streaming.experiments.run_iid ... --resume

# 非因果离线参考
python -m vae_streaming.experiments.run_offline_reference \
  --config vae_streaming/configs/cifar100.yaml \
  --vae-ckpt <ckpt> --protocol-from <result.json> --method bdm

# 汇总与图
python -m vae_streaming.experiments.summarize
python -m vae_streaming.experiments.make_figures
```

---

## 5. 结果

> 全部数字来自实际运行。主表由 `experiments/make_report.py` 生成：
> `results/cifar100/main_table.md` 与 `results/cifar10/main_table.md`；
> 逐种子明细在 `results/<dataset>/<scenario>/*.json`。

### 5.1 CIFAR-100，IID，5 轮，预算 980（最终 50:1），3 seeds

初始 Base VAE 仅见 9,801 张图；所有方法共享同一 Initial Base VAE 与云端协议。
初始精度 **35.91±0.80**。

| 方法 | 最终 | 在线平均 | 最终 F1 | 累计通信 | 组/单图方差比 |
|---|---|---|---|---|---|
| online VAE + replay + DM（重编码） | **43.19±0.46** | **41.40±0.37** | 43.06 | 5.96 MB | 0.65 |
| moment calibration | 39.46±0.52 | 38.12±0.42 | 38.33 | 5.96 MB | 1.00 |
| frozen VAE + DM | 39.05±0.78 | 38.48±0.51 | 38.39 | 5.96 MB | 1.00 |
| replay + 对齐 + Streaming DM | 39.02±0.41 | 37.10±0.25 | 37.94 | 5.96 MB | 1.00 |
| online naive DM | 38.45±2.61 | 36.10±0.85 | 37.69 | 5.96 MB | 0.98 |
| online Herding | 37.23±0.89 | 37.00±0.68 | 36.54 | 5.96 MB | 0.74 |
| online K-means | 36.84±0.84 | 35.98±0.53 | 36.46 | 5.96 MB | 0.29 |
| full method（patch 10%） | 35.67±0.62 | 34.28±0.59 | 35.43 | **1.54 MB** | 0.18 |
| online Reservoir | 35.14±0.63 | 34.50±0.80 | 34.17 | 5.96 MB | 0.99 |

观察：

1. **Replay 重编码 + Streaming DM 是明确赢家**：最终比冻结 VAE 高 **+4.1**，在线平均高 **+2.9**，
   且已接近同编码器空间的非因果离线 Batch DM（43.08）。
2. **对角仿射对齐仍无收益**：held-out 相对误差 0.160±0.008；最终 39.02 ≈ 冻结 39.05。
   5 轮内历史统计记忆抵不过对齐误差。
3. **固定补丁 10% 通信降 74%，但掉 3.4 点**，组/单图方差比 0.18。
4. **简单 coreset 全面落后**；moment calibration 是低成本强基线。
5. INT8 测试精度相对 FP32 损失 ≤0.1 点。

### 5.1b CIFAR-10，IID，5 轮，预算 980（50:1），3 seeds

CIFAR-10 的 20% Base 已能训练出较强编码器，初始精度 **76.08±0.58**，
方法间差异缩小到约 2 点，呈现明显的**天花板效应**。

| 方法 | 最终 | 在线平均 | 累计通信 | 方差比 |
|---|---|---|---|---|
| online VAE + replay + DM（重编码） | **78.84±0.66** | **77.77±0.28** | 5.96 MB | 0.88 |
| online Herding | 77.45±0.45 | 76.88±0.31 | 5.96 MB | 0.90 |
| online Reservoir | 76.81±0.46 | 76.04±0.22 | 5.96 MB | 1.08 |
| online moment | 76.74±0.56 | 75.16±0.41 | 5.96 MB | 1.00 |
| replay + align + DM | 76.69±0.81 | 74.90±0.51 | 5.96 MB | 1.00 |
| online naive DM | 76.44±0.72 | 73.56±0.91 | 5.96 MB | 1.01 |
| frozen VAE + DM | 76.33±0.25 | 76.39±0.23 | 5.96 MB | 1.00 |
| full method（patch 10%） | 76.40±0.99 | 74.20±0.50 | **1.54 MB** | 0.48 |
| online K-means | 76.23±1.46 | 74.44±0.42 | 5.96 MB | 0.40 |

观察：

1. **CIFAR-10 上方法差距小**：最佳比冻结高约 2.5 点；20% Base 已接近够用。
2. 重编码 replay + DM 与 Online Herding 最好；对齐 + DM 仍无收益（与 C100 一致）。
3. **patch 10% 在 CIFAR-10 上精度基本不变**（76.40 vs 冻结 76.33），
   同时通信降 74%——固定预算同步的代价高度依赖数据集难度；
   在 C100 上同样的 10% 会掉 3.4 点。
4. 在线早期 naive DM 的在线平均明显下降（73.56），最终才追回，说明无 replay/对齐时
   编码器更新会短期破坏特征空间。

### 5.2 Label Shift（2 seeds）

| 方法 | 最终 | 当前强调类 | 历史类 |
|---|---|---|---|
| online VAE + replay + DM | **42.63±0.11** | 52.72±0.85 | **40.10±0.36** |
| frozen VAE + DM | 39.24±0.60 | 44.30±1.20 | 37.97±0.45 |
| replay + align + DM | 38.81±0.23 | 46.20±0.37 | 36.97±0.19 |
| full method（patch 10%） | 35.13±0.14 | 41.85±0.35 | 33.45±0.08 |

历史类精度没有崩溃；重编码 replay + DM 在历史类与当前类上都最好。

### 5.3 Covariate Shift（2 seeds）

| 方法 | 原域 | 当前域（shift 强度 1.0） |
|---|---|---|
| frozen VAE + DM | **37.35±0.52** | 19.97±0.23 |
| online VAE + replay + DM | 35.42±0.08 | **35.39±0.74** |
| replay + align + DM | 33.11±0.86 | 30.96±1.33 |
| full method（patch 10%） | 30.02±0.83 | 27.65±0.56 |

**明确的稳定性—可塑性权衡**：冻结 VAE 在原域最好，但在漂移域跌到近随机；
持续更新 VAE 的方法在原域只掉约 2 点，却在当前域保持 30–35 点。
重编码 replay + DM 几乎消除了域差（35.42 vs 35.39）。
这是本阶段最清晰的正向结论。

### 5.4 消融（单种子 42）

| 设置 | 最终 | 在线平均 | 累计通信 |
|---|---|---|---|
| patch 5%（Q≈49） | 35.19 | 34.60 | 1.29 MB |
| patch 10%（Q≈98） | 35.67 | 34.28 | 1.54 MB |
| patch 20%（Q≈196） | 37.13 | 35.41 | 2.03 MB |
| 全同步，预算 980 | 39.02 | 37.10 | 5.96 MB |
| 全同步，预算 2450 | 40.85 | 39.44 | 14.9 MB |
| 全同步，预算 4900 | 42.04 | 40.49 | 29.8 MB |
| patch 10%，预算 2450 | 36.89 | 35.80 | 3.77 MB |
| patch 10%，预算 4900 | 38.29 | 36.79 | 7.50 MB |
| 云端每轮 scratch（预算 980） | **40.78** | 38.42 | 5.96 MB |
| 云端增量（预算 980） | 39.02 | 37.10 | 5.96 MB |
| 无 stability 项 | 38.88 | 37.34 | 5.96 MB |
| stability λ_s=0.1 | **40.16** | **38.48** | 5.96 MB |
| patch 10%，优化全部代表（`all`） | 37.21 | 35.48 | 1.54 MB |

观察：

1. **patch 比例单调影响精度**：10% 相对 5% 只 +0.48，20% 再 +1.46；5 轮内补丁预算过小。
2. **代表预算比同步方式更关键**：980→4900 全同步 +3.0；patch 980→4900 +2.6。
3. **P0-3 变量分离**：同样 1.54 MB，`patch_optimize=all` 为 37.21，比默认
   `synced_only` 的 35.67 高 **+1.54**。说明补丁模式下大部分精度损失来自
   “只优化 Q 个位置”，而不是通信压缩本身。但 `all` 会让边缘本地代表领先云端
   （`local_unsynced` max≈1.01），属于显式不一致，不能当作可部署协议。
4. **云端每轮 scratch 反而比增量高 1.76 点**。每轮 200 步对 980 条代表已足够从头拟合，
   持续保留的模型/优化器状态在 5 轮早期没有收益。
5. **稳定性项有效**：最终 +1.28 / 在线 +1.14；对齐误差也明显下降。
6. patch 模式方差比仍偏低（0.18–0.48），只更新固定位置会牺牲类内多样性。

### 5.5 非因果离线参考

| 参考 | 方法 | 编码器 | 归一化 | 最终 |
|---|---|---|---|---|
| stage1_bdm | BDM | 第一阶段 49k 全量 VAE | 全量训练集重估 | **59.24** |
| stage1_kmeans | K-means | 第一阶段 49k 全量 VAE | 全量训练集重估 | 58.31 |
| stage1_herding | Herding | 第一阶段 49k 全量 VAE | 全量训练集重估 | 55.39 |
| onlineenc_bdm | BDM | 本阶段最终因果 Encoder（v5） | 与在线方法共享 Base 协议 | **43.08** |
| onlineenc_kmeans | K-means | 同上 | 同上 | 41.23 |
| onlineenc_herding | Herding | 同上 | 同上 | 39.80 |

- 在同一最终 Encoder + 同一归一化协议下，离线一次 Batch DM 为 43.08，
  在线最佳（replay 重编码 + DM）为 **43.19**，**在线处理代价已基本消失**（+0.11，种子波动内）。
- 与第一阶段 49k 预训练 VAE 的 59.24 相差约 16 点，说明**当前最大瓶颈是
  20% Base 训练出的编码器质量**，而不是蒸馏或同步机制。20 轮完整流式训练应显著缩小该差距。

### 5.6 计算与通信

每个 JSON 记录：每轮 VAE 时间、编码时间、对齐时间、DM 时间、同步字节、
云端训练时间、峰值显存、Replay 实际字节、累计上传字节。
本机（RTX 4090 D）5 轮单方法边缘总耗时约 30–100 s，峰值显存 <1 GB；
云端 6 次训练合计约 2–3 s。完整同步累计约 6.1 MB，patch 10% 约 1.57 MB（理论字节）。

图（`figures/`）：

- `cifar100_accuracy_iid.png`、`cifar100_communication_iid.png`、
  `cifar100_drift_alignment_iid.png`
- `cifar100_accuracy_label_shift.png`、`cifar100_accuracy_covariate_shift.png`
- `cifar10_accuracy_iid.png`、`cifar10_communication_iid.png`、
  `cifar10_drift_alignment_iid.png`

### 5.7 变量分离实验（CIFAR-100 IID，seed 42）

把“重编码”“代表集重建”“DM 更新”“上传预算”拆成独立变量：

| 方法 | 最终 | 在线平均 | 累计通信 | 更新比例 | 上传比例 | 边缘/云端差异 |
|---|---|---|---|---|---|---|
| Replay-Reinit（只重编码 + KMeans） | 41.01 | 39.25 | 5.96 MB | 100% | 100% | 0 |
| Replay + DM（重编码 + 初始化 + DM） | **43.21** | **41.34** | 5.96 MB | 100% | 100% | 0 |
| Affine + DM（正确归一化空间） | 38.88 | 37.34 | 5.96 MB | 100% | 100% | 0 |
| Affine + DM，100% 更新 / 10% 上传 | 37.21 | 35.48 | **1.54 MB** | 100% | 10% | 1.013 |
| Affine + Partial DM，10% 更新 / 100% 上传 | 35.56 | 34.76 | 5.96 MB | 10% | 100% | 0 |
| Affine + Partial DM，10% 更新 / 10% 上传 | 35.58 | 34.73 | **1.54 MB** | 10% | 10% | 0 |

四个问题的直接回答：

1. **之前最强的 Replay 方法强在重编码，还是重新构建代表集？**
   两者都重要，但可分开：只重编码 + KMeans 重建（Replay-Reinit）已经到 41.01；
   在此基础上跑 Streaming DM 再 +2.20（43.21）。所以“Replay 强”主要来自**当前版本精确特征 + 重新初始化**，
   DM 是明确的增量收益。
2. **Alignment 失效是否与坐标转换有关？**
   坐标 bug 已修（对齐在归一化空间拟合，held-out 误差相对“不对齐”下降约 25%），
   但 `Affine + DM` 仍是 38.88，与冻结 VAE 39.05 基本持平。
   说明对齐失效**不是坐标 bug 导致的**；主因是对角仿射在 5 轮/980 预算下不足以恢复历史代表的有效性。
3. **DM 对旧代表集有没有额外收益？**
   对“当前版本重编码”的代表集有 +2.20；对“跨版本对齐后的旧代表”只有约 +0.3。
   结论：**DM 的价值依赖代表特征已经是当前版本**；在旧特征上做 DM 无法弥补对齐误差。
4. **Patch 精度下降来自减少优化，还是限制上传？**
   - 100/100：38.88
   - 100/10：37.21（仅限制上传，−1.67）
   - 10/100：35.56（仅减少更新，−3.32）
   - 10/10：35.58（两者都限制，与 10/100 几乎相同）

   **主要损失来自减少更新预算，而不是上传预算**。100/10 的 1.013 边缘/云端差异说明
   本地代表领先云端，不能当作可部署协议；10/100 是全量上传、严格一致，但只更新 10% 仍掉 3.3 点。

### 5.8 原版 DM（线性核均值匹配）对照

原版 Distribution Matching 的主实验使用**线性核 MMD**，等价于匹配特征均值；
本项目的 `l_mean` 正是均值差平方（与线性核 MMD² 成正比），因此“原版 DM”对应
`lambda_var=0, lambda_anchor=0` 的变体。注意：这不是像素空间 DM（没有 decoder-in-the-loop
与可微增强），而是在同一个 1024 维 VAE 特征空间里做线性核 DM。

Stage 1（`vae_bdm`，CIFAR-10/100 10:1，3 seeds）：

| 数据集 | 原版 DM（mean-only） | mean + var | 差异 |
|---|---|---|---|
| CIFAR-100 | 60.08 | **60.37** | −0.29 |
| CIFAR-10 | **90.01** | 89.85 | +0.16 |

Stage 2（CIFAR-100 IID，seed 42，full sync，100% 更新）：

| 方法 | 最终 | 在线平均 | 组/单图方差比 |
|---|---|---|---|
| Replay + DM（mean+var+anchor） | **43.21** | **41.34** | 0.65 |
| Replay + 原版 DM（mean-only） | 42.83 | 40.68 | 0.44 |
| Affine + DM（mean+var+anchor） | **38.88** | 37.34 | 1.00 |
| Affine + 原版 DM（mean-only） | 38.74 | **37.62** | 0.26 |

解读：

1. **原版 DM 与 mean+var 差距很小**：C100 上 mean+var 略好（stage 1 −0.29，
   stage 2 replay −0.38），C10 上 mean-only 略好（+0.16）。
2. **均值匹配会收缩方差**：stage 2 中 mean-only 的方差比只有 0.26（affine）和 0.44（replay），
   而 mean+var 恢复到 1.00/0.65。方差收缩不一定立即掉点，但会牺牲类内多样性。
3. 原版 DM 在文献里的优势主要来自**强特征空间 + 像素增强 + 较大 IPC**；
   在我们的 1024 维 VAE 潜空间、每类 10 个代表、无像素增强的条件下，
   仅做均值匹配并不优于 mean+var，也没有解决对齐误差问题。

---

## 6. 测试覆盖（任务书 19 项 + 7 项审计回归）

| # | 检查 | 测试 |
|---|---|---|
| 1 | 当前 Encoder 更新 | `test_encoder_updates` |
| 2 | 上一轮快照不接收梯度 | `test_previous_encoder_frozen` |
| 3 | 不能访问未来轮次 | `test_no_future_access` |
| 4 | Replay 不超容量 | `test_replay_capacity` |
| 5 | 回放不重复计数 | `test_replay_not_double_counted` |
| 6 | 冻结 Encoder 下统计一致 | `test_streaming_stats_match_offline` |
| 7 | 禁止跨版本直接混用 | `test_no_mixed_versions` |
| 8 | 对齐只用历史锚点 | `test_alignment_uses_history_only` |
| 9 | 边缘/云端版本一致 | `test_sync_versions_match` |
| 10 | 代表数不超预算 | `test_budget` |
| 11 | 维度 1024 | `test_dim_1024` |
| 12 | INT8 可正确解码 | `test_int8_roundtrip` |
| 13 | 字节可复核 | `test_bytes_recount` |
| 14 | 固定补丁预算一致 | `test_patch_consistency` |
| 15 | 重复消息幂等 | `test_duplicate_sync_idempotent` |
| 16 | 保留云端模型/优化器状态 | `test_cloud_state_preserved` |
| 17 | 测试集隔离 | `test_test_set_isolated` |
| 18 | 完整训练 = 断点恢复 | `test_full_pipeline_resume` |
| 19 | 第一阶段不受影响 | `test_stage1_intact` |
| 20 | 对齐在 Normalized 空间拟合/应用 | `test_alignment_normalized_space` |
| 21 | Base 缓存签名覆盖划分与架构 | `test_base_cache_signature_changes` |
| 22 | 断点签名覆盖主要实验参数 | `test_resume_signature_covers_config` |
| 23 | reencode 真正调用 StreamingDM | `test_reencode_runs_streaming_dm` |
| 24 | 完整流水线断点恢复 exact 对齐 | `test_full_pipeline_resume` |
| 25 | Replay-Reinit 仅重编码不跑 DM | `test_replay_reinit_baseline` |
| 26 | 更新预算与上传预算分离 | `test_update_upload_budget_separation` |

---

## 7. 审计修复记录

2026-11-16 对代码做了一次逐项审计，结论与修复如下。

| 编号 | 问题 | 结论 | 修复 |
|---|---|---|---|
| P0-1 | Raw / Normalized 特征空间混用 | **真 bug**：对角 scale 不变，但 bias 被错误地按 raw 空间拟合后应用到 normalized 空间 | `estimate_feature_alignment(..., protocol=...)` 在归一化空间拟合；同时记录 raw / normalized 两种 drift；新增测试 20 |
| P0-2 | `online_replay_dm` 不调用 `StreamingDM` | **真 bug**：实际只做 KMeans 重编码，变量与命名不符 | 重编码后调用 `StreamingDM.update`；新增测试 23 |
| P0-3 | Patch 同时限制优化位置和上传位置 | **变量混淆**：规范允许，但无法区分“通信预算”与“部分更新”的贡献 | 新增 `distillation.update_fraction`，与 `patch_fraction` 独立；实现 100/100、100/10、10/100、10/10 四组；`all` 记录 `local_unsynced` 边缘/云端差异 |
| P0-4 | Base VAE 缓存键不完整 | **真风险**：改 `split_seed/shuffle_seed/架构/超参` 会误用旧缓存 | 缓存签名包含数据集、划分种子、Initial 比例、模型架构与训练超参；加载时逐项校验；新增测试 21 |
| P1-5 | `reencode` 清空累计统计量 | **设计限制，非 bug**：无跨版本对齐时只能从 replay+current 重建统计 | 记录 `stats_scope=replay_plus_current`；README 明确说明 |
| P1-6 | Encoder 变化后云端没有显式补偿 | **非 bug**：每轮在同步后的新特征空间上重训 200 步即是补偿；scratch 对照甚至更好 | 保留 `IncrementalCloudLearner` 与 scratch 对照；文档说明 |
| P1-7 | 断点签名未覆盖主要参数 | **真风险**：改 `updates_per_class/cloud.lr/...` 后 resume 会静默复用旧状态 | 改为完整配置哈希；新增测试 22 |
| P1-8 | Alignment 训练/测试误差未分离 | **部分成立**：MSE 已分离，但类中心误差用了全部锚点 | 分别报告 `fit_class_center_err` 与 held-out `class_center_err`；新增测试 20 |
| P2-9 | 计时、INT8、恢复统计 | **部分成立**：评测编码未单列、resume 的 wall time 只算续跑段、INT8 测试口径未注明 | 拆分 `eval_encode_sec` / `test_int8_sec`；恢复累计 `wall_sec`；写入 `test_int8_note` |
| 追加 | Reencode 与蒸馏未拆分 | **方法学问题**：重编码 + KMeans 被当成 Replay+DM | 新增 `replay_reinit`（只重编码+初始化）与 `replay_dm`（再跑 DM）两个独立方法；新增测试 25 |
| 追加 | 样本计数口径单一 | **审计问题**：无法区分“已见样本/有效统计/Replay” | 逐轮记录 `n_cumulative_seen / n_effective_stats / n_replay`；汇总写 `cumulative_seen_samples / effective_stats_samples / replay_samples_final` |
| 追加 | 对齐只有误差，没有前/后对比 | **验证不足** | 记录 `no_align_heldout_mse`、`improvement_heldout`、`class_center_improvement`，直接验证对齐相对“不对齐”的增益 |

修复后 `vae_streaming` 测试 28 项全过；受影响结果已整体重跑（旧结果保留在
`results/_stale_pre_audit/`）。

---

## 8. 已知限制

1. **5 轮 + 20% Base 是早期阶段**：初始 VAE 只见过 9,801 张图，特征质量明显弱于
   第一阶段 49k 全量 VAE；本阶段结论只描述因果早期的行为，不外推到 20 轮收敛。
2. **初始代表集使用 Base 标签**：Base 的标签在因果协议内可访问；但这不是无监督方法。
3. **对角仿射是近似**：held-out 相对误差约 0.16；加入 stability 项可降到 0.06，
   但仍不能把对齐后的历史统计量宣称为真实历史图像在新 Encoder 下的精确分布。
4. **Patch 模式只更新 Q 个位置**：5 轮 × 10% 只能覆盖约一半代表位置，
   未覆盖位置会保留对齐后的 Base 代表，可能出现组级方差收缩（方差比 0.18）。
   消融显示 `patch_optimize=all` 可把同通信预算的精度从 35.67 提到 37.21，
   但会造成边缘/云端显式状态差异，不能直接部署。
5. **云端验证集**：使用 1,000 张真实验证图像特征（与第一阶段 v2 协议一致），
   不参与 VAE/DM/Memory 更新，也不使用测试集调参。
6. **归一化冻结在 Base**：编码器漂移后特征尺度会变化，但边缘与云端保持一致；
   这是有意的因果选择。
7. **离线参考非因果**：`run_offline_reference.py` 使用全量 49k，只能估计在线差距，
   不能当作同等数据可访问条件的可部署方法。
8. **5 轮内云端 scratch 优于增量**（40.78 vs 39.02）：每轮 200 步对 980 条代表
   足以从头拟合；保留优化器状态的价值需要 20 轮以上再验证。
9. **patch 10% 不能直接部署**：累计通信降 74%，但最终掉 3.4 点、方差比 0.18。
   若要在固定预算下使用，需要提高 patch 比例或增加轮次覆盖全部位置。
10. **对齐误差仍是主要风险**：对角仿射 held-out 相对误差 0.160；加入 stability 项
    可降到 0.062，但仍不能把对齐统计量当作精确历史分布。
