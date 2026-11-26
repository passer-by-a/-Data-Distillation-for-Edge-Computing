# 第二阶段：Memory-aware Streaming Latent Dataset Distillation

> 严格因果数据流下的 **Online VAE + 双层记忆 + 跨版本特征对齐 + 代表集蒸馏**。
> 当前主口径：**80 轮、Initial Base 20%、9 种方法 × 3 种子**，
> 在 CIFAR-10 / CIFAR-100 上各生成一张汇总表。

---

## 1. 范围与第一阶段的关系

第一阶段（`vae_bdm/`）在**冻结**的 49k 全量训练 VAE 上做静态潜空间蒸馏。
本阶段（`vae_streaming/`）研究：当数据逐轮到达、VAE Encoder 持续更新时，
如何维护一个可增量同步的 1024 维代表集，并保证历史特征不被错误地混用。

- 不覆盖 `vae_bdm/`、`checkpoints/`、`vae_bdm/results/` 的任何产物。
- 本阶段所有产物写入 `vae_streaming/{results,checkpoints,figures}`。
- 单图/组级特征仍为 `16×8×8 = 1024` 维。

---

## 2. 因果数据流协议

从 `data.load_split` 的 49,000 / 1,000 / 10,000 划分出发：

```yaml
stream:
  initial_fraction: 0.20          # Initial Base = 约 20%
  num_rounds: 80                  # 每轮 490（C100 490/489 混合）
  scenario: iid
  label_mode: supervised_stream   # supervised_stream / unlabeled_after_base
  class_balanced_initial: true
```

- **Base 配额按类独立取整**：每类 `round(20%)`，类间不严格相等。
  CIFAR-10 每类 977–983、合计 **9,800**，每轮 490（80 × 490 = 39,200）；
  CIFAR-100 每类 97–99、合计 **9,801**，每轮 490/489 混合（合计 39,199）。
  两数据集累计均覆盖 49,000 张训练图。论文若写“每类 20%”，需注明实际配额。
- IID 流：剩余样本用固定种子整体打乱一次后等分各轮。
- 云端每次更新固定 **5 个 epoch**（980 代表时约 20 步），
  且 `select_best_on_val=false`（不做逐轮验证集选优）。
- 在线 VAE 每 **2 轮**更新一次：40 次 × 150 步 = 6,000 步，batch 128。
- `CausalStream.round(t)` 有访问守卫：访问 `t+1` 及以后直接抛错。
- Base 与所有轮次索引互斥；`assert_no_leakage(t, idx)` 可审计。
- 测试集只用于评估；不进入 VAE 训练、统计记忆、代表集或 DM。
- Label Shift：所有类在 Base 中均出现；每轮强调一个类别块（权重 3:1），只改变比例。
  历史类分组显式排除当前块（`np.setdiff1d`），避免轮数超过块数后退化为全测试集。
- Covariate Shift：亮度 / 对比度 / 高斯模糊 / 颜色风格按轮次增强；测试域用同一确定性变换，
  但测试图像不参与任何更新。两种场景当前没有保留最新结果（见第 8 节）。

### 2.1 标签协议（`stream.label_mode`）

结果 JSON 的 `label_mode` 字段记录实际口径：

- `supervised_stream`（默认，当前主口径）：Initial Base 与每轮流式数据都带真实标签，
  在线 VAE 的分类项对当前窗口与 Base Replay 都生效。
- `unlabeled_after_base`：只有 Initial Base 的约 9,800 张带真实标签。`CausalStream`
  返回 `UnlabeledBatch`，该对象没有标签字段；流式标签只能来自
  **固定 Initial Base Teacher** 的伪标签：
  - Teacher 是 Initial Base VAE 的深拷贝，`eval()` + 冻结，version 恒为 0，全程不更新；
  - 阈值 `pseudo_labels.threshold=0.9`，低于阈值的样本不进入分类损失；
  - 伪标签来源固定为 `PSEUDO_SOURCE=initial_teacher_pseudo`；
  - Replay 只保存 Initial Base 的真实有标签样本；
  - `vae_losses(..., n_unlabeled)` 让 CE 只对伪标签部分求均值。
- 协议审计：`tests/test_label_protocol.py` 覆盖标签密封、Teacher 冻结、
  伪标签来源、无标签批次可运行与预算校验。

### 2.2 冻结特征归一化

逐通道 z-score 统计量只在 Initial Base Set 上拟合一次并冻结。
边缘与云端共享同一个固定线性变换，编码器更新不会破坏两侧一致性，
也避免用未来数据重估归一化。所有版本的特征、统计量、代表集都在该空间内。

---

## 3. 模块与方法

### 3.1 核心模块

| 模块 | 文件 | 说明 |
|---|---|---|
| A. Online VAE | `models/online_vae.py` | 只更新 Encoder/Decoder/分类头；固定 steps；可选 replay + stability |
| B. Raw Replay | `memory/replay_buffer.py` | 类平衡 Reservoir，容量 1,000；不上传 |
| C. Streaming Stats | `memory/running_statistics.py` | 每类 count / mean / M2；版本锁定 |
| D. Feature Alignment | `models/feature_alignment.py` | 对角仿射 `z_new ≈ a⊙z_old+b`，锚点拟合/留出检验 |
| E. Representative Memory | `memory/representative_memory.py` | 固定预算 980，稳定 ID / 版本 / 轮次 |
| F. Cloud Incremental | `cloud/incremental_trainer.py` | 保留模型与优化器状态；可切换每轮 scratch |
| G. 伪标签 | `models/pseudo_labeler.py` | `FixedTeacher` + 阈值伪标签；仅 `unlabeled_after_base` 使用 |

### 3.2 Online VAE

```
L = MSE(recon, x) + β·KL + λ_c·CE + λ_s·||E_t(x_R) − sg(E_{t−1}(x_R))||²
```

- 每次更新固定 `steps_per_round=150`、batch 128；replay 样本占 batch 的 25%。
- `E_{t−1}` 为 eval 模式深拷贝快照，不接收梯度。
- 不同方法使用相同的图像访问预算；无 replay 方法仍跑同样步数，只改变采样来源。

### 3.3 Streaming Statistics

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

### 3.4 跨版本对齐

- 锚点来自 Replay Memory（历史图像），类内交替切分后拟合/留出检验。
- 逐维 `a = Cov(z_old, z_new)/Var(z_old)`，`b = μ_new − a⊙μ_old`。
- `a` 截断到 `[1/3, 3]`；记录 fit / held-out MSE、相对误差、类别中心误差。
- 统计量转换：`μ ← a⊙μ+b`，`v ← a²⊙v`（近似，明确标记 estimated）。
- 对齐不可靠时，正确性回退是整集完整同步；成本计入累计通信。

### 3.5 Streaming DM

```
f = RandomMLP(1024 → 2048 → 1024)   # 固定随机初始化，参数冻结
L_SDM = ||mean(f(real_z)) − mean(f(syn_z))||² + λ_a·||S − S_aligned_prev||²
```

- 与第一阶段 `BatchDM` 使用同一个目标，只是真实特征来自**当前版本**的统计/回放数据。
- 逐类独立 `nn.Parameter` + 独立 AdamW；每类固定 `updates_per_class=20`。
- 主口径 `exact_target=True`：每轮都用当前 Encoder 精确重编码 Replay + 当前轮作为目标，
  更新轮与非更新轮的目标集合一致。
- `feature_type=multi_mlp` 时用 3 个不同种子的冻结随机 MLP 对损失取平均（`dm_multi`）；
  默认单 MLP 为 `dm`。两者共享其余全部预算与协议。

### 3.6 其他代表集方法

| 方法 | 机制 |
|---|---|
| `online_kmeans` | 固定预算的流式类内质心，跨版本经对角仿射继承 |
| `online_moment` | 质心 + 解析式类均值/方差校准 |
| `online_reservoir` | 类平衡原始图像缓冲，每轮用当前 Encoder 精确重编码 |
| `online_reservoir_latent` | 只保存潜变量、不重编码的诊断变体 |
| `kip` | 潜空间 RBF 核岭回归（闭式解 + 对 support 特征反传，20 步） |
| `exstream` | 类缓冲满时合并类内最近的两个旧代表，新样本占独立槽位 |
| `gcr` | 真实潜特征候选池（4× 预算）+ 梯度内积 facility-location 选择 |

GCR 实现说明：facility-location 增益初值从 0 开始（旧实现误用 `-inf`，会让首点退化为
行内积和）；候选池相似度矩阵按 1024 行分块计算，避免全池 n×n 矩阵显存溢出。

### 3.7 通信

- INT8 使用第一阶段的非对称 uint8 协议（`vae_bdm.quant.Quantizer`）。
- **Full Sync（主口径）**：每轮上传全部 980×1024 特征，80 轮累计约 80,498 KB。
- **Patch Sync**：每轮上传 `Q = round(fraction×M)` 个特征；
  10% 时 Q≈98，位置按类别均衡确定性轮换。主表未使用。
- Patch 消息附带 1024 维 `scale` / `bias` 全局映射；云端先转换全部旧特征，再替换补丁。
- 消息含 `sync_id`，云端重复应用同一消息不会修改状态。
- Patch 同步时逐槽版本号按消息实际版本写入，不再把全部槽位标成最新。

---

## 4. 运行

```bash
# 单元测试（当前 52 passed, 1 skipped）
python -m pytest vae_streaming/tests -q

# CIFAR-10：80 轮监督流式，9 方法 × 3 种子
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10_supervised_multi80_u2.yaml \
  --methods dm,dm_multi,online_kmeans,online_moment,online_reservoir,online_reservoir_latent,kip,exstream,gcr \
  --seeds 42,43,44 --rounds 80 --tag sup80v2

# CIFAR-100：同协议，只换配置
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar100_supervised_multi80_u2.yaml \
  --methods dm,dm_multi,online_kmeans,online_moment,online_reservoir,online_reservoir_latent,kip,exstream,gcr \
  --seeds 42,43,44 --rounds 80 --tag sup80v2

# 生成汇总表（tag 会从方法名中去掉）
python -m vae_streaming.experiments.make_report --dataset cifar10 \
  --rounds 80 --tag _sup80v2 --out vae_streaming/results/cifar10/main_table_multi_seed_v2.md
python -m vae_streaming.experiments.make_report --dataset cifar100 \
  --rounds 80 --tag _sup80v2 --out vae_streaming/results/cifar100/main_table_multi_seed_v2.md

# 分布变化场景（当前无保留结果）
python -m vae_streaming.experiments.run_label_shift \
  --config vae_streaming/configs/cifar100_supervised_multi80_u2.yaml --seeds 42
python -m vae_streaming.experiments.run_covariate_shift \
  --config vae_streaming/configs/cifar100_supervised_multi80_u2.yaml --seeds 42

# 原始像素表示（representation=pixel）：跳过 VAE，特征为 3×32×32=3072 维
# 标准化像素，云端 MLP 为 3072→512→类别数
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10_supervised_multi80_u2.yaml \
  --methods dm,dm_multi,exstream,gcr,online_reservoir \
  --seeds 42,43,44 --rounds 80 --tag pix80v3 --representation pixel

# 断点续跑
python -m vae_streaming.experiments.run_iid ... --stop-after 2
python -m vae_streaming.experiments.run_iid ... --resume

# 非因果离线参考
python -m vae_streaming.experiments.run_offline_reference \
  --config vae_streaming/configs/cifar100_supervised_multi80_u2.yaml \
  --vae-ckpt <ckpt> --protocol-from <result.json> --method bdm
```

---

## 5. 结果

> 全部数字来自实际运行（tag `sup80v2`，80 轮，种子 42/43/44）。
> 主表由 `experiments/make_report.py` 生成：
> `results/cifar10/main_table_multi_seed_v2.md` 与
> `results/cifar100/main_table_multi_seed_v2.md`；
> 逐种子明细在 `results/<dataset>/iid/*_sup80v2_seed*.json`。
> 所有方法共享同一 Initial Base VAE、同一数据流与同一云端协议。
> 潜空间 vs 原始像素的对照（`representation=pixel`）见
> `results/pixel_vs_vae_comparison.md`。

### 5.1 CIFAR-10，IID，80 轮，预算 980（最终 50:1），3 seeds

| 方法 | 种子 | 初始 | 最终 | 在线平均 | 最终 F1 | INT8 | 累计通信 | 方差比 | 对齐误差 |
|---|---|---|---|---|---|---|---|---|---|
| dm | 3 | 76.25±0.42 | 79.51±0.42 | 76.72±0.77 | 79.64±0.43 | 79.53±0.47 | 80498 KB | 2.164±0.156 | 0.152±0.006 |
| dm_multi | 3 | 76.25±0.42 | 79.64±0.31 | 76.73±0.83 | 79.78±0.32 | 79.66±0.29 | 80498 KB | 4.111±0.240 | 0.152±0.006 |
| online_kmeans | 3 | 76.25±0.42 | 78.55±0.40 | 75.94±0.16 | 78.27±0.49 | 78.32±0.37 | 80498 KB | 0.075±0.015 | 0.152±0.006 |
| online_moment | 3 | 76.25±0.42 | 77.23±1.14 | 75.67±1.11 | 77.14±1.09 | 77.21±1.11 | 80498 KB | 1.000±0.000 | 0.152±0.006 |
| online_reservoir | 3 | 76.25±0.42 | 78.81±0.13 | 76.56±0.43 | 78.67±0.19 | 78.76±0.10 | 80498 KB | 7.056±0.434 | 0.152±0.006 |
| online_reservoir_latent | 3 | 76.25±0.42 | 79.66±0.98 | 76.86±0.49 | 79.53±1.08 | 79.66±0.99 | 80498 KB | 0.976±0.032 | 0.152±0.006 |
| kip | 3 | 76.25±0.42 | 79.37±0.69 | 77.13±0.39 | 79.19±0.65 | 78.93±0.41 | 80498 KB | 0.201±0.002 | 0.152±0.006 |
| exstream | 3 | 76.25±0.42 | **81.07±0.38** | **77.83±0.46** | **81.13±0.32** | **81.07±0.34** | 80498 KB | 6.932±0.228 | 0.152±0.006 |
| gcr | 3 | 76.25±0.42 | 80.08±0.65 | 76.49±0.41 | 80.01±0.67 | 80.07±0.65 | 80498 KB | 2.518±0.169 | 0.152±0.006 |

### 5.2 CIFAR-100，IID，80 轮，预算 980（最终 50:1），3 seeds

| 方法 | 种子 | 初始 | 最终 | 在线平均 | 最终 F1 | INT8 | 累计通信 | 方差比 | 对齐误差 |
|---|---|---|---|---|---|---|---|---|---|
| dm | 3 | 28.38±0.24 | 37.07±0.80 | 36.01±0.68 | 36.65±1.21 | 37.07±0.78 | 80498 KB | 2.478±0.068 | 0.220±0.015 |
| dm_multi | 3 | 28.38±0.24 | 37.36±0.88 | 36.26±0.79 | 37.32±1.21 | 37.38±0.90 | 80498 KB | 4.791±0.140 | 0.220±0.015 |
| online_kmeans | 3 | 28.38±0.24 | 32.08±0.60 | 33.03±0.56 | 31.29±0.83 | 31.43±0.20 | 80498 KB | 0.057±0.003 | 0.220±0.015 |
| online_moment | 3 | 28.38±0.24 | 33.44±0.11 | 34.55±0.37 | 31.47±0.11 | 33.14±0.10 | 80498 KB | 1.000±0.000 | 0.220±0.015 |
| online_reservoir | 3 | 28.38±0.24 | 32.73±0.59 | 32.62±0.66 | 32.14±0.53 | 32.69±0.59 | 80498 KB | 9.162±0.278 | 0.220±0.015 |
| online_reservoir_latent | 3 | 28.38±0.24 | 29.52±0.85 | 30.47±0.65 | 28.09±0.80 | 29.54±0.81 | 80498 KB | 0.929±0.053 | 0.220±0.015 |
| kip | 3 | 28.38±0.24 | 35.23±0.20 | 34.84±0.61 | 34.29±0.18 | 34.75±0.46 | 80498 KB | 0.418±0.038 | 0.220±0.015 |
| exstream | 3 | 28.38±0.24 | **41.39±0.25** | **38.54±0.30** | **41.31±0.46** | **41.37±0.28** | 80498 KB | 7.882±0.206 | 0.220±0.015 |
| gcr | 3 | 28.38±0.24 | 35.10±0.52 | 34.32±0.49 | 34.51±0.56 | 35.12±0.55 | 80498 KB | 2.626±0.032 | 0.220±0.015 |

### 5.3 观察

1. **ExStream 在两个数据集上均为最优**：CIFAR-10 81.07、CIFAR-100 41.39，
   比 Initial Base 分别高 4.8 / 13.0 点，也高于其余所有固定预算方法。
2. **CIFAR-10 第二梯队密集**：gcr（80.08）、reservoir_latent（79.66）、
   dm_multi（79.64）、dm（79.51）、kip（79.37）差距在 0.8 点内；
   CIFAR-100 上 DM 系（37.36 / 37.07）明显强于质心类方法（32–33）。
3. **INT8 几乎无损**：所有方法的 INT8 与 FP32 最终精度差 ≤ 0.3 点。
4. **通信完全一致**：主口径均为 full sync，80 轮累计 80,498 KB（约 78.6 MB），
   因此方法间差异只来自代表集构建与蒸馏，不来自通信预算。
5. **方差比呈方法特征**：Moment 恒为 1.000（解析校准）；K-means 强烈收缩
   （C10 0.075 / C100 0.057）；ExStream 与 Reservoir 明显扩张（约 7–9）；
   DM 系约 2.2–4.8；latent reservoir 接近 1。
6. **对齐误差是共同误差源**：CIFAR-10 held-out 相对误差 0.152，
   CIFAR-100 0.220；所有方法共享同一对角仿射近似。

---

## 6. 测试覆盖

当前测试：`python -m pytest vae_streaming/tests -q` → **52 passed, 1 skipped**。

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
| 23 | DM 目标集合统一为 replay+current | `test_dm_target_scope_unified` |
| 24 | 更新预算与上传预算分离 | `test_update_upload_budget_separation` |

标签协议（`tests/test_label_protocol.py`，`label_mode=unlabeled_after_base`）：

| # | 检查 | 测试 |
|---|---|---|
| L1 | 流式真实标签不可访问（标签密封） | `test_1_stream_labels_inaccessible` |
| L2 | 打乱流式真实标签不影响运行结果 | `test_2_stream_label_shuffle_invariance` |
| L3 | 无真实标签也能跑完流水线 | `test_3_runs_without_stream_labels` |
| L4 | 伪标签只来自固定 Initial Base Teacher | `test_4_pseudo_labels_come_from_fixed_teacher` |
| L5 | 预算配置校验 | `test_5_budget_config` |
| L6 | 完整运行的预算统计 | `test_5_budget_from_full_run` |

---

## 7. 审计修复记录

> 历史审计记录保留在此；对应旧实验结果已删除，只保留当前 `sup80v2` 主口径结果。

### 2026-11-16 第一轮审计

| 编号 | 问题 | 结论 | 修复 |
|---|---|---|---|
| P0-1 | Raw / Normalized 特征空间混用 | **真 bug** | `estimate_feature_alignment(..., protocol=...)` 在归一化空间拟合；新增测试 20 |
| P0-2 | `online_replay_dm` 不调用 `StreamingDM` | **真 bug** | 重编码后调用 `StreamingDM.update`；新增测试 23 |
| P0-3 | Patch 同时限制优化位置和上传位置 | **变量混淆** | 新增 `distillation.update_fraction`，与 `patch_fraction` 独立 |
| P0-4 | Base VAE 缓存键不完整 | **真风险** | 缓存签名包含数据集、划分种子、比例、架构与超参；新增测试 21 |
| P1-5 | `reencode` 清空累计统计量 | **设计限制** | 记录 `stats_scope=replay_plus_current` |
| P1-6 | Encoder 变化后云端没有显式补偿 | **非 bug** | 每轮在新特征空间上重训即是补偿 |
| P1-7 | 断点签名未覆盖主要参数 | **真风险** | 改为完整配置哈希；新增测试 22 |
| P1-8 | Alignment 训练/测试误差未分离 | **部分成立** | 分别报告 fit 与 held-out 类中心误差 |
| P2-9 | 计时、INT8、恢复统计 | **部分成立** | 拆分 eval / test_int8 / wall 计时；恢复累计 wall |

### 2026-11-17 第二轮审计（长流协议）

| 编号 | 问题 | 结论 | 修复 |
|---|---|---|---|
| P0-a | `update_every_rounds` 末轮处理 | **权衡后改为 flush** | 末轮用余数样本补一次更新（`vae.final_flush=True`） |
| P0-b | DM 代表集重初始化与目标跳变 | **真问题** | DM 改用 `diagonal_affine` 继承；`exact_target=True` |
| P0-c | 三方法历史信息量不一致 | **真问题** | 新增 `stream.history_mode=full|replay` 消融口径 |
| P1-1 | 方法间 VAE 随机轨迹可能不同 | **风险** | 更新包在 `fork_rng` 内独立播种；逐轮记录 `encoder_sha1` |
| P1-2 | Base 缓存签名不含 base 索引 | **加固** | 签名加入 base 索引 SHA1 与样本数 |
| P1-3 | `stats_scope` 日志误标 | **真问题** | 跨轮维护 `stats_origin` |
| P1-4 | `n_train_used` 用 `stats.n_seen` | **真问题** | 改为实际累计已见并独立记录 `effective_stats_samples` |
| P1-5 | resume 仍做 Base 初始化 | **真问题** | 先探测断点，命中则跳过初始化 |
| P1-6 | 锚点抽样非严格分层 | **真问题** | `_stratified_anchor_indices` 按类配额抽样 |
| P1-7 | `test_domain()` 标签未使用 | **加固** | 评估显式使用返回的 `dom_y` |
| P2-1 | `partial` 判定 | **真问题** | `completed_rounds < num_rounds` |
| P2-2 | VAE checkpoint 覆盖 | **真问题** | 只在 `do_update` 时写版本文件；按 `partial` 命名 |
| P2-3 | `anchor_count=0` 边界 | **真问题** | 显式跳过对齐，使用 identity |
| P2-4 | `expected_seen_final` 忽略提前停止 | **真问题** | 按实际完成轮次计算 |

### 2026-11-17 第三轮审计（Replay 去重与统计口径）

| 编号 | 问题 | 结论 | 修复 |
|---|---|---|---|
| A1 | Pending 与 Replay 重复 | **真问题** | Replay 条目记录轮次；VAE 更新用 `exclude_rounds=pending_rounds` |
| A2 | 末轮余数未训练 | **真问题** | 末轮强制 flush |
| A3 | 分层锚点配额不回补 | **真问题**（主实验不触发） | water-filling 重新分配剩余名额 |
| A4 | 三方法历史信息不同 | **设计差异** | 保留 `--history-mode replay` 消融 |
| A5 | DM 未用累计统计 | **属实** | DM 定义为 Replay+Current 均值匹配；统计记忆只归属 Moment |
| A6 | 随机特征 DM 实现 | **不成立** | 固定 seed、`requires_grad=False`、`eval()` |
| A7 | NumPy / Python RNG 未隔离 | **不成立** | VAE 路径只用 torch RNG，全程在 `fork_rng` 内 |
| A8 | Patch 优化/上传比例耦合 | **潜在问题** | 两者各自调用 `select_patch_indices` |
| A9 | 部分运行写 `vae_final.pt` | **真问题** | 按 `partial` 命名 `vae_partial.pt` |
| A10 | `edge_total_sec` 混入非边缘耗时 | **真问题** | 拆分 edge/eval/cloud/wall 计时 |

### 2026-11-26 第四轮审计（主口径 80 轮）

| 编号 | 问题 | 修复 |
|---|---|---|
| A1 | GCR facility-location `cov` 初值为 `-inf` | 改为 0；首点按标准增益选取。gcr 已按 `sup80v2` 整体重跑 |
| A2 | label_shift 历史类分组包含当前块 | 历史组显式 `setdiff1d` 排除当前块 |
| A3 | 有监督流写入 Replay 未标 `source="stream"` | 写入时标记 `source/label_source`；`ReplayBuffer` 增加 `allow_stream_ground_truth` 区分协议 |
| A4 | GCR 全池 n×n 相似度矩阵 | 按 1024 行分块计算；budget=9800 时不再一次性构造 6.1GB 矩阵 |
| A5 | class-balanced 每类独立取整 | 代码不变（改动会改变 Base 集合）；文档明确实际配额（C10 977–983，C100 97–99） |
| B6 | Patch 同步把全部槽位版本标成最新 | 按消息携带的逐槽 `versions` 写入；未改动槽位仅在版本变化时更新 |
| B7 | streaming_dm margin 权重无上界、分子/分母位置集不一致 | `margin_max_weight` 默认 10；mask 与分母统一为当前训练位置 |
| B8 | `evaluate_vae` 用 sum 后除样本数、mu/z 口径不一致 | 改为确定性 `mu` 评估；rec/kl 按样本均值口径统计 |
| B9 | `prev_encoder.module` 假设、单锚点顺序对半切 | 用 `getattr(prev_encoder, "module", prev_encoder)`；回退按类内交替 `[::2]` |
| B10 | 每步 H2D、multi_branch 重复建模、state_dict 冗余 | 小同步集整体驻留 GPU；初始一致性校验只做一次；去掉重复的 `images/labels` 字段 |
| B11 | reencode + patch 组合语义不清 | 显式拒绝该组合（`ValueError`），不再静默产生不一致状态 |

---

## 8. 已知限制

1. **20% Base 是共同瓶颈**：初始 VAE 只见过约 9,800 张图，特征质量明显弱于
   第一阶段 49k 全量 VAE。增加轮数只能改善增量维护，不能补齐编码器上限。
2. **初始代表集使用 Base 标签**：Base 的标签在因果协议内可访问；但这不是无监督方法。
3. **对角仿射是近似**：held-out 相对误差 C10 0.152 / C100 0.220；
   不能把对齐后的历史统计量宣称为真实历史图像在新 Encoder 下的精确分布。
4. **主口径是 full sync**：patch 10% 仅保留实现，未进入主表；
   固定预算 patch 需要覆盖全部位置后才适合部署。
5. **云端验证集**：使用 1,000 张真实验证图像特征，不参与 VAE/DM/Memory 更新，
   也不使用测试集调参。
6. **归一化冻结在 Base**：编码器漂移后特征尺度会变化，但边缘与云端保持一致；
   这是有意的因果选择。
7. **离线参考非因果**：`run_offline_reference.py` 使用全量 49k，只能估计在线差距，
   不能当作同等数据可访问条件的可部署方法。
8. **3 种子共享同一数据流与 Base 索引**：种子方差只覆盖 VAE 初始化、DM 随机特征
   与云端训练；方法差距小于约 1 点时不能当作定论。
9. **GCR 显存随候选池平方增长**：当前池为 4×980=3,920，分块计算后可用；
   若把预算扩到 9,800，全池矩阵需约 6.1GB，仍属扩展性瓶颈。
10. **类平衡配额不严格相等**：C10 每类 977–983、C100 每类 97–99；
    报告“每类 20%”时需注明实际取整。
11. **Label Shift / Covariate Shift 无最新结果**：代码与分组修复已合入，
    但当前只保留 IID 主表；引用这两个场景前需重新运行。
12. **reencode + patch 组合被显式拒绝**：两者对代表集语义冲突，
    如需该组合必须先定义新的协议。
