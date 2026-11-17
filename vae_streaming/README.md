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

### 2.0 长流小批量协议

小批量、多轮次时按下面两个参数构造，避免“每轮 100 张却更新 150 步”的失配：

- `stream.class_quota`：每类每轮固定张数。Base 自动吸收类别总数差异
  （CIFAR-10 各类 967–995 张，合计 9,800 = 20%），之后每轮严格 100 张。
  392 轮 × 10 张/类正好耗尽每类剩余 3,920 张。
- `vae.update_every_rounds`：累积若干轮后再做一次在线 VAE 更新。
  取 10 时，每 10 轮累积约 1,000 张，更新仍为 150 步；
  其余轮次只做编码、统计、代表集与云端同步，Encoder 版本保持不变。
  **末轮强制 flush**：392 轮不能被 10 整除，最后一轮用余数样本（约 200 张）
  补一次小 batch 更新，逐轮记录 `vae.final_flush=True`，保证全部样本都进入 VAE 梯度。
  **Replay 与 Pending 互斥**：Replay 条目带轮次标签，更新时用
  `replay.get(exclude_rounds=pending_rounds)` 排除仍在 pending 中的轮次，
  避免同一批样本同时作为“当前数据”和“replay”进入同一个 batch。
  优化子集与上传子集分别按 `update_fraction` / `patch_fraction` 独立抽样。

**锚点抽样**：`_stratified_anchor_indices` 按类别容量做 water-filling，
某类样本不足时把剩余名额重新分配给仍有余量的类别，实际锚点数不再少于
`alignment.anchor_count`（除非 Replay 总量本身不足）。

**时间与产物口径**：`edge_train_sec` 只含 VAE 训练（初始 + 在线），
`edge_total_sec` 为边缘侧计算（训练 / 编码 / 对齐 / 统计 / 蒸馏 / 同步），
`eval_total_sec`、`cloud_total_sec`、`wall_total_sec` 各自独立统计。
部分运行（`--stop-after` 或中断）保存 `vae_partial.pt`，完整运行才是 `vae_final.pt`。

**三种方法的历史预算口径**（`stream.history_mode`）：

- `full`（默认）：按各方法自身机制保留历史 —— K-means 用固定预算的流式质心，
  Moment 用 Chan 累计统计（base + 全部历史轮），DM 用固定 1,000 的 Replay + 当前轮。
- `replay`：统一为“固定 1,000 Replay + 当前轮”重建统计，用于把 Moment 的
  历史信息量对齐到 DM / K-means 的固定预算，作为方法对比的消融口径。
  该开关只改变统计来源；K-means 的 coreset 与 DM 的目标集合本来就是固定预算，
  因此它们的数值不变。

**DM 口径统一**：`dm`（原 `replay_dm_randfeat`，其他 DM 变体删除后简化）采用与另两种方法相同的
`alignment=diagonal_affine`（代表集跨版本继承，不再每轮 K-means 重建），
并设置 `distillation.exact_target=True`：每一轮都用当前 Encoder 精确重编码
Replay + 当前轮作为 DM 目标，更新轮与非更新轮的目标集合构造方式完全一致。
逐轮记录 `distill.target_scope`（`replay_plus_current`）与 `distill.n_target`。

**随机性隔离**：在线更新（含 torchvision 增强）在 `torch.random.fork_rng`
的独立随机流中完成，按 `seed + 1e6 + 7919·round` 独立播种；全局 RNG 不被推进。
因此三种方法从 Initial Base VAE 出发得到逐位相同的 Encoder 轨迹，
逐轮用 `vae.encoder_sha1` 可复核。

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

### 3.4 Streaming DM（原版 DM：固定随机特征空间均值匹配）

```
f = RandomMLP(1024 → 2048 → 1024)   # 固定随机初始化，参数冻结
L_SDM = ||mean(f(real_z)) − mean(f(syn_z))||² + λ_a·||S − S_aligned_prev||²
```

- 与第一阶段 `BatchDM` 使用同一个目标，只是真实特征来自**当前版本**的统计/回放数据。
- 逐类独立 `nn.Parameter` + 独立 AdamW；每类固定 `updates_per_class=20`。
- 全量同步模式优化全部代表；固定预算 patch 模式只优化本轮允许更新的位置，
  其余位置严格保持对齐后的旧值。
- 不包含方差/协方差/diversity 项；`λ_a` 只是稳定性锚点。

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
# 单元测试（41 项，含完整流水线断点续跑与协议审计回归）
python -m pytest vae_streaming/tests/test_streaming_protocol.py -q

# CIFAR-100 IID 主实验（单方法）
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar100.yaml \
  --methods dm --seeds 42

# 全方法对照（IID）
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar100.yaml \
  --methods online_kmeans,online_moment,dm \
  --seeds 42,43,44

# 长流小批量：392 轮，每类每轮 10 张，每累积 10 轮做一次在线更新
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10.yaml \
  --methods online_kmeans,online_moment,dm \
  --seeds 42 --rounds 392 --class-quota 10 --update-every 10 --tag r392p2

# 历史预算对齐消融（Moment 改为 replay+current 固定预算）
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10.yaml \
  --methods online_moment --seeds 42 --rounds 392 \
  --class-quota 10 --update-every 10 --history-mode replay --tag r392p2rep

# 长流报表与图（按轮数/标签过滤，避免混进 5 轮主表）
python -m vae_streaming.experiments.make_report --dataset cifar10 \
  --rounds 392 --tag _r392p2 --out vae_streaming/results/cifar10/main_table_r392p2.md
python -m vae_streaming.experiments.make_figures --dataset cifar10 \
  --rounds 392 --tag _r392p2 --out-suffix r392p2

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
| VAE + Nonlinear DM（Replay） | **41.68±0.57** | **40.24±0.44** | 41.82±0.57 | 5.96 MB | 0.48 |
| VAE + Moment Matching | 39.46±0.52 | 38.12±0.42 | 38.33 | 5.96 MB | 1.00 |
| VAE + K-means | 36.84±0.84 | 35.98±0.53 | 36.46 | 5.96 MB | 0.29 |

观察：

1. **Replay + 原版 DM 最强**：3 seeds 最终 41.68±0.57，比 K-means 高 4.8 点，
   比 Moment Matching 高 2.2 点。
2. Moment Matching 的方差比保持 1.00；K-means 明显收缩类内方差（0.29），
   DM 居中（0.48）。
3. 原版 DM 的精确性来自 Replay 重编码 + 初始化 + DM 三件事；只做重编码与
   K-means 初始化而不跑 DM 的对照方法已删除，其历史数字不再引用。

### 5.1b CIFAR-10，IID，5 轮，预算 980（50:1），3 seeds

CIFAR-10 的 20% Base 已能训练出较强编码器，初始精度 **76.08±0.58**，
方法间差距约 2 点，呈现明显的**天花板效应**。

| 方法 | 最终 | 在线平均 | 累计通信 | 方差比 |
|---|---|---|---|---|
| VAE + Nonlinear DM（Replay） | **78.12±0.50** | **77.26±0.16** | 5.96 MB | 0.69 |
| VAE + Moment Matching | 76.74±0.56 | 75.16±0.41 | 5.96 MB | 1.00 |
| VAE + K-means | 76.23±1.46 | 74.44±0.42 | 5.96 MB | 0.40 |

观察：

1. **CIFAR-10 上 DM 仍领先**：最终 78.12，比 Moment 高 1.4 点、比 K-means 高 1.9 点，
   但绝对差距小于 CIFAR-100。
2. K-means 在 C10 上同样收缩方差（0.40），精度也最低。

### 5.2 Label Shift / Covariate Shift（M3 已跑）

分布变化场景目前只有 Replay + 原版 DM 的结果（其余两方法待跑）：

| 数据集 | 场景 | 最终 | 当前类/当前域 | 历史类/原域 |
|---|---|---|---|---|
| CIFAR-100 | Label Shift | 41.41±0.09 | 49.43±0.33 | 39.40±0.05 |
| CIFAR-100 | Covariate Shift | 34.12±0.43 | 33.89±0.90 | — |
| CIFAR-10 | Label Shift | 78.51±0.42 | 74.17±1.57 | 79.59±0.30 |
| CIFAR-10 | Covariate Shift | 74.22±0.63 | 69.79±0.48 | — |

### 5.4 Replay + 原版 DM 明细（CIFAR-100 IID，3 seeds，预算 980）

| 方法 | 最终 | 在线平均 | 累计通信 |
|---|---|---|---|
| Replay + 原版 DM（重编码 + 初始化 + DM） | **41.68±0.57** | **40.24±0.44** | 5.96 MB |

Replay 重编码提供当前版本的训练特征，DM 在此之上做逐类均值匹配。
跨版本对齐后旧特征上的 DM 对照（`Affine + DM`）已随方法删除，不再引用。

### 5.5 非因果离线参考

| 参考 | 方法 | 编码器 | 归一化 | 最终 |
|---|---|---|---|---|
| stage1_kmeans | K-means | 第一阶段 49k 全量 VAE | 全量训练集重估 | 58.31 |
| onlineenc_kmeans | K-means | 本阶段最终因果 Encoder（v5） | 与在线方法共享 Base 协议 | 41.23 |

- 与第一阶段 49k 预训练 VAE 相比，当前 20% Base 编码器在同类基线方法上低约 17 点，
  说明**当前最大瓶颈是编码器质量**，而不是蒸馏或同步机制。
- 离线 Batch DM 参考需要按原版 DM 重跑；历史 mean+var 版本的结果已删除。

### 5.6 计算与通信

每个 JSON 记录：每轮 VAE 时间、编码时间、对齐时间、DM 时间、同步字节、
云端训练时间、峰值显存、Replay 实际字节、累计上传字节。
本机（RTX 4090 D）5 轮单方法边缘总耗时约 30–100 s，峰值显存 <1 GB；
云端 6 次训练合计约 2–3 s。完整同步累计约 6.1 MB，patch 10% 约 1.57 MB（理论字节）。

图（`figures/`，均由当前三方法结果生成）：

- `cifar100_accuracy_iid.png`、`cifar100_communication_iid.png`、
  `cifar100_drift_alignment_iid.png`
- `cifar10_accuracy_iid.png`、`cifar10_communication_iid.png`、
  `cifar10_drift_alignment_iid.png`

Label/Covariate Shift 的图在结果重跑后再生成。

### 5.7 变量分离实验（CIFAR-100 IID，3 seeds）

原先把“重编码 / 代表集重建 / DM 更新”拆成独立变量做过对照；
只重编码 + KMeans 的 Replay-Reinit 与 Affine + DM 两个方法已删除，
对应历史数字不再引用。

| 方法 | 最终 | 在线平均 | 累计通信 |
|---|---|---|---|
| Replay + 原版 DM（重编码 + 初始化 + DM） | **41.68±0.57** | **40.24±0.44** | 5.96 MB |

1. **DM 的价值依赖代表特征已经是当前版本**：Replay 重编码后再跑原版 DM 是当前唯一保留的配置。
2. 更新/上传预算分离（patch / partial）的消融结果随旧目标一并删除，
   需要按原版 DM 重跑后再补。

### 5.8 原版 DM（固定随机 MLP 特征空间 + 均值匹配）

原版 DM 是当前唯一保留的蒸馏目标（两阶段相同）：在 1024 维特征上再套一个
**固定随机 MLP**，参数完全冻结，只让梯度穿过 synthetic 代表：

```python
feature_net = RandomMLP(input_dim=1024)
feature_net.eval()
for p in feature_net.parameters():
    p.requires_grad_(False)

real_feat = feature_net(real_z).detach()
syn_feat = feature_net(synthetic_z)
loss = (real_feat.mean(0) - syn_feat.mean(0)).pow(2).mean()
```

代码：`random_feature.py`；阶段一 `BatchDM`、阶段二 `StreamingDM` 都只走这一条路径。
注意：这不是像素空间 DM（没有 decoder-in-the-loop、没有可微增强）。

Stage 1（`vae_bdm`，10:1，3 个蒸馏种子 × 3 个云端种子）：

| 数据集 | 原版 DM | Full | K-means + Moment |
|---|---|---|---|
| CIFAR-100 | 59.61±0.25 | 61.56±0.15 | **60.85±0.07** |
| CIFAR-10 | 90.06±0.06 | 90.19±0.19 | 90.05±0.09 |

Stage 2（CIFAR-100 IID，3 seeds，full sync，100% 更新）：

| 方法 | 最终 | 在线平均 | 组/单图方差比 |
|---|---|---|---|
| Replay + 原版 DM | **41.68±0.57** | **40.24±0.44** | 0.48 |

解读：

1. 固定随机 MLP 在当前 1024 维 VAE 特征上，阶段一没有超过 Moment Calibration
   （C100 低约 1.2 点，C10 在噪声内持平）；阶段二则明显领先（41.68 vs 39.46）。
2. 原版 DM 的收益主要来自**训练好的特征提取器 + 像素增强 + 较大 IPC**；
   当前阶段使用固定随机 MLP 是为了保持方法学一致，后续换用更强的特征提取器后
   该方法才有比较意义。
3. 纯均值匹配会收缩类内方差（stage 2 组/单图方差比 0.48）；
   该比值只作为诊断指标记录，不参与训练目标。
4. 阶段二的 real features：`dm` 用 replay + current，
   覆盖当前版本下的历史与新增样本。

### 5.9 长流 392 轮（CIFAR-10，class_quota=10，update_every=10，tag r392p2）

单种子 42；392 轮恰好耗尽每类 3,920 张。末轮 flush（`final_flush=True`），
`vae_pending_leftover=0`，`n_train_used=49,000`；四种方法 Encoder 轨迹逐位相同
（r391 `encoder_sha1=de208c5776faa5e9`），累计通信 390,564 KB（993.8 KB/轮，INT8 full_sync）。

| 方法 | 初始 | 最终 | 在线平均 | 最终 F1 | INT8 | 方差比 | 对齐误差 |
|---|---|---|---|---|---|---|---|
| online_kmeans | 75.97 | 74.09 | 74.60 | 73.74 | 73.48 | 0.033 | 0.129 |
| online_moment | 75.97 | **77.83** | 75.56 | **77.61** | **77.83** | 1.000 | 0.129 |
| dm | 75.97 | 76.88 | **75.76** | 76.75 | 76.87 | 2.756 | 0.129 |
| online_moment（history_mode=replay 消融） | 75.97 | 77.10 | 75.71 | 77.03 | 77.04 | — | — |

时间拆账（秒）：边缘计算 227–265，评估 94–102，云端训练 225–251，墙钟 601–643；
非更新轮约 0.9 s，更新轮约 5 s。

1. Moment 最终最好，DM 在线平均最高，K-means 低于初始（74.09 < 75.97），
   需要单独排查 coreset / 质心更新。
2. 消融：固定预算（replay+current，1,100 样本）比累计统计低 0.73，
   Moment 的收益部分来自历史信息量，论文必须写清口径。
3. 与修改前 `r392u10`（K-means 78.48 / Moment 78.37 / DM 81.11）差异较大：
   DM 由 `reencode` 改为 `diagonal_affine` 可解释一部分，K-means 的差异尚无定论。
   两组数字口径不同，不能混用；需要单变量对照（关闭 replay 去重、关闭末轮 flush）定位。

---

## 6. 测试覆盖（任务书 19 项 + 6 项审计回归）

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
| 24 | 完整流水线断点恢复 exact 对齐 | `test_full_pipeline_resume` |
| 25 | 更新预算与上传预算分离 | `test_update_upload_budget_separation` |

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
| 追加 | Reencode 与蒸馏未拆分 | **方法学问题**：重编码 + KMeans 被当成 Replay+DM | 拆出“只重编码+初始化”的对照；该对照后随方法收敛删除，只保留 `dm`（重编码 + 初始化 + 原版 DM） |
| 追加 | 样本计数口径单一 | **审计问题**：无法区分“已见样本/有效统计/Replay” | 逐轮记录 `n_cumulative_seen / n_effective_stats / n_replay`；汇总写 `cumulative_seen_samples / effective_stats_samples / replay_samples_final` |
| 追加 | 对齐只有误差，没有前/后对比 | **验证不足** | 记录 `no_align_heldout_mse`、`improvement_heldout`、`class_center_improvement`，直接验证对齐相对“不对齐”的增益 |

修复后 `vae_streaming` 测试 28 项全过；受影响结果已整体重跑（旧结果保留在
`results/_stale_pre_audit/`）。

### 2026-11-17 第二轮审计（长流 392 轮）

针对长流协议逐条复核，结论与修复：

| 编号 | 问题 | 结论 | 修复 |
|---|---|---|---|
| P0-a | `update_every_rounds` 末轮处理 | **权衡后改为 flush**：392 是唯一精确耗尽每类 3,920 张的轮数，改为 390 会让 200 张永不进入流 | 末轮用余数样本补一次小 batch 更新（`vae.final_flush=True`）；`vae_pending_leftover` 只记录异常残留 |
| P0-b | DM 每轮重初始化代表集 + 目标随版本跳变 | **真问题**：`reencode` 分支在更新轮用 K-means 重建代表集并改用 replay+current，非更新轮仅用 current | DM 改用 `diagonal_affine` 继承代表集；`exact_target=True` 每轮用当前 Encoder 精确重编码 replay+current；逐轮记录 `target_scope / n_target` |
| P0-c | 三方法历史信息量不一致 | **真问题**：同轮 Moment `stats.n_seen≈26.5k`、DM `≈1.9k` | 新增 `stream.history_mode=full|replay` 消融口径；`replay` 用固定 1,000 replay+current 重建统计 |
| P1-1 | 方法间 VAE 随机轨迹可能不同 | **风险**：在线更新消耗全局 RNG（torchvision 增强） | 更新包在 `torch.random.fork_rng` 内，按 `seed+1e6+7919·round` 独立播种；记录 `vae.encoder_sha1`，三方法逐轮哈希一致 |
| P1-2 | Base 缓存签名不含 base 索引 | **加固**：缺失 `base_indices_sha1 / n_base`，`cache_version` 升到 3 | 签名加入 base 索引 SHA1 与样本数 |
| P1-3 | `stats_scope` 日志误标 | **真问题**：未更新的轮次直接标 `aligned_full_history`，忽略历史是否曾被 reset | 跨轮维护 `stats_origin`，只在真正 reset 时降级；resume 时继承上一轮 |
| P1-4 | `n_train_used` 用 `stats.n_seen` | **真问题**：可能只含 replay+current | `n_train_used` 改为实际累计已见；独立记录 `effective_stats_samples`；删除重复 `n_replay` 键 |
| P1-5 | resume 仍做 Base 初始化 | **真问题**：先训练 Base/初始同步/初始云端训练再恢复 | 先探测断点，命中则只恢复状态，跳过全部初始化 |
| P1-6 | 锚点抽样非严格分层 | **真问题**：类内连续存储时 `linspace` 可能漏类 | `_stratified_anchor_indices` 按类配额抽样；`n_anchors=0` 时跳过漂移/对齐 |
| P1-7 | `test_domain()` 标签未使用 | **加固**（`test_domain` 目前恒等返回 `test_labels`） | 评估显式使用返回的 `dom_y` |
| P2-1 | `partial` 判定 | **真问题**：停在最后一轮也可能标记未完成 | `completed_rounds < num_rounds` |
| P2-2 | VAE checkpoint 覆盖 | **真问题**：未更新轮用同一版本号反复写 | 只在 `do_update` 时写 `vae_v{version}.pt`；结束按 `partial` 写 `vae_final.pt` / `vae_partial.pt` |
| P2-3 | `anchor_count=0` 边界 | **真问题**：空锚点导致编码失败 | 显式跳过对齐，使用 identity + 0 漂移 |
| P2-4 | `expected_seen_final` 忽略提前停止 | **真问题**：始终按全部轮次计算 | 按实际完成轮次计算 |

> 未采纳：原清单中“随机特征 DM 未真正使用固定 MLP / 仅匹配均值”——已核对
> `random_feature.py` 与 `streaming_dm.py`，参数 `requires_grad=False`、
> 损失为逐类均值匹配 + 小权重锚点项，结论不成立。

### 2026-11-17 第三轮审计（Replay 去重与统计口径）

| 编号 | 问题 | 结论 | 修复 |
|---|---|---|---|
| A1 | Pending 与 Replay 重复 | **真问题**：更新轮 replay=第 0..t-1 轮、pending=第 t-9..t 轮，交集被重复计权 | Replay 条目记录轮次；VAE 更新用 `exclude_rounds=pending_rounds`，DM/锚点仍用完整 Replay |
| A2 | 末轮余数未训练 | **真问题**：392%10=2，最后 200 张只进统计/同步 | 末轮强制 flush（`vae.final_flush=True`），`vae_pending_leftover` 归零 |
| A3 | 分层锚点配额不回补 | **真问题**（主实验不触发）：容量不足时直接截断 | water-filling 重新分配剩余名额，锚点数达到 `anchor_count` |
| A4 | 三方法历史信息不同 | **设计差异**：Moment 累计统计信息量与 DM/K-means 固定预算不同 | 保留 `--history-mode replay` 消融；论文按“整套算法”表述，不宣称同预算 |
| A5 | DM 未用累计统计 | **属实** | 不新增变体：DM 定义为 Replay+Current 随机特征均值匹配，Statistical Memory 只归属 Moment |
| A6 | 随机特征 DM 实现 | **不成立**：固定 seed 0 的 `RandomMLP`，`requires_grad=False` + `eval()`，`no_grad` 下算目标均值 | 论文注明随机网络跨种子固定 |
| A7 | NumPy / Python RNG 未隔离 | **不成立**：VAE 路径只用 torch RNG，全部在 `fork_rng` 内；各组件用独立 `RandomState` 并随断点持久化 | 无需修改 |
| A8 | Patch 优化/上传比例耦合 | **潜在问题**：`update_fraction` 与 `patch_fraction` 不同时仍复用 `patch_idx` | 两者各自调用 `select_patch_indices`（现有配置不触发） |
| A9 | 部分运行写 `vae_final.pt` | **真问题**：中断权重可能被误认作完整结果 | 按 `partial` 命名 `vae_partial.pt` |
| A10 | `edge_total_sec` 混入非边缘耗时 | **真问题**：原值是整段墙钟（含云端训练/评估） | 拆为 `edge_train_sec` / `edge_total_sec` / `eval_total_sec` / `cloud_total_sec` / `wall_total_sec` |

> 验证：3 轮 `update_every=2` smoke 中 r2 出现 `final_flush=True`、`n_vae_batch=100`、
> `vae_pending_leftover=0`、`n_anchors=200`；DM `target_scope=replay_plus_current`、
> `n_target=1100`；`edge_total_sec=12.1` 与 `wall_total_sec=12.9`、`cloud=0.04`、`eval=0.7` 分离。

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
