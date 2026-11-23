# CIFAR-10 无标签流式实验报告（unlabeled_after_base · Multi-MLP DM）

- 配置：`vae_streaming/configs/cifar10_unlabeled_multi80_u2.yaml`
- 方法：`dm_multi`（Multi-MLP DM，3 个冻结随机 MLP）+ Online VAE + Incremental Cloud
- 结果：`vae_streaming/results/cifar10/iid/dm_multi_u2_seed42.json`（tag `u2`，未覆盖任何旧结果）
- 曲线：`vae_streaming/figures/dm_multi_u2_seed42_curve.png`
- 标签审计：`LABEL_PROTOCOL_AUDIT.md`
- 种子：42；`full_sync` + int8；`diagonal_affine`；`select_best_on_val = false`

## 1. 协议与预算（实测）

| 项目 | 设置 | 实测 |
|---|---|---|
| Initial Base | 9,800（980/类，有真实标签） | 9,800 |
| Streaming | 39,200，80 轮 × 490，无真实标签 | 39,200 |
| val / test | 1,000 / 10,000（仅离线评估） | 1,000 / 10,000 |
| VAE 更新 | 每 2 轮一次 × 150 步 | 40 次 / 6,000 步 |
| DM 更新 | 每轮一次 | 80 次 |
| Raw Replay | 1,000（只存 Base 真实标签样本） | 1,000（100/类，流式写入 0） |
| 蒸馏 Reservoir | 980，pseudo-balanced | 980（全程恒定） |
| Base / Pseudo（DM 目标） | — | 1,000 + 238~280（均值 257.8） |
| 单轮通信 | full_sync + int8 | 993.8 KB |
| 累计上传 | — | 82.43 MB（81 条消息） |
| 数据流 SHA1 | 固定索引排列 | `stream a914f9bb4998e17e` / `base 9d36421552fc6fdf` |

> 新协议的 Base 与旧 `class_quota=10 / 392 轮` 实验的 Base 采样规则不同（类别样本量
> 不整除导致每类 base 数略有差异），因此**跨协议指标不可直接比较**。

## 2. 主要指标

| 指标 | 数值 |
|---|---|
| Initial Accuracy（Base VAE，测试集） | **76.27%** |
| Average Online Accuracy（80 轮） | **63.78%** |
| Final Accuracy（第 80 轮） | **60.64%** |
| Average Online Macro-F1 | **63.59%** |
| Final Macro-F1 | **60.19%** |
| Final int8 测试精度 | 60.61% |
| 第 1 轮（Online 起点） | 75.86% / F1 75.80% |

80 轮 Accuracy / Macro-F1 曲线见
[dm_multi_u2_seed42_curve.png](file:///root/autodl-tmp/diffusion%20distillation/vae_streaming/figures/dm_multi_u2_seed42_curve.png)。

每 5 轮抽样（完整逐轮数据在结果 JSON 的 `rounds[]`）：

| 轮 | Acc | F1 | 伪标签接受率 | Base/Pseudo 目标数 | DM loss |
|---|---|---|---|---|---|
| 1 | 75.86 | 75.80 | 53.5% | 1000 / 262 | 5.1e-4 |
| 5 | 66.39 | 66.13 | 53.3% | 1000 / 261 | 3.8e-4 |
| 11 | 66.22 | 65.90 | 51.6% | 1000 / 253 | 7.3e-4 |
| 21 | 64.27 | 63.52 | 50.8% | 1000 / 249 | 1.2e-3 |
| 31 | 66.11 | 65.60 | 50.8% | 1000 / 249 | 8.4e-4 |
| 41 | 63.74 | 63.87 | 53.7% | 1000 / 263 | 6.7e-4 |
| 51 | 62.36 | 62.16 | 55.3% | 1000 / 271 | 6.0e-4 |
| 61 | 58.88 | 58.06 | 51.4% | 1000 / 252 | 1.2e-3 |
| 71 | 60.92 | 59.95 | 51.4% | 1000 / 252 | 6.1e-4 |
| 80 | 60.64 | 60.19 | 51.8% | 1000 / 254 | 6.2e-4 |

## 3. 40 次 Encoder 更新 SHA1

按更新轮次顺序（第 2,4,…,80 轮）：

```
0a9bd0501dbb51d4 a352eb718cd43654 903f81461ecc6088 501508fe19aa3b54
9d8a6c0fd8bc02c8 f53f23e0334725f3 2c4581f1e5198060 b0b893d62729252f
4176e32c90eff614 9377ff6eeca92026 598b7ce486b463ec a0412d2466b9c611
8aad3b1c932ec8d1 84cce31347ecd55c ab295f1e3972ca38 978447cdb80567e0
f9ab45b98b1d350e 821dfe6f833388c0 24e70a584afd5e32 371e6ea1172cba92
18b0b36f03f36815 5ce13e12bcdf333b 85e49f59a3c167b3 08acf46b219e2060
2debdb59dbaf211d f9461b5f35a03556 3911dc5a3bb725b7 bfeaef55f68bd356
c51009179e01b617 a4e0af7b5a8ddfbf ce075e2cabf18eeb 861536d73a960280
c8b481d9f4764b49 d2f867cb4c4ab0a0 1f2df3704668c42b 907ad1ab849026ba
884a72a71df6bcd7 b281fb8aa5643528 adacef3ca1d0f2a7 a450a63a284d62fd
```

40 个 SHA1 互不相同；非更新轮沿用上一版本（`vae_updated=false`，align 为 identity）。

## 4. Feature Alignment

| 指标 | 均值 | 最大 |
|---|---|---|
| Anchor Drift（relative） | 0.1020 | 0.3013 |
| Relative Alignment Error（held-out） | 0.0906 | 0.2285 |
| Held-out MSE | 0.1391 | — |
| Class-center Error（仅 Base Replay，40 次更新） | 0.0939 | 0.1521 |
| `clamped_frac` | 0 | 0 |

- 非更新轮无新 Encoder 版本，alignment 为 identity，`class_center_err = NaN` 属预期
  （不参与训练，仅记录）。
- 类中心误差仅用 Base Replay 的真实标签计算；流式真实标签未参与。

## 5. 伪标签（固定 Initial Base Teacher）

| 指标 | 数值 |
|---|---|
| 来源 | `initial_teacher_pseudo`，`teacher_version=0` 全程固定 |
| 阈值 | 0.9（未按测试精度调节） |
| 总接受 | 20,625 / 39,200（52.6%） |
| 每轮接受率 | 48.6% ~ 57.1%（均值 52.6%） |
| 每轮接受样本 | 238 ~ 280（均值 257.8） |
| 各类累计接受 | [2466, 2878, 1839, **31**, 2346, 968, 2450, 2475, 2432, 2740] |
| 空类别回退 | 55 个「类别-轮次」使用 `base_replay_only_fallback`（未用其他类别填充） |

类别 3 的 Teacher 高置信预测极少（全程 31 个），该类别在 DM 中始终回退到 Base Replay
特征作为匹配目标，符合协议要求。

## 6. DM 优化与数值检查

- 平均损失 5.4e-4，最小 1.3e-4，最大 1.8e-3；全部有限，无 NaN / Inf。
- 每轮 `mean_loss`、`mean_anchor` 与逐类 `n_base_replay / n_pseudo / pseudo_fraction /
  target_mode` 均写入 `rounds[t].distill`。
- 三个随机 MLP 全程冻结，同种子下映射可复现。

## 7. 耗时与资源

| 项目 | 数值 |
|---|---|
| Online VAE 训练 | 158.23 s（40 × 150 步） |
| DM 蒸馏 | 15.41 s（80 次） |
| 云端增量训练 | 46.90 s（80 × 200 步） |
| 边缘总耗时 | 264.28 s |
| 端到端 wall time | 268.40 s |
| 累计上传 | 82,429,705 B（82.43 MB，81 条消息） |
| Peak GPU Memory | 1.159 GB |

## 8. 标签泄漏测试

`python -m pytest vae_streaming/tests/test_label_protocol.py -q` → **6 passed**。

- Test 1 流式标签不可访问：PASS
- Test 2 流式真实标签置乱后轨迹完全不变（含 `stream_sha1` / Teacher 指纹 /
  逐轮 `encoder_sha1` / `memory_sha1` / `cloud_param_sha1` / acc / F1 / DM loss）：PASS
- Test 3 删除流式标签字段跑完整流程：PASS
- Test 4 `pseudo_label == teacher_prediction`：PASS
- Test 5 预算（39200 / 40 / 6000 / 80 / Replay≤1000 / 代表 980 / 无未来数据）：PASS

逐模块标签来源与允许性见 [LABEL_PROTOCOL_AUDIT.md](file:///root/autodl-tmp/diffusion%20distillation/LABEL_PROTOCOL_AUDIT.md)。

## 9. 结果解读（不含调参）

1. 精度下降起点与在线 VAE 更新严格对应：Initial 76.27% → 第 2 轮（首次 VAE 更新后）
   68.53%，此后在 58%~67% 区间波动，最终 60.64%。
2. 同期 VAE 自身分类头在 val 上的精度从 76.3% 降至 64.1%（均值 65.5%），
   而 Base Replay 上的 CE 已降到 0.03，属于「仅用 1,000 张 Base Replay × 6,000 步」
   的典型过拟合；云端精度与 VAE 分类头精度同步，未出现额外的蒸馏失真。
3. 伪标签接受率稳定在约 52%，类别分布长期不均衡（类别 3/5 极少），DM 已按协议
   回退到 Base Replay 目标，没有用真实流式标签做任何补偿。
4. 按任务要求**未做任何超参数搜索或阈值调节**，也未使用流式真实标签；
   该结果不能与旧 392 轮全监督结果比较优劣。

## 10. 四方法同协议对比（seed 42，80 轮）

命令：

```bash
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10_unlabeled_multi80_u2.yaml \
  --methods online_kmeans,online_moment,online_reservoir --seeds 42 --tag u2
```

| 方法 | Initial | Avg Online | Final | Avg F1 | Final F1 | Final int8 |
|---|---|---|---|---|---|---|
| Multi-MLP DM | 76.27 | 63.78 | **60.64** | 63.59 | 60.19 | 60.61 |
| Online K-means | 76.27 | 63.65 | 58.99 | 62.91 | 58.14 | 58.46 |
| Online Moment | 76.27 | 62.05 | 57.58 | 59.95 | 53.96 | 57.62 |
| Online Reservoir | 75.91 | **67.24** | **65.70** | **66.94** | **65.40** | **65.65** |

| 资源 | DM | K-means | Moment | Reservoir |
|---|---|---|---|---|
| 蒸馏耗时 (s) | 15.4 | 2.3 | 1.0 | 1.8 |
| Online VAE (s) | 158.2 | 156.7 | 159.4 | 157.2 |
| 云端训练 (s) | 46.9 | 45.0 | 46.7 | 48.7 |
| 端到端 wall (s) | 268.4 | 249.4 | 251.5 | 255.7 |
| 累计上传 | 82.43 MB | 82.43 MB | 82.43 MB | 82.43 MB |
| Peak GPU | 1.159 GB | 1.133 GB | 1.134 GB | 1.134 GB |

公平性核对（逐项实测）：

- 数据流：四方法 `stream_sha1 = a914f9bb4998e17e`、`base_indices_sha1 =
  9d36421552fc6fdf`，完全一致。
- Encoder 轨迹：81 条逐轮 `encoder_sha1` 记录（含初始）**0 处差异**；
  40 次更新 SHA1 与第 3 节列表完全相同。
- Teacher：指纹均为 `7343521e454f4f08`，`teacher_version=0`；80 轮
  `pseudo_summary` 与逐轮 `class_counts` **完全相同**（四方法共享同一批伪标签）。
- 预算：四方法均为 40 次 VAE 更新 / 6,000 优化步 / 80 次蒸馏 / 代表特征 980 /
  Raw Replay 1,000（`added_by_source = {base: 9800, stream: 0}`）。
- 对齐：最终 held-out error 均为 0.15268、improvement 0.18282（同一 Encoder 轨迹）。
- 唯一差异：Reservoir 的初始代表集是 Base 原始样本（当前 Encoder 精确重编码），
  其余三种用 `kmeans` 初始化，因此其 Initial 精度为 75.91% 而非 76.27%；
  这是方法自身定义带来的差异，Initial VAE 权重与在线轨迹完全相同。

曲线：[u2_four_methods_seed42.png](file:///root/autodl-tmp/diffusion%20distillation/vae_streaming/figures/u2_four_methods_seed42.png)。
结果文件：`online_kmeans_u2_seed42.json`、`online_moment_u2_seed42.json`、
`online_reservoir_u2_seed42.json`（同目录 `vae_streaming/results/cifar10/iid/`）。

在严格无流式真实标签、共享同一 Teacher 与同一批伪标签的条件下，Reservoir
（原始样本缓冲 + 当前 Encoder 重编码）明显优于三种特征空间方法，80 轮平均高出
DM 3.46 个点；但不改变此前结论：本协议结果不与旧 392 轮全监督结果比较优劣。
