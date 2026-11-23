# LABEL_PROTOCOL_AUDIT

**协议**：`unlabeled_after_base`（Initial Labeled Base + Unlabeled Streaming）
**运行配置**：`vae_streaming/configs/cifar10_unlabeled_multi80_u2.yaml`
**正式运行**：`vae_streaming/results/cifar10/iid/dm_multi_u2_seed42.json`
**日期**：2026-11-20

本文件逐模块列出训练链路中每一处标签输入的来源，以及该来源是否被协议允许。
优先级：**严格无流式真实标签泄漏 > 因果数据访问正确 > 四方法公平 > 性能**。

---

## 1. 协议不变量

1. Initial Base（9,800 张）允许携带真实标签，是唯一的真实标签来源。
2. Streaming（39,200 张，80 × 490）在训练期间不携带、也不可恢复真实标签。
3. 训练入口只接受三类数据对象，标签能力由类型显式约束。
4. 伪标签只能由**固定的 Initial Base Teacher** 产生，并显式标注
   `label_source = "initial_teacher_pseudo"`。
5. val / test 真实标签只出现在独立评估（指标记录）中，不参与在线更新、
   代表集选择、伪标签生成或逐轮模型选择。

---

## 2. 数据结构与密封机制

| 类型 | 字段 | 标签能力 | 使用范围 |
|---|---|---|---|
| `LabeledBatch` | images, labels, indices, source, label_source | 真实标签 | 仅 Initial Base 及其 Replay |
| `UnlabeledBatch` | images, **sample_ids**, round, domain_level | **无任何标签字段** | 流式轮次训练 |
| `PseudoLabeledBatch` | images, sample_ids, **pseudo_labels**, confidence, accept, label_source, teacher_version | 伪标签 | DM / K-means / Moment / Reservoir |

密封实现（`vae_streaming/data/causal_stream.py`）：

- `_seal_stream_labels()`：把 `train_labels` 中所有流式索引位置写成 `-1`，
  只有 `base_indices` 位置保留真实标签。
- `round(t)` 返回 `UnlabeledBatch`，**不存在** `labels` / `y` / `targets` 属性。
- `experiments/engine.py` 每轮生成伪标签后再次断言：
  `stream.train_labels[pseudo.sample_ids] == -1`，否则直接抛错。
- `label_mode` 仅允许 `supervised_stream` 与 `unlabeled_after_base`；
  后者强制 `scenario == "iid"`，因此 Label Shift 场景元数据不会进入无标签运行时。

---

## 3. 逐模块标签来源审计

| 模块 / 函数 | 实际读取的标签 | 标签来源 | 允许 | 说明 |
|---|---|---|---|---|
| `CausalStream.base` → `LabeledBatch` | `base.labels` | `ground_truth`（Initial Base） | ✅ | 9800 张，980/类 |
| `CausalStream.round(t)` → `UnlabeledBatch` | 无 | — | ✅ | 训练接口无标签字段 |
| `CausalStream.train_labels`（流式位置） | 恒为 `-1` | 密封 | ✅ | 引擎逐轮断言 |
| `CausalStream.summary()` | `per_class_train`（全量聚合计数） | 数据集统计 | ⚠️ 仅审计 | 不进入任何训练分支 |
| `train_initial_vae` | base labels | `ground_truth` | ✅ | Initial Base VAE |
| `online_vae_update(labels=None)` | 仅 Replay 段 | `ground_truth`（Base Replay） | ✅ | 当前窗口无 CE |
| `vae_losses(..., n_unlabeled)` | `y[n_unlabeled:]` | Base Replay | ✅ | CE 分组均值，不被无标签样本稀释 |
| `FixedTeacher.predict` | 无（仅图像） | — | ✅ | 参数冻结、`requires_grad=False` |
| `PseudoLabeledBatch` | `pseudo_labels` | `initial_teacher_pseudo` | ✅ | `teacher_version=0`，全轮固定 |
| `ClassBalancedReservoir`（Raw Replay） | base labels | `ground_truth` | ✅ | 运行时禁止 `source=stream & label_source=ground_truth` |
| `ClassBalancedReservoir`（蒸馏 Reservoir） | pseudo labels | `initial_teacher_pseudo` | ✅ | 标注 `source=stream`，只做 pseudo-balanced |
| `StreamingStatistics.update` | base GT + pseudo | 混合 | ✅ | `label_sources` 分别计数 |
| `StreamingDM.update` | base GT + pseudo | 混合 | ✅ | `real_sources` 标记，逐类记录 base/pseudo 数量 |
| `OnlineKMeans.update` | base GT + pseudo | 混合 | ✅ | 空类别跳过新数据 |
| `streaming_moment_calibrate` | 统计量（base+pseudo） | 混合 | ✅ | 逐类计数/方差来源可追溯 |
| `RepresentativeMemory` | `memory.labels` | Base 配额（base 派生） | ✅ | 不从流式真实标签产生 |
| `estimate_feature_alignment` | 仅 `anchors_y` 诊断 | Base Replay GT | ✅ | 仿射拟合只用新旧特征对 |
| `build_sync_message` / 通信记账 | `memory.labels` | Base 派生 | ✅ | 承载代表特征类别，非流式 GT |
| `IncrementalCloudLearner` | `memory.labels` | Base 派生 | ✅ | `val=None`（关闭逐轮选优） |
| `evaluate_vae` / 云端评估 | val / test labels | 独立评估 | ✅ | 只写入指标，不回流训练 |
| `stream.val_labels` / `test_labels` | 评估 | 独立评估 | ✅ | 不参与伪标签、代表集或模型选择 |

图例：✅ 协议允许；⚠️ 仅审计用途；❌ 出现即为泄漏（本协议中不存在）。

---

## 4. 关键防线（代码位置）

| 防线 | 位置 | 行为 |
|---|---|---|
| 类型隔离 | `data/causal_stream.py` | `UnlabeledBatch` 无标签字段 |
| 标签密封 | `CausalStream._seal_stream_labels` | 流式位置 `train_labels = -1` |
| 逐轮断言 | `experiments/engine.py` 轮首 | 伪标签来源位置的密封值必须全为 `-1` |
| Teacher 固定 | `models/pseudo_labeler.py:FixedTeacher` | deepcopy + eval + 冻结，版本恒为 0 |
| Replay 写保护 | `memory/replay_buffer.py:add_batch` | 拒绝写入流式真实标签 |
| 云端固定协议 | `cloud.incremental_trainer` + `cloud.select_best_on_val=false` | 不使用 val 标签选模型 |
| 对齐诊断限域 | `models/feature_alignment.py` | 类中心误差只用 Base Replay 标签 |

---

## 5. 防标签泄漏测试记录

命令：

```bash
python -m pytest vae_streaming/tests/test_label_protocol.py -q
```

| 测试 | 检查内容 | 结果 |
|---|---|---|
| Test 1 | 流式位置 `train_labels == -1`；`round(t)` 无 `labels/y/targets`；base 标签完整 | PASS |
| Test 2 | 在每条 `UnlabeledBatch` 上注入随机置乱 `labels` 后重跑，`stream_sha1`、Initial Base/Teacher 指纹、逐轮 `encoder_sha1`、`memory_sha1`、`cloud_param_sha1`、acc/f1、DM loss 全部一致 | PASS |
| Test 3 | 训练接口无标签字段，完整流水线运行无报错、无静默恢复 | PASS |
| Test 4 | `pseudo_labels == teacher.predict`；`accept == (confidence >= 0.9)`；Teacher 参数冻结且重复预测一致 | PASS |
| Test 5 | 39200 样本 / 40 次 VAE 更新 / 6000 优化步 / 80 次 DM / Replay ≤ 1000 / 代表特征恒 980 / 无未来数据 | PASS |

完整测试输出：`6 passed`（含基于正式 80 轮结果的预算核对）。

### 5.1 正式运行的预算实测

| 项目 | 期望 | 实测 |
|---|---|---|
| 流式样本总数 | 39,200 | 39,200（80 × 490） |
| Initial Base | 9,800 | 9,800 |
| VAE 更新次数 | 40 | 40 |
| VAE 优化步数 | 6,000 | 6,000（40 × 150） |
| DM 更新次数 | 80 | 80 |
| Raw Replay 容量 | ≤ 1,000 | 1,000（100/类，`stream` 写入 0） |
| 蒸馏 Reservoir 容量 | 980 | 980（pseudo-balanced） |
| 代表特征数 | 980 | 980（全程恒定） |
| 累计可见样本（逐轮） | 9800 + 490×(t+1) | 逐轮一致 |
| 单轮通信量 | full_sync + int8 | 993.8 KB/轮 |

### 5.2 伪标签来源检查（正式运行）

- 每轮 `pseudo.label_source == "initial_teacher_pseudo"`。
- Teacher 指纹全程不变（`teacher_version = 0`）。
- 80 轮累计接受 20,625 / 39,200（52.6%），逐轮接受率与类别分布记录在
  `rounds[t].pseudo`，汇总在 `summary.pseudo_summary`。
- 类别 3 全程仅 31 个高置信伪标签：该类别在 DM 中回退为
  `target_mode = "base_replay_only_fallback"`，未用其他类别填充。
- `stream.train_labels` 在全部流式位置恒为 `-1`（Test 1/3 逐轮校验）。

### 5.3 四方法共享性（80 轮实测）

`online_kmeans / online_moment / online_reservoir` 与 `dm_multi` 使用同一配置、
同一 seed 跑满 80 轮后逐项核对：

- `stream_sha1`、`base_indices_sha1` 完全一致；
- 81 条逐轮 `encoder_sha1`（含初始）0 处差异；
- Teacher 指纹同为 `7343521e454f4f08`，80 轮 `pseudo_summary` 与逐轮
  `class_counts` 完全相同；
- Raw Replay 均为 `added_by_source={base:9800, stream:0}`，即无任何流式样本
  或其标签写入 Replay。

因此四方法的差异只来自代表集维护算法，不来自标签或伪标签差异。

---

## 6. 审计结论

- 训练链路中不存在任何读取流式真实标签的分支；置乱实验证明其不可影响训练轨迹。
- 唯一真实标签来源为 Initial Base 及其 Replay，且只进入 Initial Base VAE 与
  在线 VAE 的 Replay 分类损失。
- 伪标签来源单一、可追溯，Teacher 与在线更新完全解耦。
- 因此本协议下的性能变化不能归因于标签泄漏；跨协议（旧 392 轮全监督）结果
  不具可比性。
