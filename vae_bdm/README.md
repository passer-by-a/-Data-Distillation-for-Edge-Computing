# VAE-BDM: VAE 潜在特征的 Batch Distribution Matching 数据集蒸馏

第一阶段（静态数据集蒸馏）：冻结 EdgeVAE 编码器，把训练图像的 1024 维 `mu` 特征
蒸馏成更少的 1024 维代表特征；边缘只上传代表特征 + 标签，云端从头训练分类器，
在独立测试图像上评估。

> 本阶段减少的是**训练集上传量**，不是单张测试图的在线推理上传量。
> VAE 预训练使用过 49k 分类标签，因此这是**有监督潜在特征空间**中的蒸馏，
> 不能称为全流程无监督。

## 1. 特征协议

- 数据划分：`RandomState(42).permutation(50000)` → train 49000 / val 1000 / test 10000。
- 图像归一化：`[-1,1]`。
- 编码：`EdgeVAE.encode_mu`（确定性 posterior mean）→ `[N,16,8,8]`。
- 特征归一化：训练集逐通道 z-score，然后 flatten → `[N,1024]`。
- 该协议与已有 90% 实验一致；新缓存与旧缓存抽样 `max|diff|=0.005`（float16 舍入）。
- 缓存：`cache/{dataset}/{train,val,test}_{features,labels}.npy` + `metadata.json` + `norm_stats.pt`。
- 编码器：431,200 参数，255.5 MFLOPs/图，约 0.49 ms/图（RTX 4090 D，实测）。

## 2. 方法

| 名称 | 说明 |
|---|---|
| Full | 全部 49k 特征（上界参考） |
| Random | 类内随机选配额个真实特征（下界参考） |
| K-means | 类内 K-means 质心 |
| K-means + Moment | K-means 质心 + 解析式均值/方差校准 |
| Batch DM | **原版 DM**：固定随机 MLP 特征空间中的逐类均值匹配 |

Batch DM（原版 DM）的关键点：
- 真实特征和合成特征都先经过一个**固定随机初始化**的 MLP（`RandomMLP`，
  默认 hidden 2048 / output 1024，`feature_seed=0`），只训练合成特征。
- 损失为 `(mean(f_real) - mean(f_syn))²` 的逐维平均，**只匹配均值**；
  不包含方差、协方差、diversity 或 anchor 项。
- `S ∈ R^{M×1024}` 是直接优化的 `nn.Parameter`，标签固定；VAE 与随机特征网络均冻结。
- 每类一个 `nn.Parameter` + 独立 AdamW；每轮随机排列全部类别，每类更新次数严格等于
  `updates_per_class`（balanced 更新），非当前类完全不更新。
- 默认 K-means 初始化。
- 每 `eval_every` 步在**真实验证集**上评估一次（合成特征训练线性头），按验证精度选 checkpoint；
  `selection: final` 可改为固定终止步数、不选点。
- 测试集不参与蒸馏、统计量或模型选择。

> 历史说明：早期版本曾使用潜空间 mean+var / full-stats / 协方差 / diversity / anchor 等
> 变体。为避免与后续更好的特征提取器实验混淆，这些目标及相关结果已全部删除；
> 当前代码只有原版 DM 一条路径。

配额：按类别样本数比例、最大余数法分配，保证总数精确等于 `round(49000/R)`，
且每类至少 1 个代表；支持 10/20/50/100:1。

## 3. 云端与量化

- Linear：`Linear(1024,C)`。
- MLP：`Linear(1024,512)-GELU-Dropout(0.2)-Linear(512,C)`。
- 从随机初始化训练；AdamW `lr=1e-3`、cosine、200 epoch、3 seeds；
  云端验证只使用独立真实验证集（`val_mode=external`）或关闭（`none`）。
- FP32 / FP16 / 非对称 INT8：校准只用蒸馏后的训练代表特征，
  测试特征复用同一 scale/zero point；量化后重新训练云端分类器。
- 真实 payload：`.bin` 特征 + `.bin` 标签 + `.json` 元数据，报告实际字节。

## 4. 运行

```bash
# 1) 特征缓存
python -m vae_bdm.scripts.extract_features --config cifar10_v2.yaml

# 2) 10:1 基线
python -m vae_bdm.scripts.run_baselines --config cifar10_v2.yaml --ratio 10 \
  --methods full,random,kmeans,kmeans_moment

# 3) 原版 DM（3 个 DM seed × 3 个云端 seed）
python -m vae_bdm.scripts.run_bdm --config cifar10_v2.yaml --ratio 10 \
  --variants random_feature --seeds 42,43,44

# 4) 参数消融
python -m vae_bdm.scripts.run_bdm --config cifar10_v2.yaml --ratio 10 \
  --variants upd100,upd300,sel_linear50,sel_mlp100,sel_final --seeds 42

# 5) 压缩率扫描
python -m vae_bdm.scripts.run_sweep --config cifar10_v2.yaml --ratios 20,50,100 \
  --methods random,kmeans,kmeans_moment --bdm_variant random_feature

# 6) 量化
python -m vae_bdm.scripts.run_quant --config cifar10_v2.yaml --ratio 10

# 7) 汇总 + 图
python -m vae_bdm.scripts.normalize_v2_results
python -m vae_bdm.scripts.summarize --config cifar10_v2.yaml --ratio 10
python -m vae_bdm.scripts.make_figures --config cifar10_v2.yaml

# 测试与审计
python -m pytest vae_bdm/tests/test_protocol_v2.py -q
python -m vae_bdm.scripts.audit_protocol
```

CIFAR-100 把配置换成 `cifar100_v2.yaml`，高压缩比用 `20,50,100`。

## 5. 结果（v2 协议，RTX 4090 D）

协议：冻结 EdgeVAE 的确定性 `mu` → 1024 维；云端 MLP（1024-512-C）；
AdamW `lr=1e-3`、200 epoch、3 个云端种子（0/1/2）；测试集只做最终评估。
原版 DM 跑 3 个蒸馏种子（42/43/44），表中 `±` 为蒸馏种子间标准差；
基线为云端种子标准差。结果在 `results/` 下的
`*_baselines_r10.json`、`*_sweep.json`、`*_quant.json`，`protocol_version=vae_bdm_v2`。

### 5.1 CIFAR-10，10:1（M=4900）

| 方法 | Top-1 | Macro-F1 |
|---|---:|---:|
| Full（49k） | **90.19±0.19** | 90.18±0.19 |
| Random | 89.90±0.03 | 89.89±0.03 |
| K-means | 90.09±0.05 | 90.07±0.05 |
| K-means + Moment | 90.05±0.09 | 90.02±0.09 |
| Batch DM（原版 DM） | 90.06±0.06 | 90.03±0.06 |

### 5.2 CIFAR-100，10:1（M=4900）

| 方法 | Top-1 | Macro-F1 |
|---|---:|---:|
| Full（49k） | **61.56±0.15** | 61.55±0.16 |
| Random | 58.30±0.13 | 58.22±0.15 |
| K-means | 59.23±0.20 | 59.43±0.22 |
| K-means + Moment | **60.85±0.07** | 60.89±0.09 |
| Batch DM（原版 DM） | 59.61±0.25 | 59.78±0.24 |

`bdm_random_feature` 三个蒸馏种子：C10 90.12 / 90.06 / 89.98；C100 59.89 / 59.66 / 59.29。

### 5.3 压缩比扫描（3 云端种子）

CIFAR-10：

| 压缩比 | M | Random | K-means | K-means+Moment | Full |
|---|---:|---:|---:|---:|---:|
| 50:1 | 980 | 89.53 | **89.66** | 89.22 | 90.19 |
| 100:1 | 490 | 89.27 | **89.58** | 89.26 | 90.19 |

CIFAR-100：

| 压缩比 | M | Random | K-means | K-means+Moment | Full |
|---|---:|---:|---:|---:|---:|
| 20:1 | 2450 | 55.49 | 59.13 | **59.68** | 61.56 |
| 50:1 | 980 | 48.72 | 58.36 | **58.72** | 61.56 |
| 100:1 | 490 | 41.16 | **57.80** | 57.06 | 61.56 |

Random 在 CIFAR-100 高压缩比下明显退化（每类真实特征太少）；
K-means 更稳，Moment Calibration 在 20:1/50:1 有增益，100:1 下逐维方差估计噪声变大。

### 5.4 Moment Calibration 对照（10:1，3 云端种子）

| 数据集 | 基础方法 | 基线 Top-1 | +Moment Top-1 | 均值 MSE 前→后 | 方差 MSE 前→后 |
|---|---|---:|---:|---:|---:|
| C10 | K-means | 90.09 | 90.05 | 3.62e-3 → 3.28e-15 | 7.34e-2 → 5.79e-13 |
| C100 | K-means | 59.23 | **60.85** | 1.02e-2 → 2.75e-15 | 1.40e-1 → 1.24e-12 |

解析式矩校准能精确匹配类均值/方差，但收益强依赖数据集与预算：
C100 10:1 提升 1.6 点，C10 10:1 基本不变；低预算下逐维方差估计噪声大。

### 5.5 量化（INT8）

代表特征沿用同一量化流程（FP32 / FP16 / 非对称 INT8，校准仅用训练代表特征，
测试特征复用同一 scale/zero point）：

| 压缩比 | 数据集 | 方法 | FP32 | INT8 | INT8 实际字节 |
|---|---|---:|---:|---:|---:|
| 10:1 | C10 | Random | 89.93 | 89.87 | 5,027,765 |
| 10:1 | C10 | K-means | 90.11 | 90.11 | 5,027,766 |
| 10:1 | C100 | K-means | 59.02 | 59.10 | 5,027,765 |
| 100:1 | C10 | K-means | 89.27 | 89.27 | 503,103 |
| 100:1 | C100 | K-means | 57.78 | 57.82 | 503,103 |

量化对 8 bit 不敏感（Δ≤0.09 点）。

### 5.6 复现检查与风险

1. Full / K-means / Random 与历史 90% 实验同量级；差异来自云端分类器与种子协议。
2. 原版 DM 的 checkpoint 由 1000 张真实验证集选点，单次验证精度有 ±0.3 量级噪声；
   所有方法共用同一协议，不按测试集回调。
3. CIFAR-100 100:1 的 Random 只剩每类 4–5 个真实特征，云端 MLP 在这个规模上严重过拟合。
4. 本阶段只模拟一次性上传；在线推理仍需为每张测试图上传 1024 维特征（4 KB FP32 / 1 KB INT8）。
5. 主表 VAE 分类头使用全部 49k 标签；重建 + KL 使用全部训练图像，因此不能称为全流程无监督。

## 6. 口径与风险

1. 分组使用标签；VAE 预训练使用 49k 标签；不能宣称无监督。
2. Random 基线是“类内随机选真实特征”，不是“随机分组后取均值”；两者数值不同。
3. 1024 元素 ≠ 1024 字节；报告字节必须带位宽。
4. K-means 可能降低类内方差；如需诊断可对代表特征直接计算方差比，但该指标不参与训练目标。
5. 云端分类器始终从随机初始化开始；Full 只作为参考。
6. 验证集只用于 DM checkpoint 选择与云端早停；测试集只做最终评估。
7. 每类一个参数 + 独立 AdamW 是协议要求；不要退回单张量 + 单优化器（会让非当前类被动量更新）。
8. 训练阶段显存峰值约 0.23–0.24 GB（C10/C100，`M=4900`）；
   固定随机特征的整类前向结果按类缓存一次，单步只做代表特征的前向。
