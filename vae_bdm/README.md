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
| Random | 类内随机选配额个真实特征 |
| Mean | 类内分块取均值 |
| K-means | 类内 K-means 质心 |
| Herding | 类内贪心核心集 |
| BDM mean | Batch DM，仅均值匹配 |
| BDM mean+var | 均值 + 逐维方差 |
| BDM full | 用完整类别统计量替代 batch 统计 |
| BDM anchor | + `||S-S0||²` 约束 |
| BDM diversity | + 类内余弦相似度约束 |
| BDM cov | + 固定随机投影协方差匹配 |
| BDM all | anchor + diversity + cov |

Batch DM 的关键点：
- `S ∈ R^{M×1024}` 是直接优化的 `nn.Parameter`，标签固定；VAE 冻结。
- 每步随机选一类，采样该类的真实特征 batch，与该类全部合成特征匹配。
- 默认 K-means 初始化；支持 Random 初始化消融。
- 每 500 步在**真实验证集**上评估一次（合成特征训练线性头），按验证精度选 checkpoint。
- 测试集不参与蒸馏、统计量或模型选择。

配额：按类别样本数比例、最大余数法分配，保证总数精确等于 `round(49000/R)`，
且每类至少 1 个代表；支持 10/20/50/100:1。

## 3. 云端与量化

- Linear：`Linear(1024,C)`。
- MLP：`Linear(1024,512)-GELU-Dropout(0.2)-Linear(512,C)`。
- 从随机初始化训练；AdamW `lr=1e-3`、cosine、200 epoch、3 seeds；
  合成特征内部分层 10% 作为云端早停验证。
- FP32 / FP16 / 非对称 INT8：校准只用蒸馏后的训练代表特征，
  测试特征复用同一 scale/zero point；量化后重新训练云端分类器。
- 真实 payload：`.bin` 特征 + `.bin` 标签 + `.json` 元数据，报告实际字节。

## 4. 运行

```bash
# 1) 特征缓存
python -m vae_bdm.scripts.extract_features --config cifar10.yaml

# 2) 10:1 基线
python -m vae_bdm.scripts.run_baselines --config cifar10.yaml --ratio 10

# 3) BDM 变体（3 个 DM seed × 3 个云端 seed）
python -m vae_bdm.scripts.run_bdm --config cifar10.yaml --ratio 10

# 4) 压缩率扫描
python -m vae_bdm.scripts.run_sweep --config cifar10.yaml

# 5) 量化
python -m vae_bdm.scripts.run_quant --config cifar10.yaml --ratio 10

# 6) 汇总 + 图
python -m vae_bdm.scripts.summarize --config cifar10.yaml --ratio 10
python -m vae_bdm.scripts.make_figures --config cifar10.yaml

# CIFAR-100
python -m vae_bdm.scripts.extract_features --config cifar100.yaml
python -m vae_bdm.scripts.run_baselines --config cifar100.yaml --ratio 10
python -m vae_bdm.scripts.run_bdm --config cifar100.yaml --ratio 10 --variants mean,mean_var,anchor
python -m vae_bdm.scripts.run_sweep --config cifar100.yaml --ratios 10,100

# 10% 标签 VAE（分类头 500/类，重建+KL 仍用全部图像）
python vae.py --label_shot 500
python -m vae_bdm.scripts.extract_features --config cifar10_lab500.yaml
python -m vae_bdm.scripts.run_baselines --config cifar10_lab500.yaml --ratio 10
python -m vae_bdm.scripts.run_bdm --config cifar10_lab500.yaml --ratio 10 --variants mean,mean_var --seeds 42,43
python -m vae_bdm.scripts.run_sweep --config cifar10_lab500.yaml
python -m vae_bdm.scripts.summarize --config cifar10_lab500.yaml --ratio 10
python -m vae_bdm.scripts.make_figures --config cifar10_lab500.yaml

# v2 统一协议（外部真实验证集 + balanced 更新 + 每类优化器）
python -m pytest vae_bdm/tests/test_protocol_v2.py -q
python -m vae_bdm.scripts.audit_protocol
python -m vae_bdm.scripts.run_baselines --config cifar100_v2.yaml --ratio 10 \
  --methods full,random,mean,kmeans,herding,kmeans_moment,random_moment
python -m vae_bdm.scripts.run_bdm --config cifar100_v2.yaml --ratio 10 \
  --variants mean_var --seeds 42,43,44
python -m vae_bdm.scripts.run_bdm --config cifar100_v2.yaml --ratio 10 \
  --variants upd100,upd300,sel_linear50,sel_mlp100,sel_final --seeds 42
python -m vae_bdm.scripts.run_sweep --config cifar100_v2.yaml --ratios 20,50,100 \
  --methods random,mean,kmeans,herding,kmeans_moment --bdm_variant mean_var
python -m vae_bdm.scripts.run_budget_ablation --config cifar100_v2.yaml \
  --ratios 100 --methods kmeans,bdm --steps 4000
python -m vae_bdm.scripts.normalize_v2_results
python -m vae_bdm.scripts.summarize --config cifar100_v2.yaml --ratio 10
python -m vae_bdm.scripts.make_figures --config cifar100_v2.yaml
# CIFAR-10 同理，把 cifar100_v2.yaml 换成 cifar10_v2.yaml，高压缩比用 50,100
```

## 5. 结果（2026-11-14，RTX 4090 D）

协议：冻结 EdgeVAE 的确定性 `mu` → 1024 维；云端 MLP（1024-512-C）；
AdamW `lr=1e-3`、200 epoch、3 个云端种子（0/1/2）；测试集只做最终评估。
`bdm_mean_var` 额外跑 3 个蒸馏种子（42/43/44），其余 BDM 变体为 1 个（42）。
表中 `±`：多种子方法为蒸馏种子间标准差，单种子方法为云端种子标准差。

### 5.1 CIFAR-10，10:1（M=4900）

| 方法 | Top-1 | Macro-F1 | 相对 Full 掉点 | 类内方差比 | 蒸馏耗时 |
|---|---:|---:|---:|---:|---:|
| Full（49k） | **90.17±0.05** | 90.16±0.05 | 0.00% | 1.000 | 0.0 s |
| Random | 89.93±0.05 | 89.92±0.05 | 0.27% | 0.999 | 0.0 s |
| Mean | 88.81±0.26 | 88.84±0.25 | 1.50% | 0.100 | 0.2 s |
| K-means | **90.11±0.05** | 90.08±0.05 | 0.06% | 0.634 | 49.8 s |
| Herding | 90.04±0.03 | 90.03±0.03 | 0.14% | 0.898 | 106.6 s |
| BDM mean | 89.94±0.03 | 89.91±0.02 | 0.25% | 0.649 | 71.4 s |
| BDM mean+var | 89.87±0.05 | 89.84±0.05 | 0.33% | 0.991 | 68.5 s |
| BDM mean+var，λ_v=0.1 | 89.88±0.06 | 89.86±0.06 | 0.31% | 0.998 | 61.2 s |
| BDM mean+var，λ_v=10 | 89.92±0.12 | 89.89±0.12 | 0.28% | 0.995 | 61.5 s |
| BDM full-class stats | 89.98±0.10 | 89.96±0.10 | 0.20% | 1.000 | 71.3 s |
| BDM + anchor | 89.93±0.08 | 89.91±0.08 | 0.26% | 0.994 | 67.0 s |
| BDM + diversity | 89.85±0.04 | 89.83±0.04 | 0.35% | 0.998 | 80.1 s |
| BDM + cov（r=64） | 89.91±0.06 | 89.90±0.06 | 0.28% | 0.995 | 72.1 s |
| BDM all | 89.90±0.01 | 89.88±0.01 | 0.29% | 0.997 | 82.3 s |
| BDM mean+var，Random init | 89.98±0.03 | 89.96±0.03 | 0.21% | 0.992 | 16.3 s |

类内方差比 = 逐类计算「合成特征类内方差 / 真实特征类内方差」，再对类平均（1024 维等权）。

### 5.2 CIFAR-10 压缩比扫描（3 云端种子，BDM 为 mean+var，蒸馏种子 42）

| 压缩比 | M | Random | Mean | K-means | Herding | Batch DM | Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10:1 | 4900 | 89.93 | 88.81 | **90.11** | 90.04 | 89.86 | 90.17 |
| 20:1 | 2450 | 89.72 | 88.93 | 89.73 | **89.97** | 89.52 | 90.17 |
| 50:1 | 980 | 89.31 | 88.85 | 89.47 | **89.60** | 89.22 | 90.17 |
| 100:1 | 490 | 89.09 | 89.04 | **89.27** | 89.24 | 88.96 | 90.17 |

### 5.3 CIFAR-10 量化与真实负载（10:1，M=4900，BDM 用蒸馏种子 42）

| 方法 | 精度 | FP32 实际字节 | FP16 实际字节 | INT8 实际字节 | INT8 vs FP32 Full |
|---|---|---:|---:|---:|---:|
| Batch DM mean+var | 89.86 / 89.86 / 89.86 | 20,080,514 | 10,045,314 | **5,027,766** | 39.9× |
| K-means | 90.11 / 90.11 / 90.11 | 20,080,514 | 10,045,314 | 5,027,766 | 39.9× |
| Herding | 90.04 / 90.04 / 90.01 | 20,080,514 | 10,045,314 | 5,027,765 | 39.9× |
| Random | 89.93 / 89.93 / 89.87 | 20,080,514 | 10,045,314 | 5,027,765 | 39.9× |
| Mean | 88.81 / 88.81 / 88.80 | 20,080,514 | 10,045,314 | 5,027,764 | 39.9× |

（精度顺序 FP32 / FP16 / INT8；字节为 `.bin` 特征 + `.bin` 标签 + `.json` 元数据的实测总大小。
Full FP32 负载按 49,000×1024×4 + 49,000×2 = 200,802,000 B 精确计算，未落盘。）

### 5.4 CIFAR-100，10:1（M=4900）

| 方法 | Top-1 | Macro-F1 | 相对 Full 掉点 | 类内方差比 | 蒸馏耗时 |
|---|---:|---:|---:|---:|---:|
| Full（49k） | **61.20±0.17** | 61.27±0.19 | 0.00% | 1.000 | 0.0 s |
| Random | 58.25±0.28 | 58.15±0.35 | 4.82% | 0.977 | 0.0 s |
| Mean | 56.59±0.29 | 56.96±0.24 | 7.54% | 0.098 | 0.2 s |
| K-means | 59.02±0.00 | 59.22±0.04 | 3.56% | 0.608 | 8.3 s |
| Herding | 59.71±0.21 | 59.75±0.19 | 2.43% | 0.873 | 6.5 s |
| BDM mean | 59.79±0.11 | 59.93±0.14 | 2.31% | 0.608 | 24.8 s |
| BDM mean+var | **60.24±0.15** | 60.11±0.22 | 1.57% | 0.951 | 28.5 s |

`bdm_mean_var` 为蒸馏种子 42/43 的均值（60.38 / 60.09），± 为两者标准差。

### 5.5 CIFAR-100 压缩比扫描（3 云端种子，BDM 为 mean+var，蒸馏种子 42）

| 压缩比 | M | Random | Mean | K-means | Herding | Batch DM | Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10:1 | 4900 | 58.25 | 56.59 | 59.02 | 59.71 | **60.38** | 61.20 |
| 20:1 | 2450 | 55.06 | 55.52 | 58.81 | 58.30 | **59.55** | 61.20 |
| 50:1 | 980 | 47.43 | 55.01 | **58.02** | 54.18 | 57.18 | 61.20 |
| 100:1 | 490 | 41.37 | 54.99 | **57.78** | 51.00 | 57.70 | 61.20 |

### 5.6 CIFAR-100 量化（10:1 / 100:1，INT8）

| 压缩比 | 方法 | FP32 | INT8 | INT8 实际字节 | vs FP32 Full |
|---|---|---:|---:|---:|---:|
| 10:1 | Batch DM | 60.38 | 60.36 | 5,027,765 | 39.9× |
| 10:1 | Herding | 59.71 | 59.69 | 5,027,766 | 39.9× |
| 10:1 | K-means | 59.02 | 59.10 | 5,027,765 | 39.9× |
| 100:1 | Batch DM | 57.70 | 57.71 | 503,102 | 399.1× |
| 100:1 | Herding | 51.00 | 50.95 | 503,103 | 399.1× |
| 100:1 | K-means | 57.78 | 57.82 | 503,103 | 399.1× |

### 5.7 四个研究问题的回答

**RQ1（固定预算下能否保留性能）**：能。CIFAR-10 10:1 的 BDM（89.87）相对 Full（90.17）
只掉 0.33%（绝对 0.30 点）；CIFAR-100 上 60.24 vs 61.20，相对掉 1.57%。

**RQ2（与传统 coreset 谁更有效）**：取决于类别数。
- CIFAR-10：K-means 90.11 > Herding 90.04 > Random 89.93 > BDM 89.87。
  BDM 没有优势，比 K-means 低 0.24 点，也略低于 Random。
  **但排序依赖分类头**：换成 Linear 头后 Random 89.79、Herding 89.79、BDM 89.78、
  K-means 89.64（`results/cifar10_head_linear_r10.json`）。论文中必须固定分类头再比较。
- CIFAR-100：BDM 60.24 > Herding 59.71 > K-means 59.02 > Random 58.25 > Mean 56.59。
  BDM 领先最强基线 0.53 点，且方差匹配的增益（+0.45）明显大于 CIFAR-10（-0.07）。

**RQ3（更高压缩比）**：BDM 的优势不随压缩比单调扩大。
- CIFAR-10：各压缩比下 K-means/Herding 仍最高；BDM 甚至略低于 Random
  （差 0.07–0.20 点），只在 100:1 与 Mean 接近。该数据集上 DM 不占优。
- CIFAR-100：BDM 在 10:1、20:1 最好；50:1、100:1 被 K-means 以 0.8/0.1 点反超。
- Herding 与 Random 在 CIFAR-100 的高压缩比下明显退化（100:1 分别 51.00、41.37），
  说明「选真实样本」类方法在极小预算 + 多类别时不稳定；BDM 与 K-means 更稳。

**RQ4（统计量的影响）**：
- 只匹配均值：C10 89.94（方差比 0.649），C100 59.79（0.608）。
- 加方差匹配：C10 89.87（0.991，-0.07），C100 60.24（0.951，**+0.45**）。
  方差匹配确实把类内方差恢复到真实水平；它是否提升精度依赖类别数与预算，
  CIFAR-100 上明显有益，CIFAR-10 上持平略负。
- **v2 协议复核（3 云端种子，10:1）**：C100 原版 DM/mean-only 60.08 vs mean+var 60.37
  （−0.29）；C10 mean-only 90.01 vs mean+var 89.85（+0.16）。
  结论与旧协议一致：C100 上方差项有益，C10 上基本持平（略偏 mean-only）。
- Anchor（89.93）、Diversity（89.85）、随机投影协方差（89.91）、三者全开（89.90）
  相对 mean+var（89.87）没有可辨增益；这些约束主要价值是稳定训练，而非提点。
- 用完整类别统计量替代 batch 统计（89.98）与 batch 版（89.87）差异很小，
  说明 256 样本的 batch 均值/方差噪声不是主要误差来源。
- `λ_v` 在 {0.1, 1, 10} 下分别为 89.88 / 89.87 / 89.92（方差比 0.998 / 0.991 / 0.995），
  对 λ_v 不敏感；即使很弱的方差项也足以把方差比拉回 1.0。
- 方差收缩本身不是掉点的充分原因：K-means 的方差比只有 0.634，仍是 C10 最好；
  Mean 的 0.100 才对应明显掉点（88.81）。极端方差坍缩有害，中等收缩无害。

### 5.8 VAE 标签预算：500/类（10%）重训（2026-11-14 补充）

主表的 VAE 分类头用了全部 49k 标签。为排除「依赖全量标签」的质疑，用
`python vae.py --label_shot 500` 重训 VAE：重建 + KL 仍用全部 49k 图像，
**分类 CE 只用 500/类（C10 共 5,000 个标签，10.2%）**，其余协议不变（100 epoch、增强）。
权重：`checkpoints/autoencoder_edge_s8_c16_lab500aug.pt`（best epoch 96）。
特征缓存与结果在 `cache/lab500/`、`results/lab500/`，配置 `configs/cifar10_lab500.yaml`。

CIFAR-10 10:1（M=4900，3 云端种子；BDM 为 2 个蒸馏种子的均值）：

| 方法 | 全标签 VAE | 10% 标签 VAE | Δ（10% − 全） | 10% 下方差比 |
|---|---:|---:|---:|---:|
| Full（49k） | 90.17±0.05 | **78.02±0.27** | -12.15 | 1.000 |
| Herding | 90.04±0.03 | 75.91±0.02 | -14.13 | 0.868 |
| Random | 89.93±0.05 | 74.91±0.26 | -15.02 | 0.997 |
| Mean | 88.81±0.26 | 73.86±0.07 | -14.95 | 0.100 |
| BDM mean+var | 89.87±0.05 | 72.99±0.22 | -16.88 | 0.989 |
| BDM mean | 89.94±0.03 | 72.63±0.25 | -17.31 | 0.752 |
| K-means | 90.11±0.05 | 72.86±0.84 | -17.25 | 0.739 |

压缩比扫描（10% 标签 VAE，3 云端种子，BDM 为 mean+var，蒸馏种子 42）：

| 压缩比 | M | Random | Mean | K-means | Herding | Batch DM | Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10:1 | 4900 | 74.91 | 73.86 | 72.86 | **75.91** | 72.76 | 78.02 |
| 20:1 | 2450 | 74.51 | 73.27 | 71.15 | **75.66** | 71.28 | 78.02 |
| 50:1 | 980 | 73.70 | 72.78 | 74.35 | **75.16** | 72.62 | 78.02 |
| 100:1 | 490 | 72.74 | 72.84 | 74.38 | **74.64** | 72.03 | 78.02 |

结论：
1. **特征上界大幅下降**：VAE 分类头标签从 49k 降到 5k，Full 特征精度从 90.17 掉到 78.02
   （-12.15）。蒸馏只能在这个上界内工作，无法弥补表征本身的退化。
2. **方法排序反转**：全标签时 K-means/Herding > Random > BDM > Mean；10% 标签时
   **Herding > Random > Mean > BDM ≈ K-means**。弱特征空间里「选真实样本」明显优于
   「生成矩匹配特征」。
3. **掉点幅度差异**：Herding 只掉 2.70%（相对），K-means/BDM 掉 6.45–6.91%。
   合成特征对表征噪声更敏感；真实特征保留了个体判别信息。
4. BDM 仍略好于 K-means（+0.13），方差匹配把类内方差比从 0.752 拉回 0.989，
   但只换来 +0.36 精度，远小于与 Herding 的 2.92 点差距。
5. 压缩比扫描中 Herding 在 10/20/50/100:1 全部最好（74.64–75.91）；BDM 始终
   71.28–72.76，50/100:1 被 K-means 反超（74.35/74.38），且蒸馏种子波动增至 ±1.1–1.4。
6. 口径提醒：这里减少的是 **VAE 预训练分类头**的标签；云端仍用完整的 4900 个
   代表特征及其标签训练，不是「云端少标签」实验。

### 5.9 复现检查与风险

1. CIFAR-10 复现：Full 90.17、K-means 90.11、Herding 90.04、Random 89.93，
   与既有 90.16 / 89.82 / 90.07 结果同量级（差异来自云端分类器与种子协议）。
2. BDM 的 checkpoint 由 1000 张真实验证集选点，单次验证精度有 ±0.3 量级的噪声。
   C10 20:1 选中 step 500（验证 89.10），其测试为 89.52，是表中最大的选点偏差。
   所有方法共用同一协议；不按测试集回调。
3. CIFAR-100 100:1 的 Random 只剩每类 4–5 个真实特征，云端 MLP 在 490 个样本上
   严重过拟合（41.37）；这是该协议下的真实结果，不是脚本错误。
4. 量化在蒸馏后的代表特征上拟合 scale/zero-point，测试特征复用同一组参数；
   FP16/INT8 的精度变化均 ≤0.09 点，说明该潜空间特征对 8 bit 量化不敏感。
5. 本阶段只模拟一次性上传；在线推理仍需为每张测试图上传 1024 维特征（4 KB FP32 / 1 KB INT8），
   这与训练集上传量的压缩是两件事。
6. 主表（§5.1–5.7）的 VAE 分类头使用全部 49k 标签；§5.8 的 10% 版本只用 5k 标签。
   两者都不改变「重建+KL 使用全部训练图像」这一事实，因此都不能称为全流程无监督。

### 5.10 结果文件

- `results/{dataset}_baselines_r10.{csv,json}`
- `results/{dataset}_bdm_r10.{csv,json}`（含每个 DM seed 的完整 loss/var_ratio 曲线）
- `results/{dataset}_sweep.{csv,json}`（10/20/50/100:1）
- `results/{dataset}_main_r10_summary.csv` 与 `summary_*_r10.txt`
- `results/cifar10_head_linear_r10.json`（Linear 分类头对照）
- `results/lab500/`（10% 标签 VAE 的全套结果；脚本同上，仅 `--config cifar10_lab500.yaml`）
- `results/{dataset}_{method}_r{ratio}_quant.json`（FP32/FP16/INT8）
- `results/payload/`（真实量化文件，`.bin` + `.json`）
- `results/reps/`（各方法代表特征，`.npz`）
- `figures/{dataset}_acc_vs_ratio.png`、`acc_vs_bytes.png`、`f1_vs_bytes.png`、
  `main_r10.png`、`bdm_variance.png`

### 5.11 v2 协议修复与统一重跑（2026-11-14）

> §5.1–5.10 的旧结果保持不变，视为 **legacy**。本节全部结果在 `vae_bdm/results/v2/`，
> `protocol_version=vae_bdm_v2`。
> **当前代码（`bdm.py` / `cloud.py`）只保留 v2 协议**：旧的 random_class、单张量优化器、
> 全局 anchor、internal 10% 验证均已删除；`configs/cifar10.yaml`、`configs/cifar100.yaml`
> 仅作为历史配置存档，不能再跑 BDM。

**修复内容**

1. 类别更新：只保留 `balanced`（每轮随机排列全部类别，每类更新次数严格相等）；
   旧 `random_class` 已删除。
2. AdamW 动量隔离：每类一个 `nn.Parameter` + 独立 `AdamW`；非当前类别完全不更新。
   旧「单张量 + 单优化器」路径已删除。
3. Anchor：只作用于当前类；旧全局 anchor 已删除。
4. 云端验证：只保留 external（独立真实验证集，全部代表特征参与训练）与 none；
   旧 internal 10% 已删除。
5. BDM 选点：`bdm_validation.head/epochs/seed` 可配置（v2 默认 MLP/50），
   并支持 `selection: final`（固定终止步数，不按验证选点）。
6. 训练预算：`fixed_epochs` 与 `fixed_steps` 两种模式，记录实际优化步数。
7. 协方差：`use_full_stats=true` 时使用预计算的完整类别投影协方差；
   代表特征 <2 时关闭并记录原因/秩信息。
8. 数值与效率：`log_every` 控制标量同步；仅 CUDA 统计峰值显存；输入合法性检查；
   云端训练使用独立 RNG 作用域，调用后恢复外部随机状态。
9. 新增解析式基线 **Moment Calibration**：`K-means + Moment`、`Random + Moment`。

**协议对照（关键差异）**

| 项 | legacy（历史，已移除） | v2（当前唯一协议） |
|---|---|---|
| 云端验证 | 蒸馏特征内部 10% | 独立真实验证 1000，全部代表特征参与训练 |
| C10 100:1 实际训练代表数 | 440 / 490 | 490 / 490 |
| C100 100:1 内部验证 | **0 个（100 类全为空，实际无早停）** | 490 / 490 + 1000 真实验证 |
| BDM 更新 | `random_class` 3000 步 | `balanced`，C10 300/类、C100 30/类 |
| BDM 参数 | 单张量 + 单 AdamW | 每类参数 + 独立 AdamW |
| BDM Anchor | 全局 | 类内 |
| BDM 选点 | Linear 50 epoch | MLP 50 epoch（可配置） |

**CIFAR-10 10:1 新旧对照**

| 方法 | legacy | v2 |
|---|---:|---:|
| Full（49k） | 90.17 | 90.19 |
| Random | 89.93 | 89.90 |
| Mean | 88.81 | 89.05 |
| K-means | 90.11 | 90.09 |
| Herding | 90.04 | 90.06 |
| K-means + Moment | — | 90.05 |
| Random + Moment | — | 89.99 |
| BDM（旧算法 + 旧协议） | 89.87 | — |
| BDM（v2 算法 + v2 协议） | — | 89.85 |

**CIFAR-100 10:1 新旧对照**

| 方法 | legacy | v2 |
|---|---:|---:|
| Full（49k） | 61.20 | 61.56 |
| Random | 58.25 | 58.30 |
| Mean | 56.59 | 56.88 |
| K-means | 59.02 | 59.23 |
| Herding | 59.71 | 59.75 |
| K-means + Moment | — | **60.85** |
| Random + Moment | — | 60.71 |
| BDM（旧算法 + 旧协议） | 60.24 | — |
| BDM（v2 算法 + v2 协议） | — | 60.37 |

**高压缩比（v2，3 云端种子）**

> legacy 列仅作历史对照。旧算法在 v2 协议下跑出的 `bdm_legacy` 行已从主 JSON 拆到
> `{dataset}_bdm_legacy_r10.json` 归档，当前代码不能再生成该协议。

CIFAR-10：

| 压缩比 | Random | Mean | K-means | Herding | K-means+Moment | Batch DM | Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50:1 | 89.53 | 88.83 | **89.66** | 89.64 | 89.22 | 89.18 | 90.19 |
| 100:1 | 89.27 | 89.05 | 89.58 | **89.65** | 89.26 | 89.43 | 90.19 |

CIFAR-100：

| 压缩比 | Random | Mean | K-means | Herding | K-means+Moment | Batch DM | Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20:1 | 55.49 | 55.68 | 59.13 | 58.61 | 59.68 | **60.04** | 61.56 |
| 50:1 | 48.72 | 55.27 | 58.36 | 55.38 | 58.72 | **59.26** | 61.56 |
| 100:1 | 41.16 | 55.03 | 57.80 | 50.92 | 57.06 | **58.29** | 61.56 |

**Moment Calibration 对照（10:1，3 云端种子）**

| 数据集 | 基础方法 | 基线 Top-1 | +Moment Top-1 | 均值 MSE 前→后 | 方差 MSE 前→后 |
|---|---|---:|---:|---:|---:|
| C10 | K-means | 90.09 | 90.05 | 3.62e-3 → 3.28e-15 | 7.34e-2 → 5.79e-13 |
| C10 | Random | 89.90 | 89.99 | 1.14e-3 → 2.39e-15 | 2.62e-3 → 8.58e-15 |
| C100 | K-means | 59.23 | **60.85** | 1.02e-2 → 2.75e-15 | 1.40e-1 → 1.24e-12 |
| C100 | Random | 58.30 | 60.71 | 1.41e-2 → 2.01e-15 | 3.43e-2 → 8.24e-14 |

解读：
- Moment Calibration 精确匹配类均值/方差（MSE→1e-12），但**是否提点强烈依赖数据集与压缩比**。
  C100 10:1 提升 1.6–2.4 点；C10 10:1 无收益（-0.04）；C10 50/100:1 反而掉 0.4 点；
  C100 100:1 也掉 0.7 点。低预算下逐维方差估计噪声大，强行匹配会放大噪声。
- v2 的 BDM 修复没有显著改变 10:1 精度（C10 -0.05，C100 -0.10，均在蒸馏种子波动内），
  但修正了「非当前类被 AdamW 动量更新」「anchor 权重随类别数缩放」「内部验证为空」等协议问题。
- C100 高压缩比下 BDM 在 20/50/100:1 都是最好（60.04 / 59.26 / 58.29），
  说明平衡更新 + 类内优化器在「少代表/多类别」时更稳；C10 上仍是 K-means/Herding 最好。

**更新次数消融（CIFAR-100 10:1，seed 42）**

| 每类更新 | 总优化步数 | Top-1 |
|---|---:|---:|
| 30（主表，3 蒸馏种子均值） | 3,000 | 60.37（seed42 60.61） |
| 100 | 10,000 | 60.87 |
| 300 | 30,000 | 60.86 |

30→100 有 +0.26 的增益（同 seed 42），100→300 饱和。旧协议 C100 平均每类只有 30 次更新，
这是旧 BDM 在 C100 被低估的原因之一。

**选点协议消融（v2 算法，seed 42）**

| 选点方式 | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Linear 50 epoch | 90.04 | 59.87 |
| MLP 50 epoch | 89.96 | 60.61 |
| MLP 100 epoch | 89.96 | 60.31 |
| 固定终止步数（不选点） | 90.04 | 60.61 |

本表均为蒸馏种子 42；主表 `bdm_mean_var` 是 3 个蒸馏种子的均值。
选点协议的影响强烈依赖数据集：C100 上 MLP 选点比 Linear 高 **0.74**；
C10 上 Linear/固定终止反而高 0.08。因此不同选点协议的结果不能混表。

**训练预算消融（50:1 / 100:1，3 云端种子）**

| 数据集/压缩比 | 方法 | fixed_epochs（200） | fixed_steps（4000） |
|---|---|---:|---:|
| C10 50:1 | K-means | 89.66（800 步） | 89.67（4000 步） |
| C10 50:1 | Batch DM | 89.18（800 步） | 89.18（4000 步） |
| C100 100:1 | K-means | 57.80（400 步） | 57.79（4000 步） |
| C100 100:1 | Batch DM | 58.29（400 步） | 58.21（4000 步） |

固定步数没有带来增益：在这些压缩比下 MLP 已饱和，计算预算不是瓶颈。
该消融只用于说明「200 epoch 与更多步数结论一致」，不能与不同预算的结果混表。

**测试与审计**

```bash
python -m pytest vae_bdm/tests/test_protocol_v2.py -q     # 12 passed
python -m vae_bdm.scripts.audit_protocol                  # 生成 protocol_audit.json
```

审计文件：`results/v2/audit/protocol_audit.json`。关键发现：legacy 在 C100 100:1
内部验证为空（100/100 类），C10 100:1 只训练 440/490 个代表；v2 统一为
「全部代表参与训练 + 1000 张真实验证」，且结果行记录 `n_cloud_train/n_cloud_val/
validation_mode/cloud_optimizer_steps`。

**v2-only 代码重跑复现**

删除旧协议分支后，用相同配置/种子在独立目录 `results/v2_rerun/` 重跑 C10/C100 10:1
全部方法：**所有方法 Δ=0.00**，BDM 的三个蒸馏种子也逐位一致。说明协议清理没有改变
v2 的训练数值，旧分支删除是纯代码清理。

## 6. 口径与风险

1. 分组使用标签；VAE 预训练使用 49k 标签；不能宣称无监督。
2. Random 基线是“类内随机选真实特征”，不是“随机分组后取均值”；两者数值不同。
3. 1024 元素 ≠ 1024 字节；报告字节必须带位宽。
4. K-means/Mean 可能降低类内方差；BDM 的 `var_ratio` 曲线与 `bdm_variance.png` 专门检查。
5. 云端分类器始终从随机初始化开始；Full 只作为参考。
6. 验证集只用于 BDM checkpoint 选择与云端早停；测试集只做最终评估。
7. BDM 每 500 步在内存中评估验证精度并保留最优 `S`；磁盘只落盘最终选中的代表特征
   （约 20 MB/份），不保存全部中间快照。每步的 mean/var/anchor/div/cov loss、
   `var_ratio`、`syn_norm` 都在 JSON 的 `dm_logs` 中。
8. 训练阶段显存峰值约 0.23–0.24 GB（C10/C100，`M=4900`）；单步耗时约 5.4 ms
   （C10 10:1，3000 步约 16 s，不含 K-means 初始化）。
