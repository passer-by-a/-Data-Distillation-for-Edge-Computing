# 交接文档

> 最后一次更新：2026-11-14
> 项目根目录：`/root/autodl-tmp/diffusion distillation`
>
> **当前主线：VAE 潜空间数据集蒸馏**（脚本在 `legacy/`，历史目录名）。
> DLDD 跨样本蒸馏模块已移入 `diffusion_dldd/`；DDPM vs VAE 对照脚本暂停。

---

## 1. 项目目标

面向**边缘—云协同**场景：边缘侧把图像压缩成潜空间（16×8×8 = 1024 元素），传输到云端分类。
核心问题是：**用什么方式学这个 1024 维潜空间最好？**

两条候选路线：
- **VAE**：重建 + KL 压缩，可加分类监督（CE）
- **DDPM**：去噪预训练，取其 U-Net 中间层作为特征

---

## 2. 当前结论（一页版）

### 2.1 精度（CIFAR-10，1024 元素，t=20，3 seeds）

| 方法 | 预训练用标签 | Top-1 |
|---|---|---|
| VAE-S（重建+KL+CE） | 全部 49k | **90.45±0.06** |
| DDPM-FT（去噪→CE 微调） | 微调阶段 49k | **89.91** |
| DDPM-Joint（去噪+CE 联合） | 49k | 85.62 |
| DDPM-A（冻结中间层 + 线性探针） | **0** | **79.07±0.18** |
| DDPM-B1（冻结 16×16 层） | 0 | 72.59±0.31 |
| VAE（纯重建+KL） | 0 | 38.96±0.11 |

**同等监督下两者持平（89.91 vs 90.45，差 0.54）；无监督下 DDPM 大幅领先（79.07 vs 38.96，+40.1）。**

### 2.2 成本（batch=1，RTX 4090）

| | VAE | DDPM-A | 倍数 |
|---|---|---|---|
| 使用路径参数 | 431,200 | 8,425,344 | 19.5× |
| 全模型参数 | 900,365 | 38,383,107 | 42.6× |
| 推理 FLOPs | 253 M | 3,241 M | 12.8× |
| 单图延迟 | 0.80 ms | 4.04 ms | 5.1× |

**结论：DDPM 在多花 12.8 倍算力的前提下，精度反而低 0.54 个点。除非完全没有标签，否则选 VAE。**

### 2.3 表征的标签成本（关键发现）

下游一律给全标签时：

| 方法 | 预训练标签 | 精度 |
|---|---|---|
| DDPM | **0** | **79.07** |
| VAE | 5,000（500/类） | 66.49 |
| VAE | 49,000（全部） | 83.25 |

**DDPM 零标签达到的水平，VAE 要花 5,000 个标签还追不上，需全部 49k 才能反超。**

图：`figures/label_cost_representation.png`

### 2.4 CIFAR-100 的反转

| 方法 | CIFAR-10 | CIFAR-100 |
|---|---|---|
| DDPM-A（8×8） | **79.07（最好）** | 37.48（第二） |
| DDPM-B1（16×16） | 72.59 | **38.16（最好）** |
| VAE-S | 90.45 | 60.76 |

10 类时深层 8×8 领先 6.5 点；100 类时反而落后 0.68 点。推测是类数增加后 8×8 的空间压缩挤掉了判别信息。

---

## 3. 已完成实验清单

### 3.1 主对比（`experiments.py core`）

| 实验 | 结果文件 | 说明 |
|---|---|---|
| 时序 t=0/20/50/100/200 | `results/core_c10_tsweep_t*.csv` | t=20 最优，t=200 崩溃 |
| 主表 t=20，3 seeds | `results/core_c10_t20_3seed.csv` | A/B1/B2 |
| VAE 有/无监督 | `results/core_c10_t0_vae.csv` | 90.45 / 38.96 |
| CIFAR-100 | `results/core_c100_t20.csv` | B1 反超 A |
| 投影/GAP 消融 | `results/core_c10_compress.csv` | 1024 投影无损，GAP 有损 |
| 分类头消融 | `results/core_c10_mlp.csv` | MLP 比 Linear 高 3–8 点 |

### 3.2 监督条件（`experiments.py supervised`）

| 实验 | 结果文件 | 说明 |
|---|---|---|
| DDPM-FT | `results/core_c10_ft.csv` | 自带头 89.91 / 冻结探针 88.56 |
| DDPM-Joint | `results/core_c10_joint.csv` | 自带头 85.62 / 冻结探针 84.49 |

### 3.3 标签预算

| 实验 | 脚本 | 结果文件 |
|---|---|---|
| 下游分类器少标签 | `experiments.py core --shot N` | `results/core_c10_shot*.csv` |
| 预训练 CE 少标签 | `label_budget.py pretrain` | `results/vae_shot_c10*.json` |
| 预训练少标签 + 下游全标签 | `label_budget.py transfer` | `results/vae_pretrain_shot_fullprobe.json` |
| 零参数分类器（NCC/1-NN） | `label_budget.py zeroshot` | `results/zeroshot_probe.json` |

### 3.4 取层验证（`experiments.py decoder-probe`）

在 `ddpm_cifar10.pt` 上同协议对比（1000 点、ResNet 探针、t=10）：

| 位置 | 8×8 特征 |
|---|---|
| 编码器 `down_blocks.2` | 70.20 |
| 解码器 `up_blocks.1.resnets.0/1/2` | 73.50 / 78.75 / **82.32** |

解码器比编码器高 12 点且随融合次数单调上升。**但我们的流程用不了解码器特征**——它无法压成 1024 维。

---

## 4. 文件结构

### 4.1 核心模块

| 文件 | 职责 |
|---|---|
| `config.py` | 全局配置，`DATASET=cifar100` 时产物自动隔离到子目录 |
| `vae.py` | EdgeVAE（900K，16×8×8）+ 训练入口 |
| `diffusion.py` | 标准 DDPM（38.4M，diffusers UNet2DModel）+ 训练/采样 |
| `models.py` | **统一特征接口**：`DDPMBackbone.forward_features(layer=...)` 提前返回 |
| `data.py` | 固定划分（`RandomState(42)`）、固定噪声、特征提取 |
| `train.py` | 冻结主干，训练投影 + 下游分类器 |
| `common.py` | 种子、指标、FLOPs、延迟、CSV、权重路径 |
| `backbones.py` | 4 个下游 backbone，统一训练评估（与 legacy 共用） |
| `utils.py` | 测试集 latent 缓存、推理延迟/能耗解析模型（与 legacy 共用） |

### 4.2 实验入口

```
experiments.py core            # DDPM-A/B1/B2 vs VAE 主对比
experiments.py supervised      # DDPM-FT / DDPM-Joint
experiments.py decoder-probe   # 编码器 vs 解码器取层

label_budget.py pretrain       # 预训练 CE 标签预算
label_budget.py transfer       # 预训练少标签 + 下游全标签
label_budget.py zeroshot       # 零参数分类器
```

旧蒸馏线在 `legacy/`，从根目录运行需加 `PYTHONPATH=.`：

```
legacy/exp_ipc.py              # 压缩率—精度曲线
legacy/exp_method_compare.py   # 方法对比
legacy/exp_energy_methods.py   # 分阶段 FLOPs / 能耗 / 传输时延
legacy/latent_prior.py         # 生成式先验
legacy/gen_encoder.py          # Flow / 先降维再扩散
legacy/exp_full_pool.py        # 全量池 K-means 蒸馏评估
```

### 4.3 绘图

```
plots.py --fig all        # 三张对比图 + 四张消融图 + summary_all.csv
plots.py --fig ablation   # 只画消融四图
plots.py --fig cls-loss   # 旧蒸馏线的 VAE 分类监督 loss 曲线
```

### 4.4 文档

| 文件 | 内容 |
|---|---|
| `README.md` | 主项目说明、命令、关键口径 |
| `COMPARE.md` | DDPM vs VAE 对比的完整记录与偏差说明 |
| `METRICS.md` | 测量口径 |
| `HANDOVER.md` | 本文件 |

### 4.5 DLDD 独立模块（已存档，2026-11-14）

**状态：Diffusion 线暂停，后续回到 VAE 主线。**
存档：`diffusion_dldd/diffusion_dldd_20261114.tar.gz`（代码+结果+图，5.5MB；缓存/中间 reps 未打包）。
状态摘要：`diffusion_dldd/DIFFUSION_SNAPSHOT_20261114.md`。

`diffusion_dldd/exp_diffusion_feature_distillation/`：跨样本训练特征蒸馏（N→M，仍是 1024 维/条）。

```
scripts/extract_features.py    # DDPM-A t=20 → 1024 维缓存（含隔离检查/监督标记）
scripts/run_baselines.py       # Full/Random/Mean/K-means/Herding/Attention-DM
scripts/run_compression_sweep.py
scripts/run_quantization.py    # FP32/FP16/INT8 后重训云端分类器
scripts/train_aggregator.py    # Attention/DM/GM/Anchor 消融
scripts/verify_pipeline.py     # 维度/隔离/真实 payload 闭环
scripts/summarize.py           # 主表 CSV + Markdown
```

关键结论（详见该目录 README）：
- DDPM 10:1：Herding 78.42 > DLDD 77.19 ≈ K-means 77.16 > Attention-DM 76.41；
- VAE 10:1：DLDD 90.02 ≈ Full 90.16，远高于 DDPM；
- CIFAR-100 10:1：K-means 41.15 最好，学习式融合掉点；
- INT8 量化只掉 0.05 点，真实 payload 约 5.03 MB（10:1）。
- **1024 维投影使用了 49k 训练标签**，不能称端到端无监督。

### 4.6 VAE-BDM 第一阶段（`vae_bdm/`，2026-11-14）

> 2026-11-17 方法收敛：只保留 **Full / Random / K-means / K-means+Moment / 原版 DM**，
> Mean / Herding / Random+Moment 的代码与结果已彻底删除。
> 本节以下为收敛前的历史记录，当前口径与数字见 `vae_bdm/README.md` §2、§5。

冻结 EdgeVAE `mu` 特征（1024 维）上的静态数据集蒸馏，独立模块，不改旧产物。

- `configs/{cifar10,cifar100}.yaml`、`features.py`、`baselines.py`、`bdm.py`、
  `cloud.py`、`quant.py`、`evaluate.py`
- `scripts/{extract_features,run_baselines,run_bdm,run_sweep,run_quant,summarize,make_figures}.py`
- 产物：`vae_bdm/cache/`、`vae_bdm/results/`、`vae_bdm/figures/`
- 协议、结果与风险：`vae_bdm/README.md`

关键结论（C10/C100，3 云端种子）：BDM 在 C10 10:1 为 89.87（K-means 90.11、Full 90.17），
不占优；C100 10:1 为 60.24（Herding 59.71、K-means 59.02、Full 61.20），领先。
方差匹配把类内方差比从 0.61–0.65 拉回 0.95–0.99：C100 提点 +0.45，C10 持平略负。
INT8 量化精度损失 ≤0.09 点；10:1 实际上传负载 5.03 MB（较 FP32 Full 39.9×），
100:1 为 0.503 MB（399×）。

标签预算（`vae.py --label_shot 500`，分类头 500/类 = 5k 标签，重建+KL 仍用全部图像）：
C10 10:1 的 Full 上界从 90.17 降到 78.02；Herding 75.91 最好，Random 74.91，
BDM mean+var 72.99 ≈ K-means 72.86。10/20/50/100:1 全部是 Herding 最好（74.64–75.91），
BDM 71.28–72.76。**弱特征空间里「选真实样本」明显优于「生成矩匹配特征」**。
权重 `checkpoints/autoencoder_edge_s8_c16_lab500aug.pt`，结果 `vae_bdm/results/lab500/`。

### 4.7 VAE-BDM v2 协议（2026-11-14）

> 2026-11-17 起本模块只对比三个方法（K-means / K-means+Moment / 原版 DM），
> 另保留 Random 与 Full 作为下界/上界；Herding、Mean、Random+Moment 已删除。
> 当前结果表见 `vae_bdm/README.md` §5（C100 10:1：Moment 60.85 ≥ DM 59.61 > K-means 59.23）。

修复：balanced 类别更新、每类参数 + 独立 AdamW、类内 anchor、external 验证集、
BDM 选点可配置（head/epochs/seed/final）、fixed_steps 预算、协方差统计口径、
RNG 隔离、输入检查、Moment Calibration 基线。
**`bdm.py` / `cloud.py` 已删除旧协议分支**（random_class、单张量优化器、全局 anchor、
internal 10% 验证），只保留 v2；旧配置仅作历史存档。

- 配置：`vae_bdm/configs/cifar{10,100}_v2.yaml`（`protocol_version: vae_bdm_v2`）
- 结果：`vae_bdm/results/v2/`；旧结果原地保留为 legacy
- 测试：`python -m pytest vae_bdm/tests/test_protocol_v2.py -q`（12 passed）
- 审计：`vae_bdm/results/v2/audit/protocol_audit.json`
  - legacy C100 100:1 内部验证为空（100/100 类），实际无早停
  - legacy C10 100:1 只有 440/490 个代表参与分类器训练
- 关键结果：C100 10:1 K-means+Moment 60.85 > BDM 60.37–60.47 > Herding 59.75 >
  K-means 59.23；C100 20/50/100:1 均为 BDM 最好（60.04/59.26/58.29）；
  C10 仍是 K-means/Herding 最好；Moment 在 C10 与高压缩比无益甚至掉点。
- 选点影响：C100 上 MLP 选点比 Linear 高 0.74；C10 上 Linear 略高 0.08，禁止混表。

---

## 5. 关键口径（改代码前必读）

1. **1024 元素不等于等算力。** VAE 输出 1024 维原样使用；DDPM 输出 16,384 维再投影到 1024。FLOPs 差 12.8 倍。
2. **DDPM 的提前返回是真的。** `forward_features` 只跑到 `down_blocks[2]`，不执行 `down_blocks[3]`、`mid_block` 和 Decoder。若部署框架必须跑完整图，成本会回到 38.4M 那一档。
3. **特征取自 `res_samples` 的下采样前张量。** Diffusers 的 `DownBlock` 返回值是下采样**后**的，直接用会得到 8×8 而非 16×16。
4. **t=0 不等于无噪声。** 按定义 `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε`，t=0 时仍加 `√(1-0.9999)≈0.01` 的噪声。
5. **EMA 衰减率必须匹配步数。** FT 只有 8000 步，用 0.9999 会让 `0.9999^8000≈0.45` 的随机初始化残留，自带头从 89.91 掉到 79.32。预训练 100k 步用 0.9999 才合适。
6. **两套分类器不要混。** VAE 内部的 `classifier` 只在预训练时用（注入标签），推理时丢弃；下游分类器才是产出精度的那个。
7. **数据划分固定**：`RandomState(42).permutation(50000)`，前 2% 验证。`DATASET=cifar100` 时同样逻辑。

---

## 6. 已知问题 / 未完成

| 项 | 状态 | 说明 |
|---|---|---|
| **VAE 标签预算缺数据增强** | ⚠️ 偏低 | `label_budget.py pretrain` 未用随机翻转/裁剪，全量点只有 83.18 而非 90.45。曲线整体偏低 3–7 点，但相对关系不变 |
| **CIFAR-100 的 FT/Joint** | ❌ 未做 | 只在 CIFAR-10 上跑过 |
| **t=0 输入分布消融** | ❌ 未做 | clean image vs 按定义加噪 |
| **λ 敏感性** | ❌ 未做 | Joint 只用 λ=0.1 |
| **目标硬件实测** | ❌ 未做 | 所有 FLOPs/延迟来自 4090，边缘 SoC 需实测 |
| **量化实验** | ❌ 未做 | 传输场景的核心：fp32 潜变量(4096B) 比原图(3072B) 还大，必须量化才有压缩收益 |

---

## 7. 环境与复现

```bash
# 依赖
pip install torch torchvision diffusers scikit-learn matplotlib pandas tqdm

# 主对比（约 40 分钟）
python experiments.py core --dataset cifar10 --t 20 --modes A,B1,B2,VAE --seeds 3 --epochs 30

# 监督实验
python experiments.py supervised --mode ft    --steps 8000 --ema 0.999
python experiments.py supervised --mode joint --steps 100000

# 标签预算
python label_budget.py pretrain --shots 5,20,100,500,4900
python label_budget.py transfer
python label_budget.py zeroshot --shots 1,5,20,50

# 图
python plots.py --fig all
```

**GPU**：RTX 4090 D（24GB）。38.4M DDPM 训练 100k 步约 3.6 小时；900K VAE 100 epoch 约 15 分钟。

**权重**：`checkpoints/` 下。CIFAR-100 的在 `checkpoints/cifar100/`。

---

## 8. 建议的下一步（VAE 主线，2026-11-14）

**刚完成**：CIFAR-100 主线补全（latent 口径、IPC=50、3 seeds）
- 方法对比：Ours 46.89 > GM 46.34 > EL2N 45.46 > Forgetting 45.11
- 方法 × backbone：VGG/ResNet 上 Ours 最好；MobileNet/ViT 上 EL2N/Forgetting 反超
- 通信：Ours 4.88 MB / 8.19 s vs 像素基线 14.65 MB / 24.58 s
- 图与 JSON 已落盘（见 README §6）

待办：
1. **CIFAR-100 7 方法先验对照**（ours + 3 基线 + gaussian/flow/diffusion）——
   现有先验权重是 IPC=100，需确认是否按 IPC=50 重跑。
2. **真实硬件实测**——当前能耗/时延为解析估算。
3. **论文表格整理**——C10/C100 主表、方法×backbone、通信成本统一口径。

---

## 9. 一句话总结（DDPM 对照线，已归档）

> 在 1024 元素预算下，**有标签时 VAE 与 DDPM 精度持平（90.45 vs 89.91），但 DDPM 贵 12.8 倍**；
> **无标签时 DDPM 显著更强（79.07 vs 38.96），相当于替 VAE 省下 5 万到 49 万个标签**。
> 因此 DDPM 的定位不是"更好的压缩器"，而是"标签稀缺时的替代方案"。该线已移入 `diffusion_dldd/`。
