# 潜空间数据集蒸馏与端—边协同

在资源受限的边缘设备上，用**冻结的 EdgeVAE** 把图像压缩到 16×8×8（1024 维）潜空间，
在潜空间内完成**数据集蒸馏**（逐类 K-means 质心），下游分类器**直接读取潜变量**训练，
并把合成 latent（int8）上传到边缘端，从而同时降低带宽、训练算力与推理能耗。

> 指标口径与公式见 [METRICS.md](METRICS.md)。

---

## 1. 方法流程

```
CIFAR-10/100 ──冻结 EdgeVAE 编码──> 潜空间 [N,16,8,8]
                                        │
                          逐类 K-means（IPC 个质心）
                                        │
                        合成潜变量 [C,IPC,16,8,8]（int8 上传）
                                        │
             边缘端下游分类器（VGG/ResNet/MobileNet/ViT，~28–32K 参数）
```

- 解码器不参与下游推理，分类全程在潜空间完成。
- 基线方法（Gradient Matching / EL2N / Forgetting）分别在像素空间或潜空间实现，
  由 `--baseline_repr {pixel,latent}` 切换，两种口径结论不同，论文中不可混用。
- **生成式先验对照**（`legacy/latent_prior.py`）：冻结同一个 EdgeVAE 潜空间不变，只替换
  「如何从类条件分布取 IPC 个点」——逐类高斯采样 / RealNVP 流采样 / latent DDPM 采样，
  三者输出同为 `[C,IPC,16,8,8]`，与 K-means 质心（Ours）同尺度、同 payload、同下游分类器。
  注意 Flow 本身保维（不压缩），3× 压缩全部由 EdgeVAE 完成。
- 能耗/时延均为**理论解析估算**，只计下游分类器（不含编码器），详见 `METRICS.md`。

---

## 2. 目录结构

### 2.1 VAE 潜空间蒸馏主线（`legacy/`）

> `legacy/` 是历史目录名；它现在是论文主线（VAE 潜空间蒸馏），不是废弃代码。

| 路径 | 说明 |
|---|---|
| `legacy/distill.py` | 潜空间蒸馏（mean / medoid / herding），产出合成潜变量 |
| `legacy/exp_ipc.py` | 压缩率—精度曲线（论文主表/主图） |
| `legacy/exp_method_compare.py` | 方法对比（4 基线 + 可选 3 个生成式先验） |
| `legacy/exp_energy_methods.py` | 分阶段 FLOPs / 能耗 / 传输时延 |
| `legacy/exp_ablation.py` | VAE 分类监督消融 |
| `legacy/exp_refine.py` | K-means 质心 vs 梯度精炼 |
| `legacy/latent_prior.py` | 潜空间生成式先验（高斯 / RealNVP / latent DDPM） |
| `legacy/gen_encoder.py` | 纯 Flow 与「先降维再扩散」+ K-means |
| `legacy/analysis.py` | FLOPs 统计、压缩率—能耗估算、潜空间可视化 |
| `legacy/exp_flat_probe.py` / `exp_flat_full.py` | 3×32×32 同形潜空间对照 |
| `legacy/exp_full_pool.py` | 全量池 K-means 蒸馏评估 |
| `legacy/exp_vae_1k.py` / `probe_eval.py` / `show_synthetic.py` | 早期诊断脚本 |

> 从仓库根目录运行，需加 `PYTHONPATH=.`：
> `PYTHONPATH=. python legacy/exp_ipc.py --auto`。
> 结果与图仍写入根目录的 `results/` / `figures/`。

### 2.2 共用库

| 路径 | 说明 |
|---|---|
| `config.py` | 全局配置（`DATASET=cifar100` 切换数据集与产物目录） |
| `vae.py` | EdgeVAE（16×8×8）定义、训练与加载 |
| `backbones.py` | 4 个下游 backbone、`PixelResNet`、参数量/FLOPs、统一训练评估 |
| `utils.py` | 测试集 latent 缓存、推理延迟/能耗解析模型、硬件常数 |

### 2.3 对照线：DDPM vs VAE（已归档）

DDPM 的跨样本蒸馏模块（DLDD）已整体移入 `diffusion_dldd/`，当前暂停；
DDPM vs VAE 的对照脚本（`diffusion.py`、`models.py`、`experiments.py`、
`label_budget.py`、`plots.py` 等）仍暂留根目录，不再作为主线。
入口见 `diffusion_dldd/exp_diffusion_feature_distillation/README.md`，
运行时加 `PYTHONPATH=diffusion_dldd`。

| 路径 | 说明 |
|---|---|
| `diffusion.py` | 标准 DDPM（38.4M）训练/采样（归档模块仍引用） |
| `models.py` / `data.py` / `train.py` / `common.py` | DDPM 特征接口与评估工具 |
| `experiments.py` / `label_budget.py` / `plots.py` | DDPM 对照实验与图 |

### 2.4 产物目录

| 路径 | 说明 |
|---|---|
| `figures/` | 论文图输出（pdf + png，运行时生成，**不入库**） |
| `results/` | 实验结果 JSON/CSV（运行时生成，**不入库**） |
| `data/ pt/ checkpoints/ logs/` | 数据集、潜变量产物、权重、日志（体积大，**不入库**） |

运行产物路径由 `config.py` 统一管理；设置环境变量 `DATASET=cifar100` 时，
`pt/ results/ checkpoints/ logs/` 会自动隔离到对应子目录。

---

## 3. 环境依赖

```bash
# Python >= 3.10
pip install torch torchvision numpy scikit-learn matplotlib thop
pip install diffusers accelerate tqdm        # 仅 diffusion.py / legacy/latent_prior.py 需要
```

- 训练/蒸馏需要 NVIDIA GPU（本仓库在 RTX 4090 24G 上验证，`torch.cuda.is_available()` 为 True）。
- `matplotlib` 使用 `Agg` 后端，无图形界面也可出图。
- 当前目录路径包含空格，命令行请始终加引号，或使用 `cd "..."`。

---

## 4. 快速开始

```bash
# ========== 对照线：DDPM vs VAE（已归档到 diffusion_dldd/） ==========
# 主对比（约 40 分钟）
python experiments.py core --dataset cifar10 --t 20 --modes A,B1,B2,VAE --seeds 3 --epochs 30

# 监督实验
python experiments.py supervised --mode ft    --steps 8000 --ema 0.999
python experiments.py supervised --mode joint --steps 100000

# 标签预算
python label_budget.py pretrain --shots 5,20,100,500,4900
python label_budget.py transfer
python label_budget.py zeroshot --shots 1,5,20,50

# 取层验证
python experiments.py decoder-probe --t 10

# 图
python plots.py --fig all

# ========== VAE 主线（legacy/） ==========
# 0) 数据：首次运行会自动下载 CIFAR-10 到 ./data（可用 CIFAR10_MIRROR 换镜像）
python -c "import config; config.get_cifar10_loaders()"

# 1) 训练 EdgeVAE（阶段 1）
python vae.py --latent_dim 16 --epochs 100

# 2) 潜空间蒸馏（阶段 2，IPC=100，源池每类 1000 张）
PYTHONPATH=. python legacy/distill.py --ipc 100 --subset 1000 --center mean

# 3) 压缩率—精度主实验（论文表 I / 图 1）
PYTHONPATH=. python legacy/exp_ipc.py --auto

# 4) 方法对比（论文表 III / 图 3）
PYTHONPATH=. python legacy/exp_method_compare.py --baseline_repr latent
PYTHONPATH=. python legacy/exp_method_compare.py --multi_backbone

# CIFAR-100（DATASET 切换数据集与产物目录）
DATASET=cifar100 PYTHONPATH=. python legacy/exp_ipc.py --auto
DATASET=cifar100 PYTHONPATH=. python legacy/exp_method_compare.py --ipc 50 --baseline_repr latent
DATASET=cifar100 PYTHONPATH=. python legacy/exp_energy_methods.py --ipc 50

# 5) 生成式先验对照
PYTHONPATH=. python legacy/latent_prior.py cache --ipc 100
PYTHONPATH=. python legacy/latent_prior.py fit   --ipc 100 --prior flow      --epochs 200
PYTHONPATH=. python legacy/latent_prior.py fit   --ipc 100 --prior diffusion --steps 40000 --lr 1e-3
PYTHONPATH=. python legacy/latent_prior.py sample --ipc 100 --prior flow      --seeds 3
PYTHONPATH=. python legacy/latent_prior.py sample --ipc 100 --prior diffusion --nfe 50 --seeds 3
PYTHONPATH=. python legacy/latent_prior.py fit   --ipc 100 --prior gaussian

# 6) 7 方法同表对比 + 资源三图
PYTHONPATH=. python legacy/exp_method_compare.py --methods ours,gradmatch,eln2,forgetting,gaussian,flow,diffusion
PYTHONPATH=. python legacy/exp_energy_methods.py --ipc 100

# 7) 纯 Flow / 先降维再扩散
PYTHONPATH=. python legacy/gen_encoder.py train   --model flow   --cls --epochs 100
PYTHONPATH=. python legacy/gen_encoder.py train   --model drdiff --epochs 100 --lr 1e-3
PYTHONPATH=. python legacy/gen_encoder.py distill --model flow   --ipc 100
PYTHONPATH=. python legacy/gen_encoder.py distill --model drdiff --ipc 100
PYTHONPATH=. python legacy/gen_encoder.py distill --model drdiff --ipc 100 --via sample
PYTHONPATH=. python legacy/gen_encoder.py eval    --model flow   --ipc 100 --seeds 3
PYTHONPATH=. python legacy/gen_encoder.py eval    --model drdiff --ipc 100 --seeds 3 --via sample
```

### 论文图 ↔ 命令

| 产出（`figures/`） | 命令 |
|---|---|
| `ddpm_vs_vae_frontier.png`、`label_budget.png`、`label_cost_representation.png` | `python plots.py --fig all` |
| `timestep_vs_accuracy.png`、`main_comparison.png`、`compression_ablation.png`、`head_ablation.png` | `python plots.py --fig ablation` |
| `ablation_vae_cls_loss_train.*` | `PYTHONPATH=. python legacy/exp_ablation.py` 后执行 `python plots.py --fig cls-loss` |
| `accuracy_compression_tradeoff.*` | `PYTHONPATH=. python legacy/exp_ipc.py --auto` |
| `latent_visualization.*` | `PYTHONPATH=. python legacy/analysis.py latent` |
| `method_x_backbone_cifar10_ipc100_pixel.*` | `PYTHONPATH=. python legacy/exp_method_compare.py --multi_backbone` |
| `flops_by_method_*`、`energy_by_method_*`、`transfer_latency_by_method_*` | `PYTHONPATH=. python legacy/exp_energy_methods.py --ipc 100` |

> `legacy/exp_method_compare.py` 支持 `--plot_only` / `--multi_plot_only`，
> 已有 `results/*.json` 时可直接重绘，无需 GPU。

---

## 5. 关键口径（写论文必须遵守）

1. 下游训练统一 50 epoch、Adam `lr=1e-3`、cosine 衰减、3 seeds；
   分类头前加 `Dropout(0.3)`，backbone 参数量对齐 ~28–32K。
2. 报告的精度一律取**最后一个 epoch**，不按测试曲线峰值挑 epoch。
3. 能耗 `E = FLOPs × 1e-11 J/FLOP`，训练 FLOPs ≈ 前向 × 3；
   传输 `latency = bytes×8 / 带宽`，`energy = bits × 1e-7 J/bit`。
4. 推理只计下游分类器前向，**不重复计 EdgeVAE 编码**（编码已计入蒸馏产出阶段）。
5. CIFAR-100 每类仅 500 张训练图，IPC 不得超过 491；`legacy/exp_ipc.py` 需自适应 `NUM_CLASSES`。
6. 生成式先验与 Ours 必须同池、同 IPC、同分类器、同 epoch/seed：
   - 先验只在 `real_subset`（每类 1000 张）的潜变量上拟合，不额外使用训练集标签；
   - 采样输出反标准化回 VAE `mu` 尺度，并断言形状 `[C×IPC,16,8,8]`；
   - 精度按「采样 seed × 分类器 seed」全部组合平均，**必须报告采样方差**；
   - 能耗口径中预训练为一次性成本（VAE 与先验训练），采样计入 production。

更多公式与硬件常数见 [METRICS.md](METRICS.md)。

---

## 6. CIFAR-100 主线结果（2026-11-14）

`DATASET=cifar100`，IPC=50（10:1，5000 点），冻结 EdgeVAE 潜空间，ResNet 下游、3 seeds、50 epoch。

### 6.1 方法对比（latent 口径）

| 方法 | Top-1 | Macro 口径备注 |
|---|---:|---|
| Ours (VAE + K-means) | **46.89±0.50** | 最后 epoch；最好 epoch 47.04 |
| Gradient matching (DC) | 46.34±0.55 | latent DC 精炼 |
| EL2N | 45.46±0.60 | 代理模型 top-50/类 |
| Forgetting | 45.11±0.18 | 代理模型 top-50/类 |

### 6.2 Ours 的跨 backbone（IPC=50）

| Backbone | Ours | 1:1 无压缩参考 |
|---|---:|---:|
| ResNet | 46.94 | 56.75 |
| ViT | **52.89** | 61.19 |
| VGG | 38.56 | 53.88 |
| MobileNet | 48.53 | 60.22 |

方法 × backbone（latent 口径，3 seeds）：

| Backbone | Ours | GM (DC) | EL2N | Forgetting |
|---|---:|---:|---:|---:|
| VGG | **38.57±0.53** | 38.05±0.35 | 35.74±0.59 | 34.02±0.50 |
| ResNet | **47.02±0.54** | 46.32±0.54 | 45.12±0.54 | 44.93±0.29 |
| MobileNet | 48.53±0.17 | 48.77±0.31 | **49.44±0.35** | 48.66±0.39 |
| ViT | 52.89±0.48 | 52.68±0.56 | **54.82±0.67** | 54.74±0.09 |

Ours 在 VGG/ResNet 上最好；MobileNet/ViT 上 EL2N/Forgetting 反超。
说明「K-means 质心」并非对所有 backbone 都占优，论文中不能写成全面最优。

### 6.3 通信与能耗（5 Mbps，int8 latent）

| 上传对象 | 字节 | 传输时延 |
|---|---:|---:|
| Raw 像素（488/类） | 143.25 MB | 240.33 s |
| Ours 合成 latent | **4.88 MB** | **8.19 s** |
| 像素基线（GM/EL2N/Forgetting） | 14.65 MB | 24.58 s |

结果文件：
`results/method_compare_cifar100_ipc50_latent.json`、
`results/cifar100/ipc_sweep_resnet_vit_vgg_mobilenet_cifar100.json`、
`results/energy_by_method_cifar100_ipc50.json`；
图：`figures/method_compare_cifar100_ipc50_latent_*`、
`figures/method_x_backbone_cifar100_ipc50_latent.*`、
`figures/{flops,energy,transfer_latency}_by_method_cifar100_ipc50.*`。

---

## 7. 仓库与 Git

仓库已通过 `.gitignore` 排除 `data/`、`pt/`、`checkpoints/`、`logs/`、`results/`、`figures/`
以及缓存与 IDE 文件；这些产物需在本地按 §4 的命令重新生成。

```bash
git init
git add .
git commit -m "Initial commit: latent dataset distillation for edge deployment"
git branch -M main
git remote add origin https://github.com/passer-by-a/-Data-Distillation-for-Edge-Computing.git
git push -u origin main
```

---

## 8. 已知注意事项

- 潜空间维度固定 `16×8×8`；`128×4×4` 的旧 VAE 口径已废弃。
- ConvNeXt 在小规模潜变量数据上过拟合严重，已从下游模型中移除。
- `PixelResNet` 未注册进 `BACKBONES`，仅在像素基线实验中使用。
- 复现实验前请确认 `pt/` 与 `checkpoints/` 中的产物存在，否则需先执行 §4 的步骤 1–2。
- 生成式先验产物命名：`pt/latent_prior_{prior}_ipc{ipc}.pt`（权重）、
  `pt/syn_latents_ipc{ipc}_{prior}_edge_s{seed}.pt`（合成潜变量，5D `[C,IPC,16,8,8]`）、
  `results/latent_prior_{prior}_ipc{ipc}.json`（成本与采样诊断，供 `legacy/exp_energy_methods.py` 读取）。
- 采样诊断里若 `class_purity` 明显偏低（<90%），说明类条件没学好，先调先验再跑下游。
