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
- 能耗/时延均为**理论解析估算**，只计下游分类器（不含编码器），详见 `METRICS.md`。

---

## 2. 目录结构

| 路径 | 说明 |
|---|---|
| `config.py` | 全局配置（数据集切换、路径、EdgeVAE 超参、DataLoader） |
| `vae.py` | EdgeVAE 定义、训练与 `load_edge_vae` |
| `distill.py` | 潜空间蒸馏（mean / medoid / herding），产出合成潜变量 |
| `backbones.py` | 4 个下游 backbone、`PixelResNet`、参数量/FLOPs、统一训练评估 |
| `utils.py` | 测试集 latent 缓存、推理延迟/能耗解析模型、硬件常数 |
| `analysis.py` | FLOPs 统计/格式化、压缩率—能耗估算、潜空间可视化 |
| `diffusion.py` | DDPM 对照线（扩散上界实验，暂保留） |
| `exp_ipc.py` | 压缩率—精度曲线（论文主表/主图） |
| `exp_ablation.py` | VAE 分类监督消融 |
| `exp_method_compare.py` | 方法对比（像素基线 + 4×4 网格图） |
| `exp_refine.py` | K-means 质心 vs 梯度精炼 |
| `exp_energy_methods.py` | 分阶段 FLOPs / 能耗 / 传输时延 |
| `plot_acc_loss.py` | 消融 loss 图 + 共享绘图工具函数 |

运行产物路径由 `config.py` 统一管理；设置环境变量 `DATASET=cifar100` 时，
`pt/ results/ checkpoints/ logs/` 会自动隔离到对应子目录。

---

## 3. 环境依赖

```bash
# Python >= 3.10
pip install torch torchvision numpy scikit-learn matplotlib thop
```

- 训练/蒸馏需要 NVIDIA GPU；当前环境 `torch.cuda.is_available() == False`，
  本轮仅做了静态语法检查，训练与出图命令需在 GPU/本机环境执行。
- `matplotlib` 使用 `Agg` 后端，无图形界面也可出图。
- 当前目录路径包含空格，命令行请始终加引号，或使用 `cd "..."`。

---

## 4. 快速开始

```bash
# 0) 数据：首次运行会自动下载 CIFAR-10 到 ./data（可用 CIFAR10_MIRROR 换镜像）
python -c "import config; config.get_cifar10_loaders()"

# 1) 训练 EdgeVAE（阶段 1）
python vae.py --latent_dim 16 --epochs 100

# 2) 潜空间蒸馏（阶段 2，IPC=100，源池每类 1000 张）
python distill.py --ipc 100 --subset 1000 --center mean

# 3) 压缩率—精度主实验（论文表 I / 图 1）
python exp_ipc.py --auto

# 4) 方法对比（论文表 III / 图 3）
python exp_method_compare.py              # 像素/潜空间基线 + ResNet/ViT
python exp_method_compare.py --multi_backbone   # 4 方法 × 4 backbone 网格图
```

### 论文图 ↔ 命令

| 产出 | 命令 |
|---|---|
| `accuracy_compression_tradeoff.*` | `python exp_ipc.py --auto` |
| `ablation_vae_cls_loss_train.*` | `python exp_ablation.py` 后执行 `python plot_acc_loss.py` |
| `latent_visualization.*` | `python analysis.py latent` |
| `method_x_backbone_cifar10_ipc100_pixel.*` | `python exp_method_compare.py --multi_backbone`（或 `--multi_plot_only <json>`） |
| `flops_by_method_*`、`energy_by_method_*`、`transfer_latency_by_method_*` | `python exp_energy_methods.py --ipc 100` |

> `exp_method_compare.py` 支持 `--plot_only` / `--multi_plot_only`，
> 已有 `results/*.json` 时可直接重绘，无需 GPU。

---



```

---
