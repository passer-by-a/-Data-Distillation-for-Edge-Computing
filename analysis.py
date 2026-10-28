"""
分析线: 边缘部署能效估算 + 潜空间分布可视化

两个子命令:
  efficiency  蒸馏前后的 能耗 / 传输时延 / 压缩比 理论估算 (含两张图)
  latent      带分类头 vs 无分类头 EdgeVAE 的潜空间分布对比 (PCA / t-SNE)

============================================================
efficiency: 对比口径 (理论估算, 可复现)
  1. 数据压缩:   原始全量图片(uint8) vs 合成潜变量(int8 量化, 1 B/标量, 不同 IPC) 的字节数 / 压缩比
  2. 传输时延:   数据字节数 ÷ 带宽
  3. 传输能耗:   数据 bit 数 × 每 bit 传输能耗
  4. 训练能耗:   训练总 FLOPs × 推理能效 (样本数 ↓ 是蒸馏的核心收益)
  5. 推理能耗:   各 backbone 单次前向 FLOPs × 能效 (边缘部署选型)

  另外产出"训练计算量/能耗 vs 压缩率"双面板图 (仅计下游分类器训练, 不含 VAE 编码/蒸馏与传输):
    训练总 FLOPs = 合成点数 × epochs × (单样本前向 FLOPs × 3), 能耗 = FLOPs × 推理能效;
    横轴压缩率 = 源子集点数/类 ÷ IPC (与 exp_ipc 同口径), 并标出"原始全量"与"1:1 源子集"参考线。

  注意: 所有能耗/时延均为量级估算, 用于横向对比"蒸馏前后"的差异趋势,
       绝对数值取决于具体硬件/信道, 可通过命令行参数替换。

latent: 对 CIFAR 测试集子集分别用两个 VAE 编码得到潜向量, PCA / t-SNE 降到 2D 着色 (按类),
  对比"判别布局"是否存在。同时给出潜变量上的线性分类准确率作为佐证。
  并额外产出论文图"真实潜变量 vs K-means 中心代表"双面板 (两侧独立 t-SNE, 同色):
    figures/latent_visualization.pdf (+ .png)
  (无分类头权重需先训练: python vae.py --latent_dim 16 --no_cls)

运行:
  python analysis.py efficiency                            # 默认 ipc 1/10/50 全对比
  python analysis.py efficiency --ipc 1 10 50 100 200 500
  python analysis.py efficiency --downstream mobilenet --ipc 1 10 50
  python analysis.py latent

出图:
  results/training_cost_vs_ratio_{downstream}.png
  results/acc_vs_train_energy.png
  results/latent_dist_pca_edge.png / latent_dist_tsne_edge.png
  figures/latent_visualization.pdf (+ .png, 论文图)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import config
from backbones import BACKBONES, count_flops, count_params, load_syn_center

# ============================================================
# efficiency: 传输/能耗参考常数 (可调)
# ============================================================
BANDWIDTH_MBPS = 5.0          # 上行带宽 (Mbps), 4G 典型 5~50
ENERGY_PER_BIT = 1e-7         # 每 bit 传输能耗 (J), 无线 ~0.1 uJ/bit 量级
ENERGY_PER_FLOP = 1e-11       # 推理能效 (J/FLOP), 边缘加速器 ~10 pJ 量级
EPOCHS = 100                  # 下游训练轮数 (默认; 下游实验统一 50, 不同口径时显式传参)
TRAIN_FWD_MULT = 3.0          # 训练每样本 FLOPs ≈ 前向 FLOPs × 3 (前向+反向)
LATENT_SPATIAL = 8            # EdgeVAE 潜空间空间尺寸 (16ch×8×8)
SUBSET = 1000                 # 每类源子集规模 (与 exp_ipc 一致, 压缩率分母)
N_FULL = 50000                # CIFAR-10/100 训练集规模

# ============================================================
# latent: 可视化参数
# ============================================================
N_PER_CLASS = 100          # 每类编码多少张
N_COMP = 50                # t-SNE 前先 PCA 到 50 维
C = config.NUM_CLASSES
KMEANS_IPC = 100           # 压缩效果图使用的 IPC (与主线一致)

# ---- 论文图输出 (IEEE 双栏) ----
FIG_DIR = "figures"
LATENT_FIG_STEM = "latent_visualization"


# ============================================================
# 一、能效估算 (子命令 efficiency)
# ============================================================

def bytes_of_image(n):
    """原始图片字节数 (8bit, 32×32×3)"""
    return n * config.IMAGE_SIZE * config.IMAGE_SIZE * config.IMAGE_CHANNELS


def bytes_of_latent(n_points, latent_dim):
    """合成潜变量字节数 (int8 部署量化, 1 B/标量, latent_dim×8×8)"""
    return n_points * latent_dim * LATENT_SPATIAL * LATENT_SPATIAL


def fmt_bytes(b):
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024 or unit == "GB":
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} GB"


def train_cost(n_pts, fwd_flops, epochs=None):
    """下游分类器训练总 FLOPs: 样本数 × epochs × (单样本前向 FLOPs × 3)

    epochs 缺省用模块常量 EPOCHS (100), 也可传与 acc 实验一致的值 (如 backbone 跑 50)。
    """
    epochs = epochs or EPOCHS
    return n_pts * epochs * fwd_flops * TRAIN_FWD_MULT


def fmt_flops(x):
    """人类可读 FLOPs: 1.23e+10 -> 12.30 GFLOPs"""
    sign = "-" if x < 0 else ""
    x = abs(x)
    for unit, base in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= base:
            return f"{sign}{x / base:.2f} {unit}FLOPs"
    return f"{sign}{x:.0f} FLOPs"


class EncMu(nn.Module):
    """只统计 image -> encoder -> mu(x)

    复用 EdgeEncoder 的原始子模块 (conv1/down/mid/mu_conv), 刻意不含:
      - logvar 分支 (logvar_conv)
      - decoder / 分类辅助头
      - VAE 训练/反向
    """

    def __init__(self, enc):
        super().__init__()
        self.conv1 = enc.conv1
        self.down = enc.down
        self.mid = enc.mid
        self.mu_conv = enc.mu_conv

    def forward(self, x):
        h = self.conv1(x)
        for d in self.down:
            h = d(h)
        h = self.mid(h)
        return self.mu_conv(h)


def measure_encoder_flops(ae, device, image_size, image_channels):
    """返回 (F_enc, method_note); profiler 失败时抛出并报告原因"""
    enc_mu = EncMu(ae.encoder).to(device).eval()
    x = torch.randn(1, image_channels, image_size, image_size, device=device)
    with torch.no_grad():
        out = enc_mu(x)
    assert out.shape[1] == ae.latent_dim, (
        f"encoder mu 输出通道 {out.shape[1]} != latent_dim {ae.latent_dim}")
    note = ("thop.profile 对 image->encoder->mu(x) 前向逐算子汇总 MACs, "
            "再 ×2 转 FLOPs (排除 logvar 分支/decoder/分类头)")
    try:
        F_enc = count_flops(enc_mu, x)
    except Exception as e:  # 明确报告失败原因, 不猜测
        raise RuntimeError(
            f"FLOPs profiler (thop) 统计 VAE encoder 失败: {type(e).__name__}: {e}\n"
            f"统计方法: thop.profile(model, inputs=(x,)); 失败则本脚本不输出该值。") from e
    if F_enc <= 0:
        raise RuntimeError(f"profiler 返回非正 FLOPs ({F_enc}), 统计无效")
    return int(F_enc), note


def encode_real_subset(ae, device, ipc, chunk=256):
    """读 pt/real_subset_ipc{ipc}.pt 并编码为确定性潜变量 mu

    返回 (z [N,16,8,8], labels); 全池编码只做一次, 不计入 K-means runtime。
    """
    path = config.find_pt(f"real_subset_ipc{ipc}.pt")
    if path is None:
        raise FileNotFoundError(
            f"未找到 pt/real_subset_ipc{ipc}.pt; 请先运行 "
            f"python distill.py --ipc {ipc} --subset 1000")
    d = torch.load(path, map_location="cpu", weights_only=True)
    imgs, labels = d["images"], d["labels"]
    latents = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), chunk):
            latents.append(ae.encode_mu(imgs[i:i + chunk].to(device)).cpu())
    return torch.cat(latents, 0), labels


def plot_training_cost(ipcs, fwd_flops, downstream):
    """训练计算量/能耗 vs 压缩率 双面板图 (左=FLOPs, 右=能耗, 均对数轴)

    压缩率 = 源子集点数/类 ÷ IPC (与 exp_ipc 同口径); 只计下游分类器训练,
    参考线为 原始全量 (完全不蒸馏) 与 1:1 源子集 (编码后直训, 不压缩)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = config.NUM_CLASSES
    pairs = sorted({SUBSET / ipc: C * ipc for ipc in ipcs}.items())   # (压缩率, 合成点数) 升序
    xs = [p[0] for p in pairs]
    ys = [train_cost(p[1], fwd_flops) for p in pairs]
    refs = [(train_cost(N_FULL, fwd_flops), f"no distillation (full {N_FULL} pts)", "--", "gray"),
            (train_cost(C * SUBSET, fwd_flops),
             f"1:1 subset ({C * SUBSET} pts, no compression)", ":", "#2ca02c")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, mult, ylab in ((axes[0], 1.0, "Training FLOPs"),
                           (axes[1], ENERGY_PER_FLOP, "Training energy (J)")):
        ax.plot(xs, [y * mult for y in ys], "o-", color="#1f77b4", ms=7,
                label=f"{downstream} on distilled set")
        for val, lbl, ls, col in refs:
            ax.axhline(val * mult, ls=ls, color=col, lw=1.5, label=lbl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}:1" for x in xs])
        ax.set_xlabel("Compression ratio (subset/class / IPC, log)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training cost vs compression ratio (downstream={downstream}, "
                 f"subset={SUBSET}/class, {EPOCHS} epochs, {TRAIN_FWD_MULT:.0f}x fwd FLOPs/sample)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(config.RESULT_DIR, f"training_cost_vs_ratio_{downstream}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"训练成本曲线已保存: {out}")


def plot_acc_energy_pareto(ipcs, accs, backbones, latent_dim=None, epochs=50,
                           ref_acc=None, out_name="acc_vs_train_energy.png"):
    """Accuracy vs 训练能耗 Pareto 图 (每条 backbone 一条线, 每 IPC 一个点)

    横轴 = 下游分类器训练能耗 J (对数), 纵轴 = 测试 acc。理想工作点靠左上。
    能耗 = train_cost(合成点数, 单样本前向 FLOPs) × 推理能效, 与 backbone 实验同 epochs。

    参数:
      ipcs:     每类合成点数的列表
      accs:     {ipc: {bb: acc}}  每个 IPC 下各 backbone 的 acc
      backbones: 要画的下游网络 (如 ['resnet','vit'])
      ref_acc:  可选 {bb: acc}, 画 "1:1 源子集(不压缩)" 星标锚点
      epochs:   与 acc 对齐的训练轮数 (backbone 主线为 50)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latent_dim = latent_dim or config.AE_LATENT_DIM
    C = config.NUM_CLASSES
    colors = {"resnet": "#1f77b4", "vit": "#d62728", "vgg": "#2ca02c",
              "mobilenet": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    for bb in backbones:
        col = colors.get(bb, "#333333")
        fwd = count_flops(BACKBONES[bb](in_ch=latent_dim, num_classes=C),
                          torch.zeros(1, latent_dim, LATENT_SPATIAL, LATENT_SPATIAL))
        vals = [(ipc, accs[ipc][bb]) for ipc in ipcs if ipc in accs]
        xs = [train_cost(C * ipc, fwd, epochs=epochs) * ENERGY_PER_FLOP for ipc, _ in vals]
        ys = [a for _, a in vals]
        ax.plot(xs, ys, "o-", color=col, ms=7, lw=1.8, label=f"{bb}")
        for (ipc, _), x, y in zip(vals, xs, ys):
            ax.annotate(f"IPC={ipc}", (x, y), textcoords="offset points",
                        xytext=(6, -2), fontsize=8, color=col)
        # 1:1 参考锚点
        if ref_acc and bb in ref_acc:
            xr = train_cost(C * SUBSET, fwd, epochs=epochs) * ENERGY_PER_FLOP
            ax.plot(xr, ref_acc[bb], "*", ms=16, color=col,
                    label=f"{bb} 1:1 no-compression ({ref_acc[bb]:.1f}%)")
            ax.annotate("1:1", (xr, ref_acc[bb]), textcoords="offset points",
                        xytext=(6, -4), fontsize=9, color=col, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Training energy (J, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Accuracy vs downstream training energy\n"
                 f"(subset={SUBSET}/class, edge 16×8×8, {epochs} epochs, "
                 f"{TRAIN_FWD_MULT:.0f}× fwd FLOPs/sample)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(config.RESULT_DIR, out_name)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Accuracy-Energy Pareto 图已保存: {out}")
    return out


def run_efficiency(ipcs, downstream="resnet", latent_dim=None):
    latent_dim = latent_dim or config.AE_LATENT_DIM
    C = config.NUM_CLASSES

    print("=" * 80)
    print(f"边缘部署收益估算 (数据集: {config.DATASET}, 潜空间维度: {latent_dim})")
    print(f"带宽假设: {BANDWIDTH_MBPS} Mbps | 传输能耗: {ENERGY_PER_BIT * 1e6:.2f} uJ/bit "
          f"| 推理能效: {ENERGY_PER_FLOP * 1e12:.1f} pJ/FLOP | 训练轮数: {EPOCHS}")
    print("=" * 80)

    # ============================================================
    # 1. 数据压缩 & 传输
    # ============================================================
    print("\n[1] 数据压缩与传输")
    print(f"{'数据形式':<16}{'点数':>8}{'数据量':>14}{'压缩比':>10}{'传输时延':>12}{'传输能耗':>12}")
    print("-" * 80)
    full_bytes = bytes_of_image(N_FULL)
    full_delay = full_bytes * 8 / (BANDWIDTH_MBPS * 1e6)
    full_energy = full_bytes * 8 * ENERGY_PER_BIT
    print(f"{'原始图片(全量)':<16}{N_FULL:>8}{fmt_bytes(full_bytes):>14}{'1.0x':>10}"
          f"{full_delay:>10.1f}s{full_energy:>10.2f}J")

    rows = []
    for ipc in ipcs:
        n_pts = C * ipc
        b = bytes_of_latent(n_pts, latent_dim)
        ratio = full_bytes / b
        delay = b * 8 / (BANDWIDTH_MBPS * 1e6)
        energy = b * 8 * ENERGY_PER_BIT
        rows.append((ipc, n_pts, b, ratio, delay, energy))
        print(f"{f'合成 ipc={ipc}':<16}{n_pts:>8}{fmt_bytes(b):>14}{f'{ratio:.1f}x':>10}"
              f"{delay:>10.2f}s{energy:>10.2f}J")

    # ============================================================
    # 2. 训练能耗 (下游网络 FLOPs × 样本数 × epoch)
    # ============================================================
    # 下游网络单样本前向 FLOPs
    model = BACKBONES[downstream](in_ch=latent_dim, num_classes=C)
    fwd_flops = count_flops(model, torch.zeros(1, latent_dim, LATENT_SPATIAL, LATENT_SPATIAL))

    print(f"\n[2] 训练能耗 (下游={downstream}, 前向 {fwd_flops:,} FLOPs, {EPOCHS} epochs)")
    print(f"{'数据':<16}{'样本数':>10}{'训练总FLOPs':>16}{'训练能耗':>14}")
    print("-" * 80)
    train_flop_per_samp = fwd_flops * TRAIN_FWD_MULT
    full_tflop = N_FULL * EPOCHS * train_flop_per_samp
    print(f"{'原始全量':<16}{N_FULL:>10}{full_tflop:>14.2e}{full_tflop * ENERGY_PER_FLOP:>12.2f}J")
    for ipc, n_pts, b, ratio, delay, energy in rows:
        tflop = n_pts * EPOCHS * train_flop_per_samp
        print(f"{f'合成 ipc={ipc}':<16}{n_pts:>10}{tflop:>14.2e}{tflop * ENERGY_PER_FLOP:>12.2f}J")

    # 训练计算量/能耗 vs 压缩率 双面板图 (只计下游训练)
    plot_training_cost(ipcs, fwd_flops, downstream)

    # ============================================================
    # 3. 推理能耗 (各 backbone 单次前向)
    # ============================================================
    print(f"\n[3] 推理能耗 (单次前向, 潜空间 {latent_dim}×{LATENT_SPATIAL}×{LATENT_SPATIAL})")
    print(f"{'backbone':<12}{'params':>12}{'FLOPs':>14}{'推理能耗':>12}")
    print("-" * 80)
    for name, cls in BACKBONES.items():
        m = cls(in_ch=latent_dim, num_classes=C)
        p = count_params(m)
        f = count_flops(m, torch.zeros(1, latent_dim, LATENT_SPATIAL, LATENT_SPATIAL))
        e = f * ENERGY_PER_FLOP
        print(f"{name:<12}{p:>12,}{f:>14,}{e * 1e6:>10.2f}uJ")
    print("=" * 80)
    print("注: 训练能耗 ∝ 样本数, 蒸馏后样本数下降 → 训练能耗/时延线性下降, 这是边缘场景最大收益。")


# ============================================================
# 二、潜空间分布可视化 (子命令 latent)
# ============================================================

def encode_subset(ae, loader, device, n_per_class=N_PER_CLASS):
    """编码训练集子集 (每类 n_per_class 张), 返回 (z_flat [N,D], labels [N])"""
    latents, labels = [], []
    counts = [0] * C
    with torch.no_grad():
        for x, y in loader:
            z = ae.encode_mu(x.to(device))          # [B,16,8,8]
            for i in range(y.size(0)):
                c = y[i].item()
                if counts[c] < n_per_class:
                    latents.append(z[i].cpu().reshape(-1))
                    labels.append(c)
                    counts[c] += 1
            if all(n >= n_per_class for n in counts):
                break
    return torch.stack(latents).numpy(), np.array(labels)


def plot_latent_pair(emb_cls, emb_nc, y, path, method, titles=None, suptitle=None,
                     show_titles=True):
    """左右并排散点图: 左=emb_cls | 右=emb_nc

    show_titles=False 时不画子图标题与总标题 (论文出图口径), 只保留共享图例。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if titles is None:
        titles = ["EdgeVAE with cls head",
                  "EdgeVAE without cls head (recon only)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, title in zip(axes, [emb_cls, emb_nc], titles):
        for c in range(C):
            mask = y == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                       label=config.CLASS_NAMES[c] if emb is emb_cls else None)
        if show_titles:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    # 只保留一份图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
    if show_titles:
        fig.suptitle(suptitle or f"Latent z distribution ({method}, {N_PER_CLASS}/class, CIFAR-10)",
                     fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0 if not show_titles else 0.95])
    fig.savefig(path, dpi=150)
    print(f"已保存: {path}")


def plot_compress_pair(emb_real, y_real, emb_syn, y_syn, n_real=None, n_syn=None,
                       ipc=KMEANS_IPC, path=None):
    """论文图 Latent-Space Visualization: 左=源池真实潜变量 | 右=K-means 中心代表

    真实点与 K-means 质心拼接后做**单次联合 t-SNE** (PCA-50 预处理, 随机种子固定),
    两面板共享同一坐标系与坐标范围, 因此同一类在左右落在相同位置 —— 该联合嵌入的
    说明写在 caption 里, 图内不加任何注释文字。
    两个面板共用同一套类颜色 (tab10, 即 matplotlib 默认循环色) 用于左右对照;
    不画图例, 类别身份由 caption 交代; 无图内大标题。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    POINT_SIZE = 6.0          # 两面板统一 marker 面积, 差异只靠点数与不透明度体现
    # 双面板: 4:3 指每个子图(单个坐标区), 不是整张合成图。
    # figsize 按坐标区实测比例选定 (本尺寸下每面板 ≈ 1.33)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95), sharex=True, sharey=True)
    panels = [
        (axes[0], emb_real, y_real, POINT_SIZE, 0.45,
         "Real latents" + (f" ($N$={n_real:,})" if n_real else "")),
        (axes[1], emb_syn, y_syn, POINT_SIZE, 0.90,
         "K-means centroids" + (f" ($M$={n_syn:,}, IPC$={ipc}$)" if n_syn else "")),
    ]
    for ax, emb, lab, size, alpha, title in panels:
        for c in range(C):
            mask = lab == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=size, alpha=alpha,
                       color=cmap(c % 10), linewidths=0)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # 联合嵌入: 两面板取同一坐标范围, 位置对应关系才在视觉上成立
    xy = np.vstack([emb_real, emb_syn])
    x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
    y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
    xpad = 0.02 * (x1 - x0)
    ypad = 0.02 * (y1 - y0)
    for ax in axes:
        ax.set_xlim(x0 - xpad, x1 + xpad)
        ax.set_ylim(y0 - ypad, y1 + ypad)
    fig.tight_layout()

    path = path or os.path.join(FIG_DIR, f"{LATENT_FIG_STEM}.pdf")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    outs = [path]
    if path.endswith(".pdf"):        # 同时存一份 png 便于预览
        png = path[:-4] + ".png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        outs.append(png)
    plt.close()
    print("论文图已保存: " + " / ".join(outs))


def encode_images(ae, imgs, device, batch=256):
    """按批编码图片张量, 返回 [N, D] 展平潜变量"""
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), batch):
            outs.append(ae.encode_mu(imgs[i:i + batch].to(device)).cpu())
    return torch.cat(outs).reshape(imgs.size(0), -1).numpy()


def run_compress(ipc=KMEANS_IPC):
    """真实潜变量 (带分类头 VAE 编码) vs K-means 合成中心, 展示压缩效果

    两侧点数 = 源子集真实图 vs IPC, 比值即压缩比。
    """
    from vae import load_edge_vae

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    ae = load_edge_vae(device)

    real_path = config.find_pt(f"real_subset_ipc{ipc}.pt")
    syn_path = load_syn_center("mean", ipc)
    if real_path is None or syn_path is None:
        print(f"[跳过] 需要 real_subset_ipc{ipc}.pt 与 syn_latents_ipc{ipc}_mean_edge.pt, "
              f"请先跑: python distill.py --ipc {ipc} --subset 1000")
        return

    print("编码真实子集...")
    real = torch.load(real_path, map_location="cpu", weights_only=True)
    z_real = encode_images(ae, real["images"], device)
    y_real = real["labels"].numpy()
    print(f"真实潜变量: {z_real.shape} ({z_real.shape[0] // C}/类)")

    syn = torch.load(syn_path, map_location="cpu", weights_only=True)["syn_latents"]
    z_syn = syn.reshape(syn.size(0) * syn.size(1), -1).numpy()
    y_syn = np.repeat(np.arange(C), syn.size(1))
    print(f"K-means 合成中心: {z_syn.shape} ({syn.size(1)}/类)")

    # ---- t-SNE 2D: 真实点与质心拼接后做单次联合嵌入, 两面板共享坐标系 ----
    print("t-SNE 降维 (real ∪ syn 联合嵌入, 共享坐标系)...")
    n_real, n_syn = z_real.shape[0], z_syn.shape[0]
    emb = _tsne_2d(np.concatenate([z_real, z_syn], axis=0))
    emb_real, emb_syn = emb[:n_real], emb[n_real:]
    plot_compress_pair(emb_real, y_real, emb_syn, y_syn,
                       n_real=n_real, n_syn=n_syn, ipc=ipc)


def linear_acc(z, y, seed=0):
    """潜变量上的线性分类准确率 (80/20 split), 佐证判别性"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_tr = int(len(y) * 0.8)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(z[idx[:n_tr]], y[idx[:n_tr]])
    return clf.score(z[idx[n_tr:]], y[idx[n_tr:]]) * 100


def _tsne_2d(z):
    pca50 = PCA(n_components=N_COMP, random_state=0).fit_transform(z)
    return TSNE(n_components=2, random_state=0, perplexity=30, init="pca",
                n_jobs=4).fit_transform(pca50)


def run_latent():
    from config import get_cifar10_loaders
    from vae import load_edge_vae

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    ae1 = load_edge_vae(device)                              # 带分类头: autoencoder_edge_s8_c16.pt
    ae2 = load_edge_vae(device, ckpt_name="autoencoder_edge_s8_c16_noclspart.pt")  # 无分类头
    titles = ["EdgeVAE with cls head (16×8×8)",
              "EdgeVAE without cls head (recon only)"]
    suptitle = "Latent z distribution: EdgeVAE 16×8×8 (CIFAR-10)"

    _, _, test_loader = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=256,
        num_workers=config.NUM_WORKERS, augment=False)

    print("编码子集...")
    z1, y = encode_subset(ae1, test_loader, device)
    z2, _ = encode_subset(ae2, test_loader, device)
    print(f"编码完成: {z1.shape}, {z2.shape}")

    # 线性分类准确率 (佐证, 与散点图对应)
    acc1 = linear_acc(z1, y)
    acc2 = linear_acc(z2, y)
    print(f"线性分类准确率 → {titles[0]}: {acc1:.1f}% | {titles[1]}: {acc2:.1f}%")

    # ---- PCA 2D ----
    pca1 = PCA(n_components=2).fit_transform(z1)
    pca2 = PCA(n_components=2).fit_transform(z2)
    plot_latent_pair(pca1, pca2, y, f"{config.RESULT_DIR}/latent_dist_pca_edge.png",
                     "PCA", titles, suptitle)

    # ---- t-SNE 2D (先 PCA 到 50 维, 论文出图: 不带标题) ----
    tsne1 = _tsne_2d(z1)
    tsne2 = _tsne_2d(z2)
    plot_latent_pair(tsne1, tsne2, y, f"{config.RESULT_DIR}/latent_dist_tsne_edge.png",
                     "t-SNE", titles, suptitle, show_titles=False)

    # ---- 真实潜变量 vs K-means 合成中心 (压缩效果) ----
    run_compress()


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析线: 边缘能效估算 + 潜空间可视化")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eff = sub.add_parser("efficiency", help="蒸馏前后能耗/时延/压缩比理论估算")
    p_eff.add_argument("--ipc", type=int, nargs="+", default=[1, 10, 50])
    p_eff.add_argument("--downstream", type=str, default="resnet",
                       choices=["vgg", "resnet", "mobilenet", "vit"])
    p_eff.add_argument("--latent_dim", type=int, default=None)

    sub.add_parser("latent", help="潜空间分布 PCA / t-SNE 对比 (带 vs 无分类头)")

    args = parser.parse_args()
    if args.cmd == "efficiency":
        run_efficiency(ipcs=args.ipc, downstream=args.downstream,
                       latent_dim=args.latent_dim)
    else:
        run_latent()
