"""
IPC 压缩比曲线: 固定每类源子集 (默认 1000 张), 不同 IPC (压缩到多少合成点/类)
在多个 backbone 上对比 精度 + 压缩率 + 数据量/传输耗时 (默认 resnet + vit 同图对比)。

背景:
  蒸馏源数据每类固定 subset=1000 (seed=42 固定抽样, 任何 IPC 用同一批真实图),
  IPC 决定每类最终保留几个 mean 中心合成点 → 压缩率(张) = subset / IPC。
  下游默认用 resnet 与 vit 训练 50 epoch 余弦衰减, 与 exp_method_compare 同条件。

默认 IPC 集: 1, 10, 50, 100(=10:1), 200(=5:1), 500(=2:1), 另加 1:1 full-latent 基线
  (不压缩: 直接用该 1000/类子集的编码潜变量训练, 作为论文图中 M=10000 的数据点)
另有"完全不换表示"底线: 同一批 1000/类 原图 (3x32x32) 直训 (~73%, 默认开,
  --no_pixel 可跳过) —— 只进终端汇总表与 loss 曲线参考, 不进论文主图。

论文主图 (Accuracy-Compression Trade-off, IEEE 双栏单栏):
  figures/accuracy_compression_tradeoff.pdf (+ .png)
  横轴为离散压缩比 N/M (1:1, 2:1, 5:1, 10:1, 20:1, 100:1, 1000:1 七个已评估预算),
  四 backbone 带 mean±std error bar; 无图内标题、无 MB 标注、无 raw-image 水平线,
  只标默认 operating point IPC=100 (10:1)。

运行 (先确保各 IPC 蒸馏产物存在, 或加 --auto 自动补跑):
  python exp_ipc.py                                    # 默认 resnet + vit + IPC 集
  python exp_ipc.py --backbone resnet vit vgg --ipc "1 100 200 500" --seeds 3
  python exp_ipc.py --backbone vit --ipc 500 --auto    # 只比 2:1, 缺产物自动蒸馏
  python exp_ipc.py --no_pixel                         # 跳过原图直训底线 (省 ~5 分钟)
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import torch

import config
from backbones import BACKBONES, count_flops, load_syn_center, train_eval_backbone
from config import get_cifar10_loaders
from utils import encode_test_latents, flat_test_cache
from vae import load_edge_vae

SUBSET = 500 if config.DATASET == "cifar100" else 1000   # 每类源真实图数 (压缩前池; cifar100 每类仅 500)
C = config.NUM_CLASSES          # 类别数 (cifar10=10 / cifar100=100)
BANDWIDTH_MBPS = 10    # 云→边缘链路假设


def ensure_distill(ipc, center, auto):
    """找产物, 缺失且 auto 时自动跑 distill (默认 subset/arch 与新主线一致)"""
    p = load_syn_center(center, ipc)
    if p is not None:
        return p
    if not auto:
        print(f"  缺少产物 syn_latents_ipc{ipc}_{center}_edge.pt, 请先跑: "
              f"python distill.py --ipc {ipc} --subset {SUBSET} --center {center}")
        return None
    print(f"  产物缺失, 自动蒸馏 ipc={ipc} (subset={SUBSET}, {center}, edge)...")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "distill.py"),
                    "--ipc", str(ipc), "--subset", str(SUBSET), "--center", center],
                   check=True)
    return load_syn_center(center, ipc)


def find_real_subset(ipcs):
    """取任意已有的 real_subset_ipc{ipc}.pt (seed=42 固定, 各 ipc 内容一致, 都是 subset 1000/类)"""
    for ipc in ipcs:
        p = config.find_pt(f"real_subset_ipc{ipc}.pt")
        if p is not None:
            return p
    for f in sorted(os.listdir(config.PT_DIR)):
        if f.startswith("real_subset_ipc") and f.endswith(".pt"):
            return os.path.join(config.PT_DIR, f)
    return None


def encode_imgs(imgs, ae, device, batch=256):
    """批量编码真实图 → 潜变量 [N, C, H, W] (保留空间形状, 与蒸馏产物同构)"""
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), batch):
            outs.append(ae.encode_mu(imgs[i:i + batch].to(device)).cpu())
    return torch.cat(outs, dim=0)


def mean_accs(backbone_name, z_train, labels, test_latents, ae, device, seeds, epochs,
              return_curves=False):
    """多 seed 训练 backbone, 返回 (均值, seeds列表[, (train_mean, test_mean)])"""
    accs = []
    curves = []
    for s in range(seeds):
        out = train_eval_backbone(
            BACKBONES[backbone_name], z_train, labels, None, ae, device, seed=s,
            epochs=epochs, return_curve=return_curves, test_latents=test_latents,
            lr_schedule="cosine")
        if return_curves:
            acc, tr_c, te_c = out
            curves.append((tr_c, te_c))
        else:
            acc = out
        accs.append(acc)
        print(f"    seed{s}: {acc:.2f}%")
    if not return_curves:
        return float(np.mean(accs)), accs
    tr = np.mean([np.asarray(c[0], float) for c in curves], 0)
    te = np.mean([np.asarray(c[1], float) for c in curves], 0)
    return float(np.mean(accs)), accs, (tr.tolist(), te.tolist())


def save_json(rows, ref, ref_seeds, pix_acc, backbones, seeds, epochs, center,
              latent_dim, spatial):
    """把 IPC 扫描结果落盘为 JSON (acc±std / 数据量 / 传输 / 训练能耗 / 损失曲线)

    能耗口径与 analysis.py 一致: 点数 × epochs × 单样本前向FLOPs × 3 × 1e-11 J
    (只计下游分类器训练, 不含 VAE 编码与传输)。
    """
    import json
    from datetime import datetime
    from analysis import train_cost, ENERGY_PER_FLOP, TRAIN_FWD_MULT, N_FULL

    fwd = {}
    for bb in backbones:
        m = BACKBONES[bb](in_ch=latent_dim, num_classes=C)
        fwd[bb] = count_flops(m, torch.zeros(1, latent_dim, spatial, spatial))

    def energy(n_pts, bb):
        return train_cost(n_pts, fwd[bb], epochs=epochs) * ENERGY_PER_FLOP

    out = {
        "meta": {
            "dataset": config.DATASET, "arch": "edge", "latent_dim": latent_dim,
            "spatial": spatial, "subset_per_class": SUBSET, "seeds": seeds,
            "epochs": epochs, "center": center, "backbones": list(backbones),
            "energy_model": {
                "J_per_flop": ENERGY_PER_FLOP, "train_fwd_mult": TRAIN_FWD_MULT,
                "epochs": epochs, "note": "downstream classifier only"},
            "fwd_flops_per_sample": fwd,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rows": [{"ipc": r["ipc"], "ratio": r["ratio"], "points": C * r["ipc"],
                  "acc": r["acc"], "std": r["std"], "seeds_acc": r["accs"],
                  "scalars": r["scalars"],
                  "mb_int8": r["mb_int8"], "t_int8_s": r["t_int8_s"],
                  "train_flops": {bb: train_cost(C * r["ipc"], fwd[bb], epochs=epochs)
                                  for bb in backbones},
                  "train_energy_J": {bb: energy(C * r["ipc"], bb) for bb in backbones},
                  "curves": {bb: {"train": r["train_c"][bb], "test": r["test_c"][bb]}
                             for bb in backbones}} for r in rows],
        "ref_1to1_no_compression": {
            "points": C * SUBSET, "acc": ref, "seeds_acc": ref_seeds,
            "std": {bb: float(np.std(ref_seeds[bb])) for bb in backbones if bb in ref_seeds},
            "train_energy_J": {bb: energy(C * SUBSET, bb) for bb in backbones}},
        "ref_raw_pixel_direct": {"acc": pix_acc, "points": C * SUBSET},
        "ref_full_dataset": {"points": N_FULL,
                             "train_energy_J": {bb: energy(N_FULL, bb) for bb in backbones}},
    }
    path = os.path.join(config.RESULT_DIR,
                        f"ipc_sweep_{'_'.join(backbones)}_{config.DATASET}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"扫描数据已保存: {path}")
    return path


def main(ipcs, backbones=("resnet",), seeds=3, epochs=50, center="mean", auto=True,
         include_pixel=True):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | backbones: {list(backbones)} | subset/类: {SUBSET} | "
          f"epochs: {epochs} | seeds: {seeds}")
    ae = load_edge_vae(device)
    print(f"EdgeVAE 已加载 (冻结, latent_dim={ae.latent_dim}, spatial={ae.spatial_size})")

    _, _, test_loader = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS)
    test_latents = encode_test_latents(test_loader, ae, device)   # 缓存一次
    print(f"测试潜变量缓存: {test_latents[0].shape}")

    # ---- 1:1 参考线: 每类 1000 张全部编码 (不压缩), 逐 backbone ----
    rs_path = find_real_subset(ipcs)
    if rs_path is None:
        raise FileNotFoundError("没找到 real_subset_ipc*.pt, 请先跑一次 distill.py")
    rs = torch.load(rs_path, map_location="cpu", weights_only=True)
    imgs, rlabels = rs["images"], rs["labels"]
    print(f"\n[1:1 参考] 真实子集 {imgs.size(0)} 张 (每类 {SUBSET}, 不压缩, 直接编码训练)...")
    z_sub = encode_imgs(imgs, ae, device)
    ref, ref_curves = {}, {}
    ref_seeds = {}
    for bb in backbones:
        acc, seeds_acc, curves = mean_accs(bb, z_sub, rlabels, test_latents, ae, device,
                                           seeds, epochs, return_curves=True)
        ref[bb], ref_curves[bb] = acc, curves
        ref_seeds[bb] = seeds_acc
        print(f"[1:1 参考] {bb}: {acc:.2f}% ± {np.std(seeds_acc):.2f}  "
              f"(seeds={[f'{a:.2f}' for a in seeds_acc]})")

    # ---- 完全不换表示底线: 同批原图直训 (不经 VAE), 逐 backbone ----
    pix_acc, pix_curves = {}, {}
    if include_pixel:
        pix_test = flat_test_cache(test_loader)
        print(f"\n[raw 参考] 原图直训 {imgs.size(0)} 张 (3x32x32, 不经 VAE/不压缩)...")
        for bb in backbones:
            acc, seeds_acc, curves = mean_accs(bb, imgs, rlabels, pix_test, ae, device,
                                               seeds, epochs, return_curves=True)
            pix_acc[bb], pix_curves[bb] = acc, curves
            print(f"[raw 参考] {bb}: {acc:.2f}% ± {np.std(seeds_acc):.2f}  "
                  f"(seeds={[f'{a:.2f}' for a in seeds_acc]})")

    # ---- 逐 IPC (每个 IPC 下逐 backbone) ----
    rows = []
    for ipc in sorted(ipcs):
        p = ensure_distill(ipc, center, auto)
        if p is None:
            continue
        syn = torch.load(p, map_location="cpu", weights_only=True)["syn_latents"]  # [10, ipc, 16, 8, 8]
        z_syn = syn.reshape(C * ipc, -1).reshape(C * ipc, *syn.shape[2:])
        labels = torch.arange(C).repeat_interleave(ipc)
        ratio = SUBSET / ipc
        print(f"\n[IPC={ipc}] 压缩率 {ratio:.0f}:1 ({SUBSET}→{ipc} 张/类) 合成点 {C*ipc} 个...")

        acc, accs, std, train_c, test_c = {}, {}, {}, {}, {}
        for bb in backbones:
            a, seeds_acc, curves = mean_accs(bb, z_syn, labels, test_latents, ae, device,
                                             seeds, epochs, return_curves=True)
            acc[bb], accs[bb], std[bb] = a, seeds_acc, float(np.std(seeds_acc))
            train_c[bb], test_c[bb] = curves

        # 传输数据量 (下游训练需下发的"数据集"): C*ipc 个潜样本 × 1024 维
        # 统一部署量化口径 int8 (1 B/标量), 与 analysis.bytes_of_latent / exp_energy_methods 一致
        scalars = C * ipc * ae.latent_dim * ae.spatial_size * ae.spatial_size
        mb_int8 = scalars * 1 / 1e6
        t_int8_s = scalars * 8 / (BANDWIDTH_MBPS * 1e6)          # int8 字节 → 秒 @带宽
        rows.append({"ipc": ipc, "ratio": ratio, "acc": acc, "accs": accs, "std": std,
                     "scalars": scalars, "mb_int8": mb_int8,
                     "t_int8_s": t_int8_s, "train_c": train_c, "test_c": test_c})
        for bb in backbones:
            print(f"  {bb}: {acc[bb]:.2f}% ± {std[bb]:.2f}  (Δvs 1:1 {acc[bb] - ref[bb]:+.2f}) | "
                  f"数据 {mb_int8:.2f}MB(int8) | 传输 {t_int8_s:.2f}s@10Mbps")

    # ---- 汇总表 ----
    width = 60 + 12 * len(backbones)
    acc_hdr = "".join(f"{f'{bb} acc':>12}" for bb in backbones)
    print("\n" + "=" * width)
    print(f"{'IPC':>5}{'压缩率':>8}{'合成点':>8}{'数据MB(int8)':>14}"
          f"{'传输s@10M':>12}{acc_hdr}")
    for r in rows:
        acc_cells = "".join(f"{r['acc'][bb]:>11.2f}%" for bb in backbones)
        print(f"{r['ipc']:>5}{f'{r['ratio']:.0f}:1':>8}{C*r['ipc']:>8}{r['mb_int8']:>14.2f}"
              f"{r['t_int8_s']:>12.2f}{acc_cells}")
    ref_cells = "".join(f"{ref[bb]:>11.2f}%" for bb in backbones)
    print(f"{'--':>5}{'1:1':>8}{C*SUBSET:>8}{C*SUBSET*1024/1e6:>14.2f}"
          f"{C*SUBSET*1024*8/1e6/10:>12.2f}{ref_cells}   <- 参考(不压缩)")
    print("=" * width)
    print(f"参考: 原图 1000/类 (uint8) = {C*SUBSET*3072/1e6:.2f} MB, 传输 "
          f"{C*SUBSET*3072*8/1e6/10:.1f}s@10Mbps")
    if include_pixel:
        pix_cells = ", ".join(f"{bb} {pix_acc[bb]:.2f}%" for bb in backbones)
        print(f"参考(完全不换表示): 同批原图直训 pixel = {pix_cells}  (仅进汇总表与 loss 曲线)")

    # ---- 曲线图: Accuracy-Compression Trade-off (4 backbone) + 逐 backbone 双面板 loss ----
    # 1:1 full-latent 基线由实测 ref/ref_seeds 注入 FULL_* 占位变量, 不臆造数值
    set_full_latent_baseline(ref, ref_seeds)
    plot_ipc(rows, backbones)
    for bb in backbones:
        plot_loss_curves_ipc(rows, bb,
                             pix_curves=pix_curves.get(bb) if include_pixel else None,
                             pix_acc=pix_acc.get(bb) if include_pixel else None)

    # ---- 落盘结构化数据 (供论文表格/复算) ----
    save_json(rows, ref, ref_seeds, pix_acc if include_pixel else {}, backbones, seeds,
              epochs, center, ae.latent_dim, ae.spatial_size)

    # ---- 附加: Accuracy vs 下游训练能耗 Pareto (数值与 acc 完全同步) ----
    try:
        from analysis import plot_acc_energy_pareto
        ipc_list = [r["ipc"] for r in rows]
        accs = {r["ipc"]: r["acc"] for r in rows}
        plot_acc_energy_pareto(ipc_list, accs, list(backbones), epochs=epochs,
                               ref_acc=ref, out_name="acc_vs_train_energy.png")
    except Exception as e:
        print(f"[warn] 跳过 Accuracy-Energy Pareto 图: {e}")

    return rows


def plot_loss_curves_ipc(rows, backbone, pix_curves=None, pix_acc=None):
    """双面板: 左=Train CE loss, 右=Test CE loss, 每 IPC 一条 (多 seed 平均), 单个 backbone 一张"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, r in enumerate(rows):
        col = cmap(k / max(len(rows) - 1, 1))
        lbl = f"IPC={r['ipc']} ({r['acc'][backbone]:.1f}%)"
        axes[0].plot(r["train_c"][backbone], lw=1.8, color=col, marker="o", markevery=5, markersize=6, label=lbl)
        axes[1].plot(r["test_c"][backbone], lw=1.8, color=col, marker="o", markevery=5, markersize=6, label=lbl)
    if pix_curves is not None:
        pix_tr, pix_te = pix_curves
        lbl = f"raw-image direct (no VAE) ({pix_acc:.1f}%)" if pix_acc else "raw-image direct (no VAE)"
        axes[0].plot(pix_tr, lw=1.8, ls=":", color="#d62728",
                     label=lbl)
        axes[1].plot(pix_te, lw=1.8, ls=":", color="#d62728",
                     label=lbl)
    axes[0].set_title("Train CE loss")
    axes[1].set_title("Test CE loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CE loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    n_seeds = len(rows[0]["accs"][backbone]) if rows else "?"
    fig.suptitle(f"{backbone} loss curves vs IPC (subset={SUBSET}/class, edge 16x8x8, "
                 f"mean over {n_seeds} seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(config.RESULT_DIR, "loss_curves_ipc_{}.png".format(backbone))
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"loss 曲线已保存: {out}")


# ============================================================
# 论文主图: Accuracy-Compression Trade-off (IEEE 双栏单栏)
# ============================================================
FIG_DIR = "figures"
FIG_STEM = "accuracy_compression_tradeoff"
DEFAULT_IPC = 100                       # 默认 operating point: IPC=100 ↔ 10:1
DEFAULT_RATIO = SUBSET / DEFAULT_IPC
# 论文图不指定颜色主题, 用 matplotlib 默认色循环; 靠 marker + 线型双重区分,
# 保证灰度打印也能分辨四条曲线。
BB_STYLE = {
    "resnet":    ("ResNet",    "o", "-"),
    "vit":       ("ViT",       "s", "--"),
    "vgg":       ("VGG",       "^", "-."),
    "mobilenet": ("MobileNet", "D", ":"),
}

# ---- 1:1 full-latent 基线占位变量 (M = 10000, 不压缩) ----
# 数值取自最终统一协议的实测结果: main() 里的 "[1:1 参考]" 段
# (每类 1000 张真实样本编码后的潜变量直接训练, 3 seeds, 50 epoch, 余弦衰减),
# 由 set_full_latent_baseline() 注入; 缺失时保持 None → 图中不画该点, 绝不臆造数值。
# 注意: Table I should be synchronized with this final protocol.
FULL_RESNET = None
FULL_VIT = None
FULL_VGG = None
FULL_MOBILENET = None
FULL_STD = {}


def set_full_latent_baseline(ref, ref_seeds):
    """把 1:1 full-latent 参考段的实测均值/标准差注入 FULL_* 占位变量

    ref: {backbone: acc}; ref_seeds: {backbone: [每 seed 的 acc]} (来自 main() 的 1:1 段)
    """
    global FULL_RESNET, FULL_VIT, FULL_VGG, FULL_MOBILENET, FULL_STD
    FULL_RESNET = ref.get("resnet")
    FULL_VIT = ref.get("vit")
    FULL_VGG = ref.get("vgg")
    FULL_MOBILENET = ref.get("mobilenet")
    FULL_STD = {bb: float(np.std(v)) for bb, v in ref_seeds.items()}


def plot_ipc(rows, backbones):
    """Accuracy-Compression Trade-off: 各 backbone 在离散 IPC 预算下的 acc vs 压缩比 N/M

    论文图口径 (IEEE 双栏单栏, 信息交给 caption):
      - 无图内标题; 无 raw-image direct 水平线; 无逐点 IPC/MB 标注 (存储量留给 Table I);
      - 唯一注释是默认 operating point IPC=100 (10:1), 落在曲线下方空白区;
      - 每点带 mean ± std error bar (3 seeds);
      - 折线只是离散预算点之间的视觉引导 (lines are visual guides between evaluated
        discrete budgets), 不代表测过连续压缩比。
    x 轴映射: 1:1 = full latent (M=10000) / 2:1 = IPC500 / 5:1 = IPC200 / 10:1 = IPC100
              / 20:1 = IPC50 / 100:1 = IPC10 / 1000:1 = IPC1。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_mean = {"resnet": FULL_RESNET, "vit": FULL_VIT,
                 "vgg": FULL_VGG, "mobilenet": FULL_MOBILENET}
    rows = sorted(rows, key=lambda r: r["ratio"])          # 压缩比升序: 1:1 → 1000:1
    xticks = ([1.0] if any(full_mean.get(bb) is not None for bb in backbones) else []) \
        + [r["ratio"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y_lo, y_hi = np.inf, -np.inf
    for bb in backbones:
        name, mk, ls = BB_STYLE.get(bb, (bb, "o", "-"))
        x, mu, sd = [], [], []
        if full_mean.get(bb) is not None:                  # 1:1 full-latent 基线点
            x.append(1.0)
            mu.append(full_mean[bb])
            sd.append(FULL_STD.get(bb))                    # 缺 std → None, 只画均值点
        for r in rows:
            x.append(r["ratio"])
            mu.append(r["acc"][bb])
            sd.append(r["std"][bb])
        ax.errorbar(x, mu, yerr=None if any(s is None for s in sd) else sd,
                    marker=mk, ls=ls, lw=1.2, ms=4.5, capsize=2,
                    elinewidth=0.8, capthick=0.8, label=name)
        sd_num = [0.0 if s is None else s for s in sd]
        y_lo = min(y_lo, float(np.min(np.asarray(mu) - np.asarray(sd_num))))
        y_hi = max(y_hi, float(np.max(np.asarray(mu) + np.asarray(sd_num))))

    # ---- 纵轴: 按数据自动定, 收敛到建议区间 (25~30 / 91~92), 但不裁掉任何数据 ----
    lo = float(np.clip(np.floor(y_lo - 2.0), 25, 30))
    hi = float(np.clip(np.ceil(y_hi + 1.5), 91, 92))
    ax.set_ylim(min(lo, y_lo - 1.0), max(hi, y_hi + 1.0))

    # ---- 默认 operating point: IPC=100 ↔ 10:1 (注释落在曲线下方空白区, 不遮数据) ----
    ax.axvline(DEFAULT_RATIO, ls="--", lw=0.8, color="0.45", alpha=0.7, zorder=1)
    ax.annotate(f"IPC={DEFAULT_IPC} ({DEFAULT_RATIO:.0f}:1)",
                xy=(DEFAULT_RATIO, ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9, color="0.35")

    # ---- 横轴: 压缩比 N/M, 刻度只落在实际评估过的离散预算上 ----
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.0f}:1" for t in xticks])
    ax.set_xlim(min(xticks) * 0.72, max(xticks) * 1.35)
    ax.minorticks_off()
    ax.set_xlabel(r"Compression Ratio $N/M$", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # legend 竖排放在曲线下方空白区 (左下角)
    ax.legend(ncol=1, frameon=False, fontsize=9, loc="lower left",
              handlelength=1.6, handletextpad=0.5, labelspacing=0.35)

    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, f"{FIG_STEM}.pdf")
    png = os.path.join(FIG_DIR, f"{FIG_STEM}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"论文图已保存: {pdf} / {png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPC 压缩比 vs 精度 (固定子集, 多 backbone 同图对比)")
    parser.add_argument("--ipc", type=str, default="1 10 50 100 200 500",
                        help="每类合成点数的空格分隔列表 (默认含 500=2:1 / 200=5:1 / 100=10:1)")
    parser.add_argument("--backbone", type=str, nargs="+", default=["resnet", "vit"],
                        choices=list(BACKBONES),
                        help="下游网络, 可传多个画在同一张曲线上 (默认 resnet vit)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--center", type=str, default="mean", choices=["mean", "medoid", "herding"])
    parser.add_argument("--auto", action="store_true", default=False,
                        help="缺蒸馏产物时自动跑 distill.py (默认关, 建议先手动批量蒸馏)")
    parser.add_argument("--no_pixel", action="store_true", default=False,
                        help="跳过原图直训底线参考 (~73%%, 省几分钟训练)")
    args = parser.parse_args()
    ipcs = [int(x) for x in args.ipc.split() if int(x) > 0 and int(x) < SUBSET]
    main(ipcs, backbones=tuple(args.backbone), seeds=args.seeds, epochs=args.epochs,
         center=args.center, auto=args.auto, include_pixel=not args.no_pixel)
