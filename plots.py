"""论文图（子命令）

  python plots.py --fig all        # 三张对比图 + 四张消融图 + summary_all.csv
  python plots.py --fig frontier   # 精度-成本前沿
  python plots.py --fig probe      # 下游分类器标签预算
  python plots.py --fig pretrain   # 预训练 CE 标签预算
  python plots.py --fig ablation   # 时间步 / 主对比 / 投影 / 分类头消融
  python plots.py --fig cls-loss   # 旧蒸馏线的 VAE 分类监督 loss 曲线

合并自 plot_compare.py / make_figures.py / plot_acc_loss.py。
`curve_bounds` / `square_grid_fixed_x` / `save` 仍供 legacy/exp_method_compare.py 复用。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D, "results")
F = os.path.join(D, "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 180, "axes.grid": True,
                     "grid.alpha": 0.3})

ABLATION_JSON = os.path.join(R, "ablation_cls_vs_noclspart_ipc100.json")
ABL_KEYS = ["resnet", "vit"]
ABL_NAMES = ["ResNet", "ViT"]
ABL_MARKERS = {("resnet", "cls"): "o", ("resnet", "noclspart"): "v",
               ("vit", "cls"): "s", ("vit", "noclspart"): "^"}


# ============================================================
# 对比三图
# ============================================================

def fig_frontier():
    """精度 vs 推理 FLOPs / 延迟"""
    pts = [
        ("VAE-S", 90.45, 253, 0.80, "#d62728", "s"),
        ("DDPM-FT", 89.91, 3241, 4.04, "#1f77b4", "o"),
        ("DDPM-Joint", 85.62, 3241, 4.04, "#ff7f0e", "o"),
        ("DDPM (frozen)", 79.07, 3241, 4.04, "#1f77b4", "^"),
        ("VAE (frozen)", 38.96, 253, 0.80, "#d62728", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, xlabel in [(axes[0], 2, "Inference FLOPs (M, log)"),
                            (axes[1], 3, "Latency per image (ms)")]:
        for name, acc, fl, lat, c, mk in pts:
            x = fl if key == 2 else lat
            ax.scatter(x, acc, s=110, c=c, marker=mk, zorder=3,
                       edgecolors="white", linewidths=1.2)
            left = name.startswith("DDPM")
            ax.annotate(name, (x, acc), textcoords="offset points",
                        xytext=(-10, -12) if left else (8, -4), fontsize=9,
                        ha="right" if left else "left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_ylim(30, 97)
        if key == 2:
            ax.set_xscale("log")
            ax.set_xlim(150, 6000)
        else:
            ax.set_xlim(0, 5.2)
        ax.set_title("supervised (●/■) vs unsupervised (▲)")
    fig.suptitle("DDPM vs VAE: 1024-dim features, CIFAR-10", fontsize=12)
    fig.tight_layout()
    p = os.path.join(F, "ddpm_vs_vae_frontier.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_probe():
    """下游分类器标签预算 (预训练: DDPM 零标签 / VAE-S 全标签)"""
    shots = [1, 5, 20, 50, 4900]
    data = {
        "DDPM-A": ([21.03, 36.84, 50.82, 60.43, 79.07],
                   [1.53, 1.31, 0.83, 0.31, 0.18], "#1f77b4", "o", "-"),
        "DDPM-B1": ([19.37, 30.03, 41.05, 50.87, 72.59],
                    [1.22, 0.69, 1.01, 0.48, 0.31], "#ff7f0e", "s", "-"),
        "VAE-S (pretrained with full labels)":
            ([65.36, 81.36, 86.70, 88.21, 90.45],
             [1.00, 0.31, 0.18, 0.14, 0.06], "#d62728", "^", "--"),
        "VAE (label-free pretrain)":
            ([17.98, 19.72, 21.06, 23.53, 38.96],
             [0.29, 0.29, 0.14, 0.12, 0.11], "#7f7f7f", "v", "-"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, (acc, std, c, mk, ls) in data.items():
        ax.errorbar(shots, acc, yerr=std, marker=mk, color=c, label=name,
                    capsize=3, lw=1.8, ls=ls)
    ax.set_xscale("log")
    ax.set_xticks(shots)
    ax.set_xticklabels(["1", "5", "20", "50", "4900\n(full)"])
    ax.set_xlabel("labeled samples per class (downstream classifier)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Label budget vs accuracy (CIFAR-10, 1024-dim)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    p = os.path.join(F, "label_budget.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_pretrain():
    """表征的标签成本: 预训练标签数 → 下游都给全标签时的精度"""
    vae_x = [5, 20, 100, 500, 4900]
    vae_y = [47.32, 51.54, 55.54, 66.49, 83.25]
    vae_s = [0.07, 0.07, 0.07, 0.05, 0.07]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhline(79.07, color="#1f77b4", lw=1.8,
               label="DDPM-A: 0 pretraining labels (frozen features)")
    ax.plot(vae_x, vae_y, marker="s", color="#d62728", lw=1.8,
            label="VAE: N labels in pretraining CE (frozen features)")
    ax.scatter([0], [38.96], marker="v", s=90, color="#7f7f7f", zorder=5,
               edgecolors="white", linewidths=1.0,
               label="VAE: 0 pretraining labels (no CE)")
    ax.scatter([0], [79.07], marker="o", s=90, color="#1f77b4", zorder=5,
               edgecolors="white", linewidths=1.0)
    ax.scatter([4900], [90.45], marker="*", s=230, color="#d62728", zorder=5,
               edgecolors="white", linewidths=1.0)
    ax.annotate("VAE-S original\n(+augmentation)", (4900, 90.45),
                textcoords="offset points", xytext=(-95, -4), fontsize=8.5)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.5, 9000)
    ax.set_xticks([0, 5, 20, 100, 500, 4900])
    ax.set_xticklabels(["0", "5", "20", "100", "500", "4900\n(full)"])
    ax.set_xlabel("labeled samples per class used in representation pretraining")
    ax.set_ylabel("Top-1 accuracy (%) with full-label probe")
    ax.set_title("Label cost of the representation (CIFAR-10, 1024-dim)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    p = os.path.join(F, "label_cost_representation.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ============================================================
# 消融四图 + 汇总 CSV
# ============================================================

def agg(path, modes=None):
    df = pd.read_csv(path)
    if modes:
        df = df[df["mode"].isin(modes)]
    g = df.groupby("mode")["test_acc"].agg(["mean", "std"]).reset_index()
    return g.sort_values("mean", ascending=False)


def fig_ablation():
    colors = {"A": "#1f77b4", "B1": "#ff7f0e", "B2": "#2ca02c",
              "VAE": "#d62728", "VAE_unsup": "#9467bd"}

    # 1. 时间步曲线
    ts = [0, 20, 50, 100, 200]
    files = {0: "core_core_c10_t0.csv",
             20: "core_c10_tsweep_t20.csv", 50: "core_c10_tsweep_t50.csv",
             100: "core_c10_tsweep_t100.csv", 200: "core_c10_tsweep_t200.csv"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ["A", "B1", "B2"]:
        vals = []
        for t in ts:
            g = agg(os.path.join(R, files[t]), [m])
            vals.append(float(g["mean"].iloc[0]))
        ax.plot(ts, vals, marker="o", label=f"DDPM-{m}", color=colors[m])
        for t, v in zip(ts, vals):
            if t == 20:
                ax.annotate(f"{v:.1f}", (t, v), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=9, color=colors[m])
    ax.set_xlabel("diffusion timestep $t$")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Effect of timestep (CIFAR-10, 1024-dim features)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "timestep_vs_accuracy.png"))
    plt.close(fig)

    # 2. 主对比 (t=20)
    main_df = pd.concat([pd.read_csv(os.path.join(R, "core_c10_t20_3seed.csv")),
                         pd.read_csv(os.path.join(R, "core_c10_t0_vae.csv"))])
    g = main_df[main_df["mode"].isin(["A", "B1", "B2", "VAE", "VAE_unsup"])] \
        .groupby("mode")["test_acc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(g))
    ax.bar(xs, g["mean"], yerr=g["std"], capsize=4,
           color=[colors[m] for m in g["mode"]])
    ax.set_xticks(xs)
    ax.set_xticklabels(["DDPM-A", "DDPM-B1", "DDPM-B2", "VAE (sup.)", "VAE (unsup)"]
                       if list(g["mode"]) == ["A", "B1", "B2", "VAE", "VAE_unsup"]
                       else list(g["mode"]))
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("1024-element features, CIFAR-10 (t=20, 3 seeds)")
    for x, v in zip(xs, g["mean"]):
        ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "main_comparison.png"))
    plt.close(fig)

    # 3. 压缩消融
    c = agg(os.path.join(R, "core_c10_compress.csv"))
    order = [m for m in ["A", "A_raw", "A_gap", "B1", "B_raw", "B_gap"]
             if m in set(c["mode"])]
    c = c.set_index("mode").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = np.arange(len(c))
    ax.bar(xs, c["mean"], yerr=c["std"], capsize=4,
           color=["#1f77b4" if m.startswith("A") else "#ff7f0e" for m in c["mode"]])
    ax.set_xticks(xs)
    labels = []
    for m, d in zip(c["mode"], c["dim"] if "dim" in c else [None] * len(c)):
        labels.append(m.replace("_raw", " raw").replace("_gap", " GAP"))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Projection / pooling ablation (t=20)")
    for x, v in zip(xs, c["mean"]):
        ax.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "compression_ablation.png"))
    plt.close(fig)

    # 4. Linear vs MLP
    lin = pd.read_csv(os.path.join(R, "core_c10_t20_3seed.csv"))
    mlp = pd.read_csv(os.path.join(R, "core_c10_mlp.csv"))
    rows = []
    for m in ["A", "B1", "VAE"]:
        l = lin[lin["mode"] == m]["test_acc"]
        h = mlp[mlp["mode"] == m]["test_acc"]
        if len(l) and len(h):
            rows.append((m, l.mean(), l.std(), h.mean(), h.std()))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.18, [r[1] for r in rows], width=0.36, label="Linear",
           yerr=[r[2] for r in rows], capsize=3)
    ax.bar(xs + 0.18, [r[3] for r in rows], width=0.36, label="MLP (512)",
           yerr=[r[4] for r in rows], capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"DDPM-{r[0]}" if r[0] != "VAE" else "VAE" for r in rows])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Classifier head ablation (t=20)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "head_ablation.png"))
    plt.close(fig)

    # 5. 汇总 CSV
    out = []
    for name, path in [("core_t0", "core_core_c10_t0.csv"),
                       ("core_t20", "core_c10_t20_3seed.csv"),
                       ("vae_t0", "core_c10_t0_vae.csv"),
                       ("compress", "core_c10_compress.csv"),
                       ("mlp", "core_c10_mlp.csv")]:
        df = pd.read_csv(os.path.join(R, path))
        g = df.groupby("mode").agg(
            acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
            f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"),
            proj_params=("proj_params", "first"), head_params=("head_params", "first"),
            backbone_params=("backbone_params", "first"),
            flops_m=("flops_m", "first"), latency_ms=("latency_ms", "first")).reset_index()
        g.insert(0, "experiment", name)
        out.append(g)
    pd.concat(out).to_csv(os.path.join(R, "summary_all.csv"), index=False)
    print("figures:", sorted(os.listdir(F)))
    print("summary:", os.path.join(R, "summary_all.csv"))


# ============================================================
# 旧蒸馏线: VAE 分类监督 loss 曲线
# ============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(F, exist_ok=True)
    pdf = os.path.join(F, f"{stem}.pdf")
    png = os.path.join(F, f"{stem}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图已保存: {pdf} / {png}")


def curve_bounds(lo, hi, kind="loss", frac=0.28):
    """曲线图的数据范围 → (y 下限, 至少要到的 y 高度)

    loss: 下界最多到 -0.2 且不出刻度; acc: 上下对称留白 (给图例/曲线空间)。
    """
    if kind == "acc":
        return lo - frac * (hi - lo), hi + frac * (hi - lo)
    return min(lo - 0.2, 0.0), hi + frac * (hi - lo)


def square_grid_fixed_x(ax, x_step=5.0, bottom=None, top_min=None,
                        y_steps=(0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0,
                                 2.0, 2.5, 5.0), n_iter=4):
    """固定 x 刻度步长 (默认 5 → 0, 5, 10, ...), 靠"拉伸 y 轴"让背景格子为正方形。

    bottom / top_min: 数据要求的 y 下限与"至少要到的高度"(含图例留白); None 则沿用当前。
    返回 (y_step, 实测格宽/格高); 无可行解时返回 None。
    """
    fig = ax.figure
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    if bottom is None:
        bottom = ax.get_ylim()[0]
    if top_min is None:
        top_min = ax.get_ylim()[1]
    need = top_min - bottom
    x0, x1 = ax.get_xlim()
    n_x = (x1 - x0) / x_step
    if n_x <= 0:
        return None
    ys = None
    for _ in range(n_iter):
        fig.canvas.draw()                      # 量真实坐标区尺寸 (随刻度标签宽度变化)
        bb = ax.get_window_extent()
        w_in, h_in = bb.width / fig.dpi, bb.height / fig.dpi
        n_y_ideal = n_x * h_in / w_in          # 正方形格子所需的纵向格数
        feasible = [c for c in y_steps if n_y_ideal * c >= need - 1e-9]
        ys = min(feasible) if feasible else y_steps[-1]
        ax.yaxis.set_major_locator(MultipleLocator(ys))
        ax.set_ylim(bottom, bottom + n_y_ideal * ys)   # 拉伸到恰好 n_y 个格子
    fig.canvas.draw()
    bb = ax.get_window_extent()
    w_in, h_in = bb.width / fig.dpi, bb.height / fig.dpi
    y0, y1 = ax.get_ylim()
    ratio = (w_in / n_x) / (h_in / ((y1 - y0) / ys))
    return ys, ratio


def plot_ablation_cls_loss(abl, split):
    """VAE 带/不带分类头的下游 loss 曲线 (实线=带分类头 / 虚线=无分类头)"""
    key = "train_curve" if split == "train" else "test_curve"
    col = {"resnet": "#1f77b4", "vit": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(5.0, 3.75))
    for mode, ls, tag in (("cls", "-", "with cls"),
                          ("noclspart", "--", "without cls")):
        for k, name in zip(ABL_KEYS, ABL_NAMES):
            curve = abl["results"][mode][k][key]
            ax.plot(range(1, len(curve) + 1), curve, color=col[k], ls=ls, lw=1.5,
                    marker=ABL_MARKERS[(k, mode)], markevery=5, markersize=6,
                    label=f"{name}, {tag}")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, len(abl["results"]["cls"][ABL_KEYS[0]][key]))
    lo = min(min(abl["results"][m][k][key]) for m in ("cls", "noclspart") for k in ABL_KEYS)
    hi = max(max(abl["results"][m][k][key]) for m in ("cls", "noclspart") for k in ABL_KEYS)
    bottom, top_min = curve_bounds(lo, hi, "loss", 0.30)   # 顶部留白给右上角竖排图例
    ax.legend(ncol=1, frameon=False, fontsize=9, loc="upper right",
              handlelength=2.0, handletextpad=0.5, labelspacing=0.35)
    fig.tight_layout()
    ys, ratio = square_grid_fixed_x(ax, x_step=5.0, bottom=bottom, top_min=top_min)
    print(f"  [ablation_cls_{split}] x_step=5, y_step={ys}, 格宽/格高={ratio:.3f}")
    save(fig, f"ablation_vae_cls_loss_{split}")


def fig_cls_loss():
    abl = load_json(ABLATION_JSON)
    am = abl["meta"]
    print("========== VAE 分类监督消融 (CIFAR-10) ==========")
    print(f"ipc={am['ipc']}, subset={am['subset']}, {am['seeds']} seeds, "
          f"{am['epochs']} ep ← {os.path.basename(ABLATION_JSON)}")
    print(f"{'backbone':<11}{'w/ cls':>16}{'w/o cls':>16}{'Drop (pt)':>11}")
    for k, name in zip(ABL_KEYS, ABL_NAMES):
        c, cs = abl["results"]["cls"][k]["acc"], abl["results"]["cls"][k]["std"]
        n, ns = (abl["results"]["noclspart"][k]["acc"],
                 abl["results"]["noclspart"][k]["std"])
        print(f"{name:<11}{f'{c:.2f}±{cs:.2f}':>16}{f'{n:.2f}±{ns:.2f}':>16}"
              f"{n - c:>11.2f}")
    plot_ablation_cls_loss(abl, "train")
    print("=================================================")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="论文图")
    ap.add_argument("--fig", default="all",
                    choices=["all", "frontier", "probe", "pretrain",
                             "ablation", "cls-loss"])
    a = ap.parse_args()

    if a.fig in ("all", "frontier"):
        print(fig_frontier())
    if a.fig in ("all", "probe"):
        print(fig_probe())
    if a.fig in ("all", "pretrain"):
        print(fig_pretrain())
    if a.fig in ("all", "ablation"):
        fig_ablation()
    if a.fig == "cls-loss":
        fig_cls_loss()


if __name__ == "__main__":
    main()
