"""
论文消融图: VAE 分类监督 (cls) vs 纯重建 (noclspart) 的下游训练 loss

产出:
  figures/ablation_vae_cls_loss_train.pdf (+ .png)
数据:
  results/ablation_cls_vs_noclspart_ipc100.json  (由 exp_ablation.py 生成)

运行:
  python exp_ablation.py     # 若结果不存在
  python plot_acc_loss.py

说明: `curve_bounds` / `square_grid_fixed_x` / `save` 同时被 exp_method_compare.py 复用。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

FIG_DIR = "figures"
ABLATION_JSON = os.path.join("results", "ablation_cls_vs_noclspart_ipc100.json")

ABL_KEYS = ["resnet", "vit"]                  # 消融只跑了这两个 backbone
ABL_NAMES = ["ResNet", "ViT"]
# 每条曲线一个 marker (线型仍表示变体: 实线=带分类头, 虚线=无分类头)
ABL_MARKERS = {("resnet", "cls"): "o", ("resnet", "noclspart"): "v",
               ("vit", "cls"): "s", ("vit", "noclspart"): "^"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
    png = os.path.join(FIG_DIR, f"{stem}.png")
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


def main():
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


if __name__ == "__main__":
    main()
