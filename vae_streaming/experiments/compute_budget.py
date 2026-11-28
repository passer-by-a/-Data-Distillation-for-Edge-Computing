"""计算量与上行通信量统计（论文图 1 / 图 2）

图 1: DM / ExStream / Reservoir 的累计计算量分解（分组堆叠柱）
      - 左右两个面板: CIFAR-10 / CIFAR-100，共享纵轴尺度
      - 每个方法并排 Pixel / VAE latent 两根柱
      - 四段: Base 初始化 / 在线表示维护 / 在线代表更新 / 在线云端训练
图 2: Pixel 与 VAE 的累计上行特征载荷折线（0 = 已完成 Base 首次上传）

FLOPs 口径: 1 次乘加 = 2 FLOPs；只统计训练与表示构建，
验证/测试前向不计入（另计）。模型算子用 torch.utils.flop_counter
在 CPU 上实测，再乘以结果 JSON 中的实际执行次数。
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from vae import EdgeVAE
from vae_bdm.cloud import build_cloud
from random_feature import RandomMLP

RESULTS = Path("vae_streaming/results")
FIGURES = Path("vae_streaming/figures")
DATASETS = ["cifar10", "cifar100"]
METHODS = ["dm", "exstream", "online_reservoir"]
REPS = ["pixel", "vae"]
DIM = {"vae": 1024, "pixel": 3072}
NCLASS = {"cifar10": 10, "cifar100": 100}
NUM_ROUNDS = 80

COMPONENTS = ["base", "online_repr", "rep_update", "cloud_train"]
COMP_LABELS = {
    "base": "Base init",
    "online_repr": "Online repr. maintenance",
    "rep_update": "Representative update",
    "cloud_train": "Cloud training",
}
COMP_COLORS = {
    "base": "#4C72B0",
    "online_repr": "#55A868",
    "rep_update": "#C44E52",
    "cloud_train": "#8172B2",
}


# ----------------------------------------------------------------------
def _count(fn):
    try:
        ctx = FlopCounterMode(display=False)
    except TypeError:
        ctx = FlopCounterMode()
    with ctx as fc:
        fn()
    return float(fc.get_total_flops())


def measure_units():
    """单个算子单元的 FLOPs（每张图 / 每个特征）"""
    torch.manual_seed(0)
    vae = EdgeVAE(latent_dim=16, num_classes=10, base_channels=64)
    vae.train()
    x = torch.randn(4, 3, 32, 32)

    def vae_fb():
        out = vae(x)
        loss = out[0].mean() + out[2].mean() + out[3].mean()
        vae.zero_grad()
        loss.backward()

    units = {
        "vae_fwd": _count(lambda: vae(x)) / 4.0,
        "vae_fb": _count(vae_fb) / 4.0,
        "enc_fwd": _count(lambda: vae.encode_mu(x)) / 4.0,
    }
    vae.zero_grad()
    for c in (10, 100):
        net = build_cloud("mlp", 1024, c)
        xb = torch.randn(8, 1024)

        def cloud_fb():
            net.zero_grad()
            net(xb).sum().backward()
        units[f"cloud_fb_{c}"] = _count(cloud_fb) / 8.0
        net = build_cloud("mlp", 3072, c)
        xb = torch.randn(8, 3072)

        def cloud_fb_px():
            net.zero_grad()
            net(xb).sum().backward()
        units[f"cloud_fb_px_{c}"] = _count(cloud_fb_px) / 8.0
    # DM 冻结随机 MLP: 1024 → 2048 → 1024，Pixel 为 3072 → 2048 → 1024
    dm = RandomMLP(input_dim=1024, hidden_dim=2048, output_dim=1024, seed=0)
    xd = torch.randn(8, 1024)
    units["dm_fwd"] = _count(lambda: dm(xd)) / 8.0
    xd2 = xd.clone().requires_grad_(True)

    def dm_fb():
        dm.zero_grad()
        dm(xd2).sum().backward()
    units["dm_fb"] = _count(dm_fb) / 8.0
    dm_px = RandomMLP(input_dim=3072, hidden_dim=2048, output_dim=1024, seed=0)
    xp = torch.randn(8, 3072)
    units["dm_fwd_px"] = _count(lambda: dm_px(xp)) / 8.0
    xp2 = xp.clone().requires_grad_(True)

    def dm_fb_px():
        dm_px.zero_grad()
        dm_px(xp2).sum().backward()
    units["dm_fb_px"] = _count(dm_fb_px) / 8.0
    return units


# ----------------------------------------------------------------------
def load_run(dataset, method, rep, tag, seed):
    sub = "" if rep == "vae" else "pixel/"
    p = RESULTS / dataset / sub / "iid" / f"{method}_{tag}_seed{seed}.json"
    if not p.exists():
        alt = RESULTS / dataset / "iid" / f"{method}_{tag}_seed{seed}.json"
        if alt.exists():
            d = json.load(open(alt))
            if d.get("representation") == rep:
                return d
        return None
    return json.load(open(p))


def budget(d, rep, u, method):
    """返回四段累计 FLOPs（单位: TFLOPs）"""
    ds = d["stream"]["dataset"]
    c = NCLASS[ds]
    dim = DIM[rep]
    rounds = [r for r in d["rounds"] if r["round"] >= 0]
    n_base = int(d["stream"]["n_base"])
    n_stream = int(sum(d["stream"]["round_sizes"]))
    n_upd = sum(1 for r in rounds if r.get("vae_updated"))
    n_anchor = int(rounds[0]["alignment"].get("n_anchors", 0))
    cloud_init = 5 * 980              # 初始拟合: 5 epoch × 980 代表
    cloud_round = 5 * 980             # 每轮 5 epoch × 980 代表
    cloud_per_sample = u[f"cloud_fb_{c}"] if rep == "vae" \
        else u[f"cloud_fb_px_{c}"]

    out = {}
    if rep == "vae":
        # 在线 VAE: 每 2 轮一次，每次 150 步 × 128 batch
        n_train_imgs = n_upd * 150 * 128
        base = (30 * n_base * u["vae_fb"] + n_base * u["enc_fwd"]
                + cloud_init * cloud_per_sample)
        online = (n_train_imgs * u["vae_fb"]
                  + (n_stream + 2 * n_anchor * len(rounds)) * u["enc_fwd"])
        extra_re = {"dm": 1000, "exstream": 0, "online_reservoir": 980}
    else:
        std = 2.0 * dim
        base = n_base * std + cloud_init * cloud_per_sample
        online = n_stream * std
        extra_re = {"dm": 1000, "exstream": 0, "online_reservoir": 980}
    if rep == "vae":
        online += len(rounds) * extra_re[method] * u["enc_fwd"]
    else:
        online += len(rounds) * extra_re[method] * 2.0 * dim

    if method == "dm":
        fwd_u = u["dm_fwd"] if rep == "vae" else u["dm_fwd_px"]
        fb_u = u["dm_fb"] if rep == "vae" else u["dm_fb_px"]
        rep_cost = len(rounds) * (
            (1000 + n_stream / len(rounds)) * fwd_u + 20 * 980 * fb_u)
    elif method == "exstream":
        # 类缓冲满后，每个新样本一次类内 q×q 距离矩阵（cdist）
        q = 980.0 / c
        per_round = (n_stream / len(rounds)) * q * q * dim * 2.0
        rep_cost = len(rounds) * per_round
    else:
        rep_cost = 0.0
    out["base"] = base / 1e12
    out["online_repr"] = online / 1e12
    out["rep_update"] = rep_cost / 1e12
    out["cloud_train"] = len(rounds) * cloud_round * cloud_per_sample / 1e12
    out["cloud_init"] = cloud_init * cloud_per_sample / 1e12
    out["_meta"] = {"n_updates": n_upd, "n_base": n_base,
                    "n_stream": n_stream}
    return out


# ----------------------------------------------------------------------
def plot_compute(data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # VAE 与 Pixel 的计算量相差 1-3 个数量级（在线 VAE 训练占主导），
    # 因此左右柱分别使用左右两个纵轴，否则 Pixel 柱不可见。
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    axes2 = []
    width = 0.34
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        ax2 = ax.twinx()
        axes2.append(ax2)
        x = np.arange(len(METHODS))
        for m_i, m in enumerate(METHODS):
            b = data[ds]["vae"][m]
            bottom = 0.0
            for k in COMPONENTS:
                ax.bar(x[m_i] - width / 2, b[k], width, bottom=bottom,
                       color=COMP_COLORS[k], edgecolor="white",
                       linewidth=0.4)
                bottom += b[k]
            ax.text(x[m_i] - width / 2, bottom * 1.01, f"{bottom:.0f}",
                    ha="center", va="bottom", fontsize=6.5)
            b = data[ds]["pixel"][m]
            bottom = 0.0
            for k in COMPONENTS:
                ax2.bar(x[m_i] + width / 2, b[k], width, bottom=bottom,
                        color=COMP_COLORS[k], edgecolor="white",
                        linewidth=0.4, alpha=0.75, hatch="//")
                bottom += b[k]
            ax2.text(x[m_i] + width / 2, bottom * 1.02, f"{bottom:.0f}",
                     ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["DM", "ExStream", "Reservoir"])
        ax.set_title(ds.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 2100)
        ax2.set_ylim(0, 70)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    axes[0].set_ylabel("VAE latent — cumulative compute (TFLOPs)")
    axes2[1].set_ylabel("Pixel 3072 — cumulative compute (TFLOPs)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMP_COLORS[k])
               for k in COMPONENTS]
    labels = [COMP_LABELS[k] for k in COMPONENTS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="0.6"),
                plt.Rectangle((0, 0), 1, 1, facecolor="0.85", hatch="//",
                              edgecolor="0.4")]
    labels += ["left bar: VAE (left axis)", "right bar: Pixel (right axis)"]
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = out_dir / "compute_stack.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"写出 {out}")


def plot_compute_cloud(data, out_dir):
    """只统计云端侧计算：初始拟合 + 80 轮在线训练（与边缘表示成本无关）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    segs = [("cloud_init", "Cloud init fit (5 ep)", "#9DB8DA"),
            ("cloud_train", "Online cloud training (80 × 5 ep)", "#8172B2")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    width = 0.34
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        x = np.arange(len(METHODS))
        for k, rep in enumerate(REPS):
            off = (-0.5 if rep == "vae" else 0.5) * width
            bottoms = []
            for m in METHODS:
                b = data[ds][rep][m]
                bottom = 0.0
                for key, _, color in segs:
                    ax.bar(x + off, b[key], width, bottom=bottom,
                           color=color, edgecolor="white", linewidth=0.4,
                           hatch=None if rep == "vae" else "//")
                    bottom += b[key]
                bottoms.append(bottom)
            for xi, tot in zip(x + off, bottoms):
                ax.text(xi, tot * 1.02, f"{tot:.2f}", ha="center",
                        va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["DM", "ExStream", "Reservoir"])
        ax.set_title(ds.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Cloud-side cumulative compute (TFLOPs)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="0.75"),
                plt.Rectangle((0, 0), 1, 1, facecolor="0.85", hatch="//",
                              edgecolor="0.4")]
    labels = [lab for _, lab, _ in segs]
    labels += ["left bar: VAE latent", "right bar: Pixel 3072"]
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = out_dir / "compute_cloud_side.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"写出 {out}")


def plot_comm(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_round = {"Pixel 3072": 980 * 3072 / 1e6,
                 "VAE latent": 980 * 1024 / 1e6}
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for (name, mb), color, mk in zip(per_round.items(),
                                     ["#C44E52", "#4C72B0"],
                                     ["s", "o"]):
        ys = [mb * (i + 1) for i in range(NUM_ROUNDS + 1)]
        ax.plot(range(NUM_ROUNDS + 1), ys, "-", color=color,
                linewidth=2.0, marker=mk, markevery=20, markersize=5,
                label=f"{name} ({mb:.3f} MB/round)")
        ax.annotate(f"{ys[-1]:.1f} MB", (NUM_ROUNDS, ys[-1]),
                    textcoords="offset points", xytext=(-10, -14),
                    ha="right", va="top", fontsize=8.5, color=color,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=1.2))
    ax.set_xlabel("Completed streaming rounds (0 = base uploaded)")
    ax.set_ylabel("Cumulative uplink feature payload (MB)")
    ax.set_xlim(0, NUM_ROUNDS)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Pixel vs VAE latent: feature payload only "
                 "(INT8, 980 reps/round)", fontsize=9)
    fig.tight_layout()
    out = out_dir / "comm_cumulative.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"写出 {out}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ep80v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(FIGURES))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    u = measure_units()
    print("单元 FLOPs:")
    for k, v in u.items():
        print(f"  {k}: {v/1e9:.3f} GFLOPs")

    data = {}
    rows = []
    for ds in DATASETS:
        data[ds] = {}
        for rep in REPS:
            data[ds][rep] = {}
            for m in METHODS:
                d = load_run(ds, m, rep, args.tag, args.seed)
                if d is None:
                    raise SystemExit(f"缺少 {ds}/{rep}/{m}")
                b = budget(d, rep, u, m)
                data[ds][rep][m] = b
                rows.append({"dataset": ds, "representation": rep,
                             "method": m,
                             **{k: round(b[k], 4) for k in COMPONENTS},
                             "cloud_init": round(b["cloud_init"], 4),
                             "cloud_side": round(b["cloud_init"]
                                                 + b["cloud_train"], 4),
                             "total": round(sum(b[k] for k in COMPONENTS), 4)})
    (out_dir / "compute_budget.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    plot_compute(data, out_dir)
    plot_compute_cloud(data, out_dir)
    plot_comm(out_dir)
    for r in rows:
        print(f"{r['dataset']:8s} {r['representation']:5s} "
              f"{r['method']:17s} total {r['total']:8.1f} "
              f"(base {r['base']:.1f}, repr {r['online_repr']:.1f}, "
              f"update {r['rep_update']:.1f}, cloud {r['cloud_train']:.1f})")


if __name__ == "__main__":
    main()
