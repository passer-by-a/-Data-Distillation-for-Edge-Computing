"""云端分类器训练曲线：准确率 / train loss / test loss。

3 张图，每张 2×2 布局：行 = 数据集，列 = 左 VAE / 右 像素表示。
横轴为云端 epoch（新结果每个 epoch 一个点；旧结果按每轮点数换算）。
用法:
  python -m vae_streaming.experiments.plot_cloud_curves \
      --tag ep80v2 --seed 42 --methods dm,dm_multi,exstream,gcr,online_reservoir
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METRICS = [
    ("acc", "Test accuracy (%)", "accuracy"),
    ("train_loss", "Train loss (CE)", "train_loss"),
    ("test_loss", "Test loss (CE)", "test_loss"),
]
REPS = [("vae", "VAE latent"), ("pixel", "Pixel 3072")]
DATASETS = ["cifar10", "cifar100"]
# 图中显示名与固定配色（与 generate_report 的 5 方法口径一致）
METHODS = ["dm", "dm_multi", "exstream", "gcr", "online_reservoir"]
DISPLAY = {"dm": "DM", "dm_multi": "DM-Multi", "exstream": "ExStream",
           "gcr": "GCR", "online_reservoir": "Reservoir"}
COLORS = {"dm": "tab:blue", "dm_multi": "tab:orange",
          "exstream": "tab:green", "gcr": "tab:red",
          "online_reservoir": "tab:purple"}


def _repr_of(obj):
    """结果 JSON 实际使用的表示（summary / config 字段）。"""
    s = obj.get("summary") or {}
    return str(s.get("representation")
               or (obj.get("config") or {}).get("representation") or "vae")


def _result_path(dataset, method, tag, seed, rep):
    name = f"{method}_{tag}_seed{seed}.json"
    # 新布局: results/{ds}/{rep}/iid/；旧 pixel 结果曾混写在 results/{ds}/iid/
    for cand in (RESULTS / dataset / rep / "iid" / name,
                 RESULTS / dataset / "iid" / name):
        if not cand.exists():
            continue
        # 必须核对文件内的表示，避免把 VAE 结果当成像素结果
        try:
            with open(cand) as f:
                obj = json.load(f)
        except (OSError, ValueError):
            continue
        if _repr_of(obj) == rep:
            return cand, obj
    return None, None


def load_curves(dataset, methods, tag, seed):
    """返回 {(rep, method): {"x", "xlabel", "acc", "train_loss", "test_loss"}}。

    横轴为云端分类器 epoch：前段（1..base_ep）是 Initial Base 代表集上的
    初始拟合，rounds[-1].per_epoch 存在时逐 epoch 画出；之后是流式阶段的
    逐 epoch 记录（rounds[*].per_epoch）；旧结果只有每轮一个点，横轴按
    (round+1)×epochs_per_round 换算，初始拟合只保留终点（epoch 0）。
    """
    out = {}
    for rep, _ in REPS:
        for m in methods:
            p, obj = _result_path(dataset, m, tag, seed, rep)
            if p is None:
                print(f"  [skip] {dataset}/{rep}/{m}: 无结果文件")
                continue
            init = next((r for r in obj["rounds"]
                         if int(r.get("round", -1)) == -1), None)
            recs = [r for r in obj["rounds"]
                    if int(r.get("round", -1)) >= 0]
            recs.sort(key=lambda r: r["round"])
            eps = [e for r in recs for e in (r.get("per_epoch") or [])]
            if eps:
                eps.sort(key=lambda e: e["epoch"])
                curve = {
                    "x": [e["epoch"] for e in eps],
                    "xlabel": "cloud epoch",
                    **{k: [e.get(k) for e in eps] for k, _, _ in METRICS},
                }
            else:
                n_ep = int((recs[0].get("cloud") or {}).get(
                    "epochs_per_round") or 1) if recs else 1
                curve = {
                    "x": [(int(r["round"]) + 1) * n_ep for r in recs],
                    "xlabel": f"cloud epoch (1 point / {n_ep} epochs)",
                    **{k: [r.get(k) for r in recs] for k, _, _ in METRICS},
                }
            init_eps = (init or {}).get("per_epoch") or []
            if init_eps:
                # 初始拟合逐 epoch 轨迹：epoch 1..base_ep
                init_eps.sort(key=lambda e: e["epoch"])
                curve["x"] = [e["epoch"] for e in init_eps] + curve["x"]
                for k, _, _ in METRICS:
                    curve[k] = ([e.get(k) for e in init_eps]
                                + curve[k])
            elif init is not None:
                # 旧结果：初始拟合只留终点，作为 epoch 0 的起点
                curve["x"] = [0] + curve["x"]
                for k, _, _ in METRICS:
                    curve[k] = [init.get(k)] + curve[k]
            out[(rep, m)] = curve
    return out


def plot_metric(metric, ylabel, fname, data, methods, out_dir, tag, seed):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), sharex=True)
    handles = {}
    for i, ds in enumerate(DATASETS):
        for j, (rep, rep_name) in enumerate(REPS):
            ax = axes[i][j]
            for m in methods:
                c = data.get(ds, {}).get((rep, m))
                if not c:
                    continue
                y = c[metric]
                if all(v is None for v in y):
                    continue
                ln, = ax.plot(c["x"], y, label=DISPLAY.get(m, m),
                              color=COLORS.get(m), linewidth=1.6)
                handles.setdefault(DISPLAY.get(m, m), ln)
            ax.set_title(f"{ds.upper()} · {rep_name}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel(
                    "cloud epoch (first 5 = initial base fit)")
    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=len(handles), frameon=False, fontsize=10)
    fig.suptitle(f"{ylabel} (tag {tag}, seed {seed})")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = out_dir / f"{fname}_seed{seed}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  写出 {out}")


def plot_metric_combined(metric, ylabel, fname, data, methods, out_dir,
                         tag, seed):
    """每个数据集单独一张方形图：颜色 = 方法，实线 = VAE，虚线 = Pixel"""
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        for m in methods:
            for rep, rep_name in REPS:
                c = data.get(ds, {}).get((rep, m))
                if not c:
                    continue
                y = c[metric]
                if all(v is None for v in y):
                    continue
                style = "-" if rep == "vae" else "--"
                ax.plot(c["x"], y, style, color=COLORS.get(m),
                        linewidth=1.6 if rep == "vae" else 1.4,
                        label=f"{DISPLAY.get(m, m)} · {rep_name}")
        ax.set_title(f"{ds.upper()}  (tag {tag}, seed {seed})", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("cloud epoch (first 5 = initial base fit)")
        ax.grid(alpha=0.3)
        # 刻度每 100 epoch 一格，视觉上不那么密
        ax.xaxis.set_major_locator(MultipleLocator(100))
        # 顶部留出图例空间，避免遮住曲线
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.16 * (hi - lo))
        ax.legend(loc="upper right", ncol=2, fontsize=6.8,
                  frameon=True, framealpha=0.9, borderpad=0.35,
                  handlelength=2.0, columnspacing=0.9)
        fig.suptitle(f"{ylabel} — solid: VAE, dashed: Pixel", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = out_dir / f"{fname}_{ds}_3m_combined_seed{seed}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  写出 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ep80v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--datasets", default="cifar10,cifar100")
    ap.add_argument("--combined", action="store_true",
                    help="VAE/Pixel 同图（颜色=方法，实线/虚线=表示）")
    ap.add_argument("--out", default=str(ROOT / "figures" / "cloud_curves"))
    args = ap.parse_args()

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    elif args.combined:
        methods = ["dm_multi", "exstream", "online_reservoir"]
    else:
        methods = list(METHODS)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {ds: load_curves(ds, methods, args.tag, args.seed)
            for ds in datasets}
    plot_fn = plot_metric_combined if args.combined else plot_metric
    for metric, ylabel, fname in METRICS:
        plot_fn(metric, ylabel, fname, data, methods, out_dir,
                args.tag, args.seed)


if __name__ == "__main__":
    main()
