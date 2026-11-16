"""生成流式实验曲线: 准确率 / 累计通信 / 特征漂移与对齐误差

同一方法多个随机种子时绘制均值 ± 标准差。
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STYLE = {
    "frozen_stream_dm": ("frozen VAE + DM", "tab:blue"),
    "online_naive_dm": ("online VAE + naive DM", "tab:orange"),
    "online_replay_dm": ("online VAE + replay (re-encode)", "tab:green"),
    "online_replay_align_dm": ("online VAE + replay + align + DM", "tab:red"),
    "online_kmeans": ("online K-means", "tab:purple"),
    "online_herding": ("online Herding", "tab:brown"),
    "online_moment": ("moment calibration", "tab:pink"),
    "online_reservoir": ("online reservoir", "tab:gray"),
    "full_method": ("full method (patch 10%)", "black"),
}


def load(results_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"),
                              recursive=True)):
        if "_stale_pre_audit" in p.split(os.sep):
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if "summary" not in o or "rounds" not in o:
            continue
        if o["summary"].get("partial"):
            continue
        out.append(o)
    return out


def group_by_method(objs, scenario):
    groups = {}
    for o in objs:
        s = o["summary"]
        if s.get("scenario") != scenario:
            continue
        if s["method"] not in STYLE:      # 消融/离线结果不混入主图
            continue
        groups.setdefault(s["method"], []).append(o)
    return groups


def _curve(o, key):
    return [r[key] for r in o["rounds"]]


def _mean_std(runs):
    n = min(len(r) for r in runs)
    a = np.stack([np.asarray(r[:n], dtype=float) for r in runs], 0)
    return a.mean(0), (a.std(0) if len(runs) > 1 else None), n


def fig_accuracy(objs, out_dir, scenario="iid", prefix=""):
    plt.figure(figsize=(6.6, 4.3))
    for method, runs in group_by_method(objs, scenario).items():
        m, s, n = _mean_std([_curve(o, "acc") for o in runs])
        label, color = STYLE.get(method, (method, None))
        x = np.arange(-1, n - 1)
        plt.plot(x, m, marker="o", label=f"{label} (n={len(runs)})", color=color)
        if s is not None:
            plt.fill_between(x, m - s, m + s, color=color, alpha=0.15)
    plt.xlabel("stream round")
    plt.ylabel("test top-1 (%)")
    plt.title(f"CIFAR-100 streaming ({scenario}, mean±std over seeds)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    p = os.path.join(out_dir, f"{prefix}accuracy_{scenario}.png")
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    return p


def fig_comm(objs, out_dir, scenario="iid", prefix=""):
    plt.figure(figsize=(6.6, 4.3))
    for method, runs in group_by_method(objs, scenario).items():
        curves = []
        for o in runs:
            cum = [o["communication"]["initial_upload_bytes"]]
            for e in o["communication"]["entries"]:
                if e["kind"] == "round_sync":
                    cum.append(cum[-1] + e["theoretical_bytes"])
            curves.append([b / 1024 for b in cum])
        m, s, n = _mean_std(curves)
        label, color = STYLE.get(method, (method, None))
        x = np.arange(-1, n - 1)
        plt.plot(x, m, marker="o",
                 label=f"{label} ({runs[0]['summary']['sync_mode']})",
                 color=color)
        if s is not None:
            plt.fill_between(x, m - s, m + s, color=color, alpha=0.15)
    plt.xlabel("stream round")
    plt.ylabel("cumulative upload (KB)")
    plt.title(f"Cumulative communication ({scenario})")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    p = os.path.join(out_dir, f"{prefix}communication_{scenario}.png")
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    return p


def fig_drift(objs, out_dir, scenario="iid", prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    for method, runs in group_by_method(objs, scenario).items():
        drift, align = [], []
        for o in runs:
            rs = o["rounds"][1:]
            drift.append([r["anchor_drift"]["relative"] for r in rs])
            align.append([r.get("alignment", {}).get("relative_err")
                          for r in rs])
        if not drift or not drift[0]:
            continue
        label, color = STYLE.get(method, (method, None))
        m, s, n = _mean_std(drift)
        axes[0].plot(range(n), m, marker="o", label=label, color=color)
        if s is not None:
            axes[0].fill_between(range(n), m - s, m + s, color=color, alpha=0.15)
        if all(all(v is not None for v in a) for a in align):
            m2, s2, n2 = _mean_std(align)
            axes[1].plot(range(n2), m2, marker="o", label=label, color=color)
            if s2 is not None:
                axes[1].fill_between(range(n2), m2 - s2, m2 + s2,
                                     color=color, alpha=0.15)
    axes[0].set_title("anchor feature drift (relative MSE)")
    axes[1].set_title("diagonal affine held-out error")
    for ax in axes:
        ax.set_xlabel("stream round")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    p = os.path.join(out_dir, f"{prefix}drift_alignment_{scenario}.png")
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="vae_streaming/results")
    ap.add_argument("--figures", default="vae_streaming/figures")
    ap.add_argument("--scenario", default="iid")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    objs = load(os.path.join(ROOT, args.results))
    if args.dataset:
        objs = [o for o in objs if o["summary"].get("dataset") == args.dataset]
    prefix = f"{args.dataset}_" if args.dataset else ""
    fd = os.path.join(ROOT, args.figures)
    os.makedirs(fd, exist_ok=True)
    if not objs:
        print("没有结果可绘图")
        return
    for fn in (fig_accuracy, fig_comm, fig_drift):
        print(fn(objs, fd, args.scenario, prefix))


if __name__ == "__main__":
    main()
