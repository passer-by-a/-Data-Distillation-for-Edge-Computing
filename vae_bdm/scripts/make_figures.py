"""Step 7: 图表"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vae_bdm.scripts._common import load_cfg


def load_json(p):
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ds, R, F = cfg["dataset"], cfg["results_dir"], cfg["figures_dir"]
    sweep = load_json(os.path.join(R, f"{ds}_sweep.json"))["rows"]
    # 压缩比扫描若未包含 10:1，从主结果补齐（BDM 只取蒸馏种子 42）
    extra = []
    p = os.path.join(R, f"{ds}_baselines_r10.json")
    if os.path.exists(p):
        extra += json.load(open(p))["rows"]
    p = os.path.join(R, f"{ds}_bdm_r10.json")
    if os.path.exists(p):
        extra += [r for r in json.load(open(p))["rows"]
                  if r.get("dm_seed") == 42]
    have = {(r["method"], float(r["ratio"])) for r in sweep}
    for r in extra:
        key = (r["method"], float(r["ratio"]))
        if key not in have:
            sweep.append(r)
            have.add(key)
    main_path = os.path.join(R, f"{ds}_main_r10_summary.csv")

    methods = ["random", "kmeans", "kmeans_moment", "bdm_random_feature"]
    labels = {"random": "Random", "kmeans": "K-means",
              "kmeans_moment": "K-means + Moment",
              "bdm_random_feature": "Nonlinear DM"}
    plt.figure(figsize=(6.4, 4.6))
    for m in methods:
        rs = [r for r in sweep if r["method"] == m]
        if not rs:
            continue
        rs.sort(key=lambda r: float(r["ratio"]))
        plt.errorbar([float(r["ratio"]) for r in rs],
                     [float(r["acc_mean"]) for r in rs],
                     yerr=[float(r["acc_std"]) for r in rs],
                     marker="o", capsize=3, label=labels.get(m, m))
    full = next((r for r in sweep if r["method"] == "full"), None)
    if full:
        plt.axhline(float(full["acc_mean"]), ls="--", c="k", lw=1.2,
                    label="Full features")
    plt.xscale("log")
    plt.xticks([10, 20, 50, 100], ["10:1", "20:1", "50:1", "100:1"])
    plt.xlabel("compression ratio N/M")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title(f"{ds.upper()}: VAE-BDM vs coreset baselines")
    plt.grid(alpha=0.3); plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(F, f"{ds}_acc_vs_ratio.png"), dpi=200)
    plt.close()

    def bytes_plot(f1=False):
        plt.figure(figsize=(6.4, 4.6))
        for m in methods:
            rs = [r for r in sweep if r["method"] == m]
            if not rs:
                continue
            rs.sort(key=lambda r: float(r["ratio"]))
            x = [r["payload"]["int8"]["distill_bytes"] / 1024 for r in rs]
            y = [float(r["f1_mean"] if f1 else r["acc_mean"]) for r in rs]
            plt.plot(x, y, marker="o", label=labels.get(m, m))
        plt.xscale("log")
        plt.xlabel("uploaded payload (KiB, INT8, incl. labels)")
        plt.ylabel("Macro-F1 (%)" if f1 else "Top-1 accuracy (%)")
        plt.title(f"{ds.upper()}: {'F1' if f1 else 'accuracy'} vs upload bytes")
        plt.grid(alpha=0.3); plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(F, f"{ds}_{'f1' if f1 else 'acc'}_vs_bytes.png"), dpi=200)
        plt.close()

    bytes_plot(False); bytes_plot(True)

    if os.path.exists(main_path):
        rows = list(csv.DictReader(open(main_path)))
        rows = [r for r in rows if r["method"] in
                ("full", "random", "kmeans", "kmeans_moment",
                 "bdm_random_feature")]
        names = [r["method"] for r in rows]
        acc = [float(r["acc_mean"]) for r in rows]
        err = [float(r["acc_std"]) for r in rows]
        plt.figure(figsize=(7.6, 4.2))
        xs = np.arange(len(rows))
        plt.bar(xs, acc, yerr=err, capsize=4)
        plt.xticks(xs, names, rotation=25, ha="right")
        plt.ylabel("Top-1 accuracy (%)")
        plt.ylim(min(acc) - 2.0, max(acc) + 1.0)
        plt.title(f"{ds.upper()} @ 10:1")
        for x, v in zip(xs, acc):
            plt.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(F, f"{ds}_main_r10.png"), dpi=200)
        plt.close()

    print(f"图已输出: {F}")


if __name__ == "__main__":
    main()
