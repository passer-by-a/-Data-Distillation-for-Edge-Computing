"""汇总主表: baselines + BDM variants + quant"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from vae_bdm.scripts._common import load_cfg
from vae_bdm.features import load_features
from vae_bdm.evaluate import reps_path


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def class_var_ratio(feats, reps, rep_y, num_classes):
    """类内方差的合成/真实比（逐类算，再对类平均）"""
    x = feats["train"]["x"].numpy()
    y = feats["train"]["y"].numpy()
    ratios = []
    for c in range(num_classes):
        rc = x[y == c]
        sc = reps[rep_y == c]
        if len(sc) < 2:
            continue
        vr = float(rc.var(0).mean())
        vs = float(sc.var(0).mean())
        ratios.append(vs / max(vr, 1e-12))
    return float(np.mean(ratios)) if ratios else float("nan")


def attach_var_ratio(cfg, feats, rec, ratio):
    if rec["method"] == "full":
        rec["var_ratio"] = 1.0
        return
    p = reps_path(cfg, rec["method"], ratio)
    if not os.path.exists(p):
        rec["var_ratio"] = float("nan")
        return
    obj = np.load(p, allow_pickle=True)
    rec["var_ratio"] = class_var_ratio(feats, obj["features"], obj["labels"],
                                       cfg["num_classes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ds, r = cfg["dataset"], a.ratio
    R = cfg["results_dir"]
    feats = load_features(cfg)
    base = load(os.path.join(R, f"{ds}_baselines_r{r:g}.json"))
    bdm = load(os.path.join(R, f"{ds}_bdm_r{r:g}.json"))
    full = None
    rows = []
    if base:
        for rec in base["rows"]:
            if rec["method"] == "full":
                full = rec
            attach_var_ratio(cfg, feats, rec, r)
            rows.append({"method": rec["method"], "n_reps": rec["n_reps"],
                         "acc_mean": rec["acc_mean"], "acc_std": rec["acc_std"],
                         "f1_mean": rec["f1_mean"], "f1_std": rec["f1_std"],
                         "distill_sec": rec["distill_sec"],
                         "cloud_sec": rec["total_sec"] - rec["distill_sec"],
                         "n_cloud_train": rec.get("n_cloud_train"),
                         "validation_mode": rec.get("validation_mode"),
                         "var_ratio": rec["var_ratio"]})
            if "moment_calibration" in rec:
                mc = rec["moment_calibration"]
                rows[-1].update({"mc_pre_mean_mse": mc.get("pre_mean_mse"),
                                 "mc_post_mean_mse": mc.get("post_mean_mse"),
                                 "mc_pre_var_mse": mc.get("pre_var_mse"),
                                 "mc_post_var_mse": mc.get("post_var_mse"),
                                 "mc_clamped_frac": mc.get("clamped_frac")})
    if bdm:
        by = {}
        for rec in bdm["rows"]:
            attach_var_ratio(cfg, feats, rec, r)
            by.setdefault(rec["method"], []).append(rec)
        for m, rs in by.items():
            acc = np.array([x["acc_mean"] for x in rs])
            f1 = np.array([x["f1_mean"] for x in rs])
            vr = np.array([x["var_ratio"] for x in rs], dtype=float)
            # 多蒸馏种子: ± 反映蒸馏种子波动; 单种子: ± 用云端种子波动
            acc_std = float(acc.std()) if len(rs) > 1 else float(rs[0]["acc_std"])
            f1_std = float(f1.std()) if len(rs) > 1 else float(rs[0]["f1_std"])
            rows.append({"method": m, "n_reps": rs[0]["n_reps"],
                         "dm_seeds": len(rs),
                         "acc_mean": float(acc.mean()), "acc_std": acc_std,
                         "f1_mean": float(f1.mean()), "f1_std": f1_std,
                         "distill_sec": float(np.mean([x["distill_sec"] for x in rs])),
                         "cloud_sec": float(np.mean([x["total_sec"] - x["distill_sec"] for x in rs])),
                         "n_cloud_train": rs[0].get("n_cloud_train"),
                         "validation_mode": rs[0].get("validation_mode"),
                         "var_ratio": float(np.nanmean(vr))})
    if full:
        for x in rows:
            x["acc_drop_rel"] = (full["acc_mean"] - x["acc_mean"]) / full["acc_mean"] * 100
    path = os.path.join(R, f"{ds}_main_r{r:g}_summary.csv")
    keys = ["method", "n_reps", "dm_seeds", "n_cloud_train", "validation_mode",
            "acc_mean", "acc_std", "f1_mean", "f1_std", "acc_drop_rel",
            "var_ratio", "distill_sec", "cloud_sec",
            "mc_pre_mean_mse", "mc_post_mean_mse", "mc_pre_var_mse",
            "mc_post_var_mse", "mc_clamped_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n## {ds.upper()} @ {r:g}:1  (protocol={cfg.get('protocol_version','legacy')})\n")
    print("| Method | M | Cloud train | Val | Top-1 | Macro-F1 | Rel. drop | Var ratio | Distill (s) | Cloud (s) |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for x in rows:
        drop = f"{x.get('acc_drop_rel', 0):.2f}%" if "acc_drop_rel" in x else "-"
        vr = x.get("var_ratio", float("nan"))
        vr = f"{vr:.3f}" if np.isfinite(vr) else "-"
        print(f"| {x['method']} | {x['n_reps']} | {x.get('n_cloud_train', '-')} | "
              f"{x.get('validation_mode') or '-'} | "
              f"{x['acc_mean']:.2f}±{x['acc_std']:.2f} | "
              f"{x['f1_mean']:.2f}±{x['f1_std']:.2f} | {drop} | {vr} | "
              f"{x['distill_sec']:.1f} | {x['cloud_sec']:.1f} |")
    mc_rows = [x for x in rows if "mc_pre_var_mse" in x]
    if mc_rows:
        print("\n### Moment Calibration\n")
        print("| Method | mean MSE before | after | var MSE before | after | clamped |")
        print("|---|---:|---:|---:|---:|---:|")
        for x in mc_rows:
            print(f"| {x['method']} | {x['mc_pre_mean_mse']:.3e} | "
                  f"{x['mc_post_mean_mse']:.3e} | {x['mc_pre_var_mse']:.3e} | "
                  f"{x['mc_post_var_mse']:.3e} | {x['mc_clamped_frac']:.3f} |")
    qs = sorted(glob.glob(os.path.join(R, f"{ds}_*_r{r:g}_quant.json")))
    if qs:
        full_bytes = 49000 * 1024 * 4 + 49000 * 2
        print("\n### 量化与实际上传负载\n")
        print("| Method | Mode | Top-1 | Macro-F1 | Actual bytes | vs FP32 Full |")
        print("|---|---|---:|---:|---:|---:|")
        for qp in qs:
            q = load(qp)
            name = os.path.basename(qp).replace(f"{ds}_", "").replace(f"_r{r:g}_quant.json", "")
            for m, v in q["results"].items():
                total = v.get("payload", {}).get("total", "-")
                ratio = (f"{full_bytes / total:.1f}×" if isinstance(total, int)
                         else "-")
                print(f"| {name} | {m} | {v['acc_mean']:.2f}±{v['acc_std']:.2f} | "
                      f"{v['f1_mean']:.2f}±{v['f1_std']:.2f} | {total} | {ratio} |")
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
