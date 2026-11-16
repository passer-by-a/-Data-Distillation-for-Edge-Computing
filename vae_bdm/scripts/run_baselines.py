"""Step 2: Full/Random/Mean/K-means/Herding 基线"""
import argparse
import csv
import json
import os

from vae_bdm.scripts._common import load_cfg
from vae_bdm.features import load_features
from vae_bdm.evaluate import run_method, save_json

METHODS = ["full", "random", "mean", "kmeans", "herding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    feats = load_features(cfg)
    seeds = [int(s) for s in a.seeds.split(",")]
    methods = [m for m in a.methods.split(",") if m]
    rows = []
    for m in methods:
        rows.append(run_method(cfg, feats, m, a.ratio, seeds=seeds,
                               save_reps=(m != "full"), verbose=True))
    tag = f"{cfg['dataset']}_baselines_r{a.ratio:g}"
    path = os.path.join(cfg["results_dir"], tag + ".csv")
    keys = ["dataset", "method", "ratio", "n_train", "n_reps", "actual_ratio",
            "acc_mean", "acc_std", "f1_mean", "f1_std", "distill_sec",
            "total_sec"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    save_json({"rows": rows, "ratio": a.ratio},
              os.path.join(cfg["results_dir"], tag + ".json"))


if __name__ == "__main__":
    main()
