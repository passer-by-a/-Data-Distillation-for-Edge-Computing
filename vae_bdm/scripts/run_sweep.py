"""Step 5: 压缩率扫描"""
import argparse
import csv
import json
import os

from vae_bdm.scripts._common import load_cfg
from vae_bdm.features import load_features
from vae_bdm.evaluate import run_method, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--ratios", default="10,20,50,100")
    ap.add_argument("--methods", default="random,kmeans,kmeans_moment")
    ap.add_argument("--bdm_variant", default="random_feature")
    ap.add_argument("--dm_seed", type=int, default=42)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    feats = load_features(cfg)
    seeds = [int(s) for s in a.seeds.split(",")]
    ratios = [float(r) for r in a.ratios.split(",") if r]
    methods = [m for m in a.methods.split(",") if m]

    full = run_method(cfg, feats, "full", 1, seeds=seeds, save_reps=False,
                      verbose=False)
    rows = []
    for r in ratios:
        for m in methods:
            rows.append(run_method(cfg, feats, m, r, seeds=seeds,
                                   save_reps=True, verbose=True))
        dcfg = json.loads(json.dumps(cfg))
        dcfg["dm"]["seed"] = a.dm_seed
        rows.append(run_method(dcfg, feats, "bdm", r, variant=a.bdm_variant,
                               seeds=seeds, save_reps=True, verbose=True))
        rec = dict(full); rec["ratio"] = r
        rows.append(rec)

    tag = f"{cfg['dataset']}_sweep"
    keys = ["dataset", "method", "ratio", "n_train", "n_reps", "actual_ratio",
            "acc_mean", "acc_std", "f1_mean", "f1_std", "distill_sec",
            "total_sec", "best_val_acc", "best_step"]
    with open(os.path.join(cfg["results_dir"], tag + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    save_json({"rows": rows}, os.path.join(cfg["results_dir"], tag + ".json"))
    print(f"扫描完成: results/{tag}.csv")


if __name__ == "__main__":
    main()
