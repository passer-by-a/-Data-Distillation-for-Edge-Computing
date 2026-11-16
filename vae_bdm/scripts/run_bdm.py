"""Step 3-4: Batch DM 及统计量消融"""
import argparse
import csv
import json
import os

from vae_bdm.scripts._common import load_cfg
from vae_bdm.features import load_features
from vae_bdm.evaluate import VARIANTS, run_method, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--variants", default="mean,mean_var,full,anchor,diversity,cov,all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--save_seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    feats = load_features(cfg)
    seeds = [int(s) for s in a.seeds.split(",")]
    variants = [v for v in a.variants.split(",") if v]
    tag = f"{cfg['dataset']}_bdm_r{a.ratio:g}"
    rows = []
    if os.path.exists(os.path.join(cfg["results_dir"], tag + ".json")):
        old = json.load(open(os.path.join(cfg["results_dir"], tag + ".json")))
        rows = old.get("rows", [])
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(v)
        for s in seeds:
            rows = [r for r in rows if not (r["method"] == f"bdm_{v}" and
                                            r.get("dm_seed") == s)]
            dcfg = json.loads(json.dumps(cfg))
            dcfg["dm"]["seed"] = s
            rec = run_method(dcfg, feats, "bdm", a.ratio, variant=v,
                             seeds=[0, 1, 2], save_reps=(s == a.save_seed),
                             verbose=True)
            rec["dm_seed"] = s
            rows.append(rec)
            print(f"  -> [{v} seed={s}] {rec['acc_mean']:.2f}%", flush=True)
    keys = ["dataset", "method", "ratio", "dm_seed", "n_reps", "acc_mean",
            "acc_std", "f1_mean", "f1_std", "best_val_acc", "best_step",
            "distill_sec", "peak_gpu_gb"]
    with open(os.path.join(cfg["results_dir"], tag + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    save_json({"rows": rows, "ratio": a.ratio},
              os.path.join(cfg["results_dir"], tag + ".json"))


if __name__ == "__main__":
    main()
