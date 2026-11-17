"""在当前 Group-VAE 编码器特征空间中重跑 K-means/Herding/BDM 基线"""
import argparse
import json
import os

import numpy as np
import torch

from group_vae.config_utils import ensure_dirs, load_config, result_path
from vae_bdm.config_utils import load_config as load_v2
from vae_bdm.evaluate import run_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--tag", required=True, help="encode_groups 的 tag")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--methods", default="kmeans,herding")
    ap.add_argument("--bdm_variant", default="mean_var")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bdm_seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ensure_dirs(cfg)
    R = cfg["results_dir"]
    tr = np.load(os.path.join(R, a.tag + "_train.npz"))
    va = np.load(os.path.join(R, a.tag + "_val.npz"))
    te = np.load(os.path.join(R, a.tag + "_test.npz"))
    g = np.load(os.path.join(R, a.tag + ".npz"), allow_pickle=True)
    meta = json.loads(str(g["meta"]))
    feats = {
        "train": {"x": torch.from_numpy(tr["features"]).float(),
                  "y": torch.from_numpy(tr["labels"]).long()},
        "val": {"x": torch.from_numpy(va["features"]).float(),
                "y": torch.from_numpy(va["labels"]).long()},
        "test": {"x": torch.from_numpy(te["features"]).float(),
                 "y": torch.from_numpy(te["labels"]).long()},
        "meta": {"vae_ckpt": meta.get("vae_ckpt"),
                 "feature_norm": "per-channel z-score (shared)",
                 "protocol": meta.get("protocol_version")},
    }
    v2 = load_v2(f"vae_bdm/configs/{cfg['dataset']}_v2.yaml")
    v2["results_dir"] = os.path.join(R, "baselines")
    v2["figures_dir"] = os.path.join(R, "figures")
    v2["protocol_version"] = cfg.get("protocol_version", "group_vae_v1")
    os.makedirs(v2["results_dir"], exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for m in [x for x in a.methods.split(",") if x]:
        if m == "bdm":
            dcfg = json.loads(json.dumps(v2))
            dcfg["dm"]["seed"] = a.bdm_seed
            rec = run_method(dcfg, feats, "bdm", a.ratio,
                             variant=a.bdm_variant, seeds=seeds,
                             save_reps=False, verbose=True)
        else:
            rec = run_method(v2, feats, m, a.ratio, seeds=seeds,
                             save_reps=False, verbose=True)
        rows.append(rec)
    p = os.path.join(v2["results_dir"], f"baselines_r{a.ratio:g}.json")
    with open(p, "w") as f:
        json.dump({"rows": rows, "tag": a.tag, "meta": meta}, f,
                  indent=2, ensure_ascii=False)
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
