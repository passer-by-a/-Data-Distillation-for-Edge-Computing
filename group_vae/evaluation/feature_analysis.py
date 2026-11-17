"""组级与单图特征分布差异分析"""
import argparse
import json
import os

import numpy as np
import torch

from group_vae.config_utils import load_config, result_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    R = cfg["results_dir"]
    g = np.load(os.path.join(R, a.tag + ".npz"), allow_pickle=True)
    s = np.load(os.path.join(R, a.tag + "_train.npz"))
    gf = torch.from_numpy(g["features"]).float()
    gy = torch.from_numpy(g["labels"]).long()
    sf = torch.from_numpy(s["features"]).float()
    sy = torch.from_numpy(s["labels"]).long()
    out = {"per_class": {}, "global": {}}
    vr, md, nr = [], [], []
    for c in range(cfg["num_classes"]):
        gm = gf[gy == c]; sm = sf[sy == c]
        if len(gm) < 2:
            continue
        mu_g, mu_s = gm.mean(0), sm.mean(0)
        v_g, v_s = gm.var(0, unbiased=False), sm.var(0, unbiased=False)
        mean_dist = float((mu_g - mu_s).pow(2).mean().sqrt())
        var_ratio = float(v_g.mean() / v_s.mean().clamp_min(1e-8))
        norm_ratio = float(gm.norm(dim=1).mean() / sm.norm(dim=1).mean().clamp_min(1e-8))
        # 组内多样性：组特征两两距离均值
        idx = torch.randperm(len(gm))[:min(512, len(gm))]
        z = gm[idx]
        div = float(torch.cdist(z, z).mean())
        out["per_class"][str(c)] = {"mean_dist": mean_dist,
                                    "var_ratio": var_ratio,
                                    "norm_ratio": norm_ratio,
                                    "group_diversity": div}
        vr.append(var_ratio); md.append(mean_dist); nr.append(norm_ratio)
    out["global"] = {"mean_dist": float(np.mean(md)),
                     "var_ratio": float(np.mean(vr)),
                     "norm_ratio": float(np.mean(nr)),
                     "n_groups": int(len(gf)), "n_singles": int(len(sf))}
    p = result_path(cfg, a.tag + "_dist.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["global"], indent=2, ensure_ascii=False))
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
