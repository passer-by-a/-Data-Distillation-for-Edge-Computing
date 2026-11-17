"""P1: 固定 Epoch vs 固定优化步数 的训练预算消融"""
import argparse
import json
import os

from vae_bdm.scripts._common import ROOT, load_cfg
from vae_bdm.features import load_features
from vae_bdm.evaluate import run_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10_v2.yaml")
    ap.add_argument("--ratios", default="50,100")
    ap.add_argument("--methods", default="kmeans,bdm")
    ap.add_argument("--variant", default="random_feature")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    feats = load_features(cfg)
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for r in [float(x) for x in a.ratios.split(",") if x]:
        for m in [x for x in a.methods.split(",") if x]:
            for mode, steps in (("fixed_epochs", None),
                                ("fixed_steps", a.steps)):
                rec = run_method(cfg, feats, m, r,
                                 variant=a.variant if m == "bdm" else None,
                                 seeds=seeds, save_reps=False, verbose=True,
                                 budget_mode=mode, budget_steps=steps)
                rec["budget_label"] = (f"fixed_epochs:{a.epochs}" if steps is None
                                       else f"fixed_steps:{steps}")
                rows.append(rec)
    out_dir = os.path.join(ROOT, "vae_bdm", "results", "v2", "ablations")
    os.makedirs(out_dir, exist_ok=True)
    tag = a.tag or f"{cfg['dataset']}_budget"
    p = os.path.join(out_dir, tag + ".json")
    with open(p, "w") as f:
        json.dump({"protocol_version": cfg.get("protocol_version"),
                   "steps": a.steps, "epochs": a.epochs, "rows": rows},
                  f, indent=2, ensure_ascii=False)
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
