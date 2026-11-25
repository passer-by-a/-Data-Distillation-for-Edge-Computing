"""Step 6: FP32/FP16/INT8 量化与真实 payload"""
import argparse
import os

from vae_bdm.scripts._common import load_cfg
from vae_bdm.features import load_features
from vae_bdm.evaluate import eval_quantized, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--method", default="bdm")
    ap.add_argument("--variant", default="random_feature")
    ap.add_argument("--modes", default="fp32,fp16,int8")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    feats = load_features(cfg)
    modes = [m for m in a.modes.split(",") if m]
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = os.path.join(cfg["results_dir"], "payload")
    tag = a.method if not a.variant else f"{a.method}_{a.variant}"
    res = eval_quantized(cfg, feats, a.method, a.ratio, variant=a.variant or None,
                         modes=modes, seeds=seeds, out_dir=out_dir, tag=tag)
    save_json({"method": a.method, "variant": a.variant, "ratio": a.ratio,
               "results": res},
              os.path.join(cfg["results_dir"],
                           f"{cfg['dataset']}_{tag}_r{a.ratio:g}_quant.json"))
    print("\n量化     Acc          F1       实际字节")
    for m, v in res.items():
        print(f"{m:<8} {v['acc_mean']:6.2f}±{v['acc_std']:.2f}  "
              f"{v['f1_mean']:6.2f}±{v['f1_std']:.2f}  "
              f"{v.get('payload', {}).get('total', '-')}")


if __name__ == "__main__":
    main()
