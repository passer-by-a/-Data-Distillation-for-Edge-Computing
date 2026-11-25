"""汇总 rerun80 的五方法主表（CIFAR-10 / CIFAR-100）

用法：
  python -m vae_streaming.experiments.summarize_rerun80
"""
import argparse
import json

import numpy as np

from ..config_utils import load_config, result_path

METHODS = (("dm", "Plain DM"), ("dm_multi", "Multi-MLP DM"),
           ("online_kmeans", "K-means"), ("online_moment", "Moment"),
           ("online_reservoir", "Reservoir"))


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{np.mean(xs):.2f}±{np.std(xs, ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rerun80")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--datasets", default="cifar10,cifar100")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = load_config("vae_streaming/configs/cifar10_supervised_multi80_u2.yaml")
    out = {}
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"\n===== {dataset} =====")
        print(f"{'method':>13} {'r0':>14} {'final':>14} {'online':>14} "
              f"{'final_F1':>14} {'INT8':>14} {'var_ratio':>10}")
        out[dataset] = {}
        for key, name in METHODS:
            r0, fin, onl, f1, i8, vr = [], [], [], [], [], []
            for seed in seeds:
                p = result_path(
                    cfg, f"{dataset}/iid/{key}_{args.tag}_seed{seed}.json")
                try:
                    d = json.load(open(p, "r", encoding="utf-8"))
                except FileNotFoundError:
                    continue
                rs = d["rounds"]
                stream = [r for r in rs if int(r["round"]) >= 0]
                r0.append(float(rs[0]["acc"]))
                fin.append(float(rs[-1]["acc"]))
                onl.append(float(np.mean([r["acc"] for r in stream])))
                f1.append(float(rs[-1]["f1"]))
                i8.append(float(d["summary"]["final_test_int8_acc"]))
                vr.append(float(d["summary"]["final_group_var_ratio"]))
            out[dataset][key] = {
                "name": name, "n_seeds": len(fin),
                "r0": r0, "final": fin, "online": onl, "final_f1": f1,
                "int8": i8, "var_ratio": vr}
            print(f"{name:>13} {_ms(r0):>14} {_ms(fin):>14} {_ms(onl):>14} "
                  f"{_ms(f1):>14} {_ms(i8):>14} {_ms(vr):>10}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
