"""三类数据流场景顺序执行"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vae_streaming.config_utils import load_config  # noqa: E402
from vae_streaming.experiments.engine import run_method  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vae_streaming/configs/cifar100.yaml")
    ap.add_argument("--scenarios", default="iid,label_shift,covariate_shift")
    ap.add_argument("--methods",
                    default="online_kmeans,online_moment,dm")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.rounds is not None:
        cfg["stream"]["num_rounds"] = args.rounds
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for scenario in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        cfg["stream"]["scenario"] = scenario
        for seed in seeds:
            for m in methods:
                run_method(cfg, m, seed=seed, verbose=True)
