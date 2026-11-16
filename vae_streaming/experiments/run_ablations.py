"""消融实验入口: 读取 configs/ablation.yaml 的 ablations 列表并逐个运行"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vae_streaming.config_utils import load_config  # noqa: E402
from vae_streaming.experiments.engine import deep_merge, run_method  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vae_streaming/configs/ablation.yaml")
    ap.add_argument("--names", default=None, help="逗号分隔的消融名；默认全部")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = list(cfg["ablations"].keys()) if args.names is None \
        else [s.strip() for s in args.names.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for name in names:
        run_cfg = {k: v for k, v in cfg.items() if k != "ablations"}
        run_cfg = deep_merge(run_cfg, cfg["ablations"][name])
        run_cfg["_run_name"] = name
        if args.rounds is not None:
            run_cfg["stream"]["num_rounds"] = args.rounds
        for seed in seeds:
            run_method(run_cfg, None, seed=seed, verbose=True)
