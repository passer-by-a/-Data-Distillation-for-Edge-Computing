"""实验脚本共用 CLI"""
import argparse

from ..config_utils import load_config
from .engine import run_method


def build_parser(default_scenario, default_methods):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vae_streaming/configs/cifar100.yaml")
    ap.add_argument("--scenario", default=default_scenario)
    ap.add_argument("--methods", default=",".join(default_methods))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--initial-fraction", type=float, default=None)
    ap.add_argument("--initial-epochs", type=int, default=None)
    ap.add_argument("--vae-steps", type=int, default=None)
    ap.add_argument("--cloud-steps", type=int, default=None)
    ap.add_argument("--patch-fraction", type=float, default=None)
    ap.add_argument("--quant", default=None, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--sync", default=None, choices=["full_sync", "patch"])
    ap.add_argument("--head", default=None, choices=["linear", "mlp"])
    ap.add_argument("--tag", default=None,
                    help="附加到方法名后的标签，避免不同设置覆盖同一结果文件")
    ap.add_argument("--resume", action="store_true",
                    help="从 stream_state.pt 断点续跑")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="完成指定轮后保存断点并停止")
    ap.add_argument("--quiet", action="store_true")
    return ap


def run_cli(default_scenario, default_methods):
    args = build_parser(default_scenario, default_methods).parse_args()
    cfg = load_config(args.config)
    cfg["stream"]["scenario"] = args.scenario
    if args.rounds is not None:
        cfg["stream"]["num_rounds"] = args.rounds
    if args.initial_fraction is not None:
        cfg["stream"]["initial_fraction"] = args.initial_fraction
    if args.budget is not None:
        cfg["memory"]["representative_budget"] = args.budget
    if args.initial_epochs is not None:
        cfg["vae"]["initial_epochs"] = args.initial_epochs
    if args.vae_steps is not None:
        cfg["vae"]["steps_per_round"] = args.vae_steps
    if args.cloud_steps is not None:
        cfg["cloud"]["steps_per_round"] = args.cloud_steps
    if args.patch_fraction is not None:
        cfg["communication"]["patch_fraction"] = args.patch_fraction
    if args.quant is not None:
        cfg["communication"]["quantization"] = args.quant
    if args.sync is not None:
        cfg["communication"]["mode"] = args.sync
    if args.head is not None:
        cfg["cloud"]["head"] = args.head
    if args.tag:
        cfg["_run_tag"] = args.tag

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    for seed in seeds:
        for method in methods:
            results.append(run_method(cfg, method, seed=seed,
                                      verbose=not args.quiet,
                                      resume=args.resume,
                                      stop_after=args.stop_after))
    return results
