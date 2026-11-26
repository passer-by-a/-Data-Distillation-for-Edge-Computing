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
    ap.add_argument("--class-quota", type=int, default=None,
                    help="IID 下每类每轮的固定样本数（base 自动吸收类别差异）")
    ap.add_argument("--group-rounds", type=int, default=None,
                    help="先按该轮数构造流式数据，再等分到 --rounds 轮"
                         "（保证数据与顺序不变，只改粒度）")
    ap.add_argument("--update-every", type=int, default=None,
                    help="每累积多少轮做一次在线 VAE 更新（默认 1）")
    ap.add_argument("--history-mode", default=None, choices=["full", "replay"],
                    help="历史预算口径：full=累计全部历史；replay=固定预算 "
                         "replay+current（消融对齐用）")
    ap.add_argument("--label-mode", default=None,
                    choices=["supervised_stream", "unlabeled_after_base"],
                    help="标签协议：supervised_stream（原始全监督）或 "
                         "unlabeled_after_base（仅 Base 有真实标签）")
    ap.add_argument("--pseudo-threshold", type=float, default=None,
                    help="固定 Teacher 伪标签置信度阈值（仅无标签协议）")
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
    ap.add_argument("--kip-transfer-diag", action="store_true",
                    help="KIP 只读诊断：记录编码器更新前后同一批验证图的 "
                         "A/B/C 核分类，不改变训练行为")
    ap.add_argument("--kip-cross-weight", type=float, default=None,
                    help="kip_cross_version 的跨版本项权重（默认取预设 1.0）")
    ap.add_argument("--representation", default=None, choices=["vae", "pixel"],
                    help="特征表示：vae=潜空间（默认）；pixel=Base 标准化的"
                         "原始像素（3072 维）")
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
    if args.class_quota is not None:
        cfg["stream"]["class_quota"] = args.class_quota
    if args.group_rounds is not None:
        cfg["stream"]["group_rounds"] = args.group_rounds
    if args.update_every is not None:
        cfg["vae"]["update_every_rounds"] = args.update_every
    if args.history_mode is not None:
        cfg["stream"]["history_mode"] = args.history_mode
    if args.label_mode is not None:
        cfg["stream"]["label_mode"] = args.label_mode
    if args.pseudo_threshold is not None:
        cfg.setdefault("pseudo_labels", {})["threshold"] = args.pseudo_threshold
    if args.budget is not None:
        cfg["memory"]["representative_budget"] = args.budget
    if args.initial_epochs is not None:
        cfg["vae"]["initial_epochs"] = args.initial_epochs
    if args.vae_steps is not None:
        cfg["vae"]["steps_per_round"] = args.vae_steps
    if args.cloud_steps is not None:
        # epochs_per_round > 0 时云端步数按 epoch 推导，会覆盖 steps_per_round；
        # 显式指定 --cloud-steps 时清除它，避免参数静默失效
        cfg["cloud"]["steps_per_round"] = args.cloud_steps
        cfg["cloud"]["epochs_per_round"] = 0
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
    if args.kip_transfer_diag:
        cfg.setdefault("diagnostics", {})["kip_transfer"] = True
    if args.kip_cross_weight is not None:
        cfg.setdefault("distillation", {})["kip_cross_weight"] = float(
            args.kip_cross_weight)
    if args.representation is not None:
        cfg["representation"] = args.representation

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
