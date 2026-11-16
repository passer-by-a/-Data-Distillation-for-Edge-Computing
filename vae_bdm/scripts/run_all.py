"""一键跑第一阶段: 特征 → 基线 → BDM → 扫描 → 量化 → 图表"""
import argparse
import os
import subprocess
import sys

from vae_bdm.scripts._common import ROOT


def run(args):
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m"] + args, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--variants", default="mean,mean_var,full,anchor,diversity,cov,all")
    a = ap.parse_args()
    pkg = "vae_bdm.scripts"
    run([f"{pkg}.extract_features", "--config", a.config])
    run([f"{pkg}.run_baselines", "--config", a.config, "--ratio", str(a.ratio)])
    run([f"{pkg}.run_bdm", "--config", a.config, "--ratio", str(a.ratio),
         "--variants", a.variants])
    run([f"{pkg}.run_sweep", "--config", a.config])
    run([f"{pkg}.run_quant", "--config", a.config, "--ratio", str(a.ratio)])
    run([f"{pkg}.summarize", "--config", a.config, "--ratio", str(a.ratio)])
    run([f"{pkg}.make_figures", "--config", a.config])


if __name__ == "__main__":
    main()
