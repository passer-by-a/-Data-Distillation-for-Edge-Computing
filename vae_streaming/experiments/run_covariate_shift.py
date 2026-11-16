"""Covariate Shift 数据流实验入口"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vae_streaming.experiments.cli import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli("covariate_shift", ["online_replay_align_dm"])
