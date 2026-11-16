"""脚本公共入口"""
import os
import sys

import torch

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PKG)
for p in (ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from vae_bdm.config_utils import ensure_dirs, load_config  # noqa: E402
from vae_bdm.features import load_features  # noqa: E402


def load_cfg(name):
    cfg = load_config(os.path.join(PKG, "configs", name))
    ensure_dirs(cfg)
    return cfg


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
