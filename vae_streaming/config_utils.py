"""vae_streaming 配置与路径"""
import os
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent

_PATH_KEYS = ("vae_ckpt", "results_dir", "figures_dir", "checkpoints_dir")


def resolve(path):
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def load_config(path):
    with open(resolve(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = resolve(path)
    for k in _PATH_KEYS:
        if k in cfg:
            cfg[k] = resolve(cfg[k])
    return cfg


def ensure_dirs(cfg):
    for k in ("results_dir", "figures_dir", "checkpoints_dir"):
        if k in cfg:
            os.makedirs(cfg[k], exist_ok=True)


def result_path(cfg, name):
    os.makedirs(cfg["results_dir"], exist_ok=True)
    return os.path.join(cfg["results_dir"], name)


def figure_path(cfg, name):
    os.makedirs(cfg["figures_dir"], exist_ok=True)
    return os.path.join(cfg["figures_dir"], name)


def checkpoint_path(cfg, name):
    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    p = os.path.join(cfg["checkpoints_dir"], name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
