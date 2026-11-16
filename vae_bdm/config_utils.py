"""VAE-BDM 配置与路径"""
import os
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent


def resolve(path):
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def load_config(path):
    with open(resolve(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = resolve(path)
    for k in ("vae_ckpt", "cache_dir", "results_dir", "figures_dir"):
        if k in cfg:
            cfg[k] = resolve(cfg[k])
    return cfg


def ensure_dirs(cfg):
    for k in ("cache_dir", "results_dir", "figures_dir"):
        os.makedirs(cfg[k], exist_ok=True)


def cache_dir(cfg):
    p = os.path.join(cfg["cache_dir"], cfg["dataset"])
    os.makedirs(p, exist_ok=True)
    return p


def result_path(cfg, name):
    os.makedirs(cfg["results_dir"], exist_ok=True)
    return os.path.join(cfg["results_dir"], name)


def figure_path(cfg, name):
    os.makedirs(cfg["figures_dir"], exist_ok=True)
    return os.path.join(cfg["figures_dir"], name)
