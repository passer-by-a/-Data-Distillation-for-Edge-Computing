"""Step 1: 提取 VAE mu 特征并缓存为 npy"""
import argparse

from vae_bdm.scripts._common import load_cfg
from vae_bdm.features import extract_and_cache, load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cifar10.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    extract_and_cache(cfg, force=a.force)
    f = load_features(cfg)
    for s in ("train", "val", "test"):
        print(f"{s}: {tuple(f[s]['x'].shape)} labels {tuple(f[s]['y'].shape)}")


if __name__ == "__main__":
    main()
