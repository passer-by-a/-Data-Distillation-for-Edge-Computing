"""训练/评估共用：加载 VAE、特征归一化、分组"""
import os

import numpy as np
import torch

from data import load_split
from vae import EdgeVAE

from group_vae.data.group_sampler import build_groups


def load_vae(cfg, device="cuda"):
    vae = EdgeVAE(latent_dim=16, num_classes=cfg["num_classes"]).to(device)
    ck = torch.load(cfg["vae_ckpt"], map_location=device, weights_only=True)
    vae.load_state_dict(ck["model_state_dict"])
    vae.eval()
    return vae


def load_norm(cfg):
    p = os.path.join(cfg["cache_dir"], cfg["dataset"], "norm_stats.pt")
    st = torch.load(p, map_location="cpu")
    return st["mean"].float(), st["std"].float()


def build_train_groups(cfg, ratio=10.0, group_size=None, strategy="random",
                       seed=42):
    d = load_split(cfg["dataset"], val_ratio=cfg["val_ratio"],
                   seed=cfg["split_seed"])
    labels = d["train"][1].numpy()
    m = int(round(len(labels) / float(ratio)))
    k = int(group_size or cfg["group"]["size"])
    feats = None
    if strategy in ("nearest", "diverse"):
        p = os.path.join(cfg["cache_dir"], cfg["dataset"], "train_features.npy")
        feats = np.load(p)
    groups = build_groups(labels, m, k, strategy=strategy, seed=seed,
                          feats=feats, num_classes=cfg["num_classes"])
    return d, groups


def subset_groups(groups, sel):
    return {"indices": groups["indices"][sel], "mask": groups["mask"][sel],
            "labels": groups["labels"][sel], "quotas": groups["quotas"],
            "group_size": groups["group_size"],
            "strategy": groups["strategy"]}


def freeze_vae(model):
    for p in model.vae.parameters():
        p.requires_grad = False
    model.vae.eval()


def group_params(model):
    return [p for n, p in model.named_parameters()
            if not n.startswith("vae.") and p.requires_grad]
