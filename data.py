"""CIFAR 数据加载: 固定划分 + 固定噪声 + 特征提取

划分与主项目 config.get_cifar10_loaders 完全一致:
  np.random.RandomState(42).permutation(50000), 前 2% 作验证集, 其余作训练集。
图像以 uint8 保存, 使用时转 [-1,1]。
"""

import os

import numpy as np
import torch
from torchvision import datasets, transforms

from common import DEV

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CLS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


def _to_tensor_u8(ds):
    xs = torch.empty(len(ds), 3, 32, 32, dtype=torch.uint8)
    ys = torch.empty(len(ds), dtype=torch.long)
    for i, (x, y) in enumerate(ds):
        xs[i] = (x * 255).round().byte() if x.dtype.is_floating_point else x
        ys[i] = y
    return xs, ys


def load_split(name="cifar10", val_ratio=0.02, seed=42):
    """返回 {train, val, test} 的 (images uint8, labels)"""
    cls = _CLS[name]
    tf = transforms.ToTensor()
    tr = cls(root=DATA_ROOT, train=True, download=True, transform=tf)
    te = cls(root=DATA_ROOT, train=False, download=True, transform=tf)
    x_tr, y_tr = _to_tensor_u8(tr)
    x_te, y_te = _to_tensor_u8(te)
    n = len(tr)
    perm = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_ratio)
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])
    return {"train": (x_tr[tr_idx], y_tr[tr_idx]),
            "val": (x_tr[val_idx], y_tr[val_idx]),
            "test": (x_te, y_te)}


def make_noise(n, seed=1234):
    """固定噪声实现, 所有方法共用, 保证可复现"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=g).half()


@torch.no_grad()
def extract(backbone, images_u8, labels, mode, t=0, noise=None,
            noise_mode="schedule", batch=256, layer=None, verbose=False):
    """返回 CPU float32 特征 (展平) 与标签"""
    from models import LAYERS
    layer = layer or LAYERS.get(mode, "middle")
    sched = getattr(backbone, "scheduler", None)
    feats = []
    for i in range(0, images_u8.size(0), batch):
        x0 = images_u8[i:i + batch].to(DEV).float().div(127.5).sub(1)
        if mode in ("VAE", "VAE_unsup"):
            f = backbone.forward_features(x0)
        else:
            if noise_mode == "clean":
                x_t = x0
            else:
                nb = noise[i:i + batch].to(DEV).float()
                abar = sched.alphas_cumprod[int(t)]
                x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * nb
            tt = torch.full((x_t.size(0),), int(t), device=DEV, dtype=torch.long)
            f = backbone.forward_features(x_t, tt, layer=layer)
        feats.append(f.float().cpu())
        if verbose and (i // batch) % 50 == 0:
            print(f"    {i + x0.size(0)}/{images_u8.size(0)}", flush=True)
    return torch.cat(feats, 0), labels
