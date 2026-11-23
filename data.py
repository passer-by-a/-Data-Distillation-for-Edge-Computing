"""CIFAR 数据加载: 固定划分

划分与主项目 config.get_cifar10_loaders 完全一致:
  np.random.RandomState(42).permutation(50000), 前 2% 作验证集, 其余作训练集。
图像以 uint8 保存, 使用时转 [-1,1]。
"""

import os

import numpy as np
import torch
from torchvision import datasets, transforms

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
