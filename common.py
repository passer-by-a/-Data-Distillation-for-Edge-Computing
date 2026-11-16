"""共享工具: 路径、随机种子、指标、成本测量"""

import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = {
    "cifar10": {"ddpm": os.path.join(ROOT, "checkpoints", "ddpm_cifar10.pt"),
                "vae": os.path.join(ROOT, "checkpoints", "autoencoder_edge_s8_c16.pt"),
                "vae_unsup": os.path.join(ROOT, "checkpoints",
                                          "autoencoder_edge_s8_c16_noclspart.pt")},
    "cifar100": {"ddpm": os.path.join(ROOT, "checkpoints", "cifar100", "ddpm_cifar100.pt"),
                 "vae": os.path.join(ROOT, "checkpoints", "cifar100",
                                     "autoencoder_edge_s8_c16.pt"),
                 "vae_unsup": os.path.join(ROOT, "checkpoints", "cifar100",
                                           "autoencoder_edge_s8_c16_noclspart.pt")},
}


def resolve_ckpt(dataset, kind):
    p = CKPT[dataset][kind]
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 {dataset} 的 {kind} 权重: {p}")
    return p


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100


def macro_f1(pred, y, num_classes):
    from sklearn.metrics import f1_score
    return f1_score(y.cpu().numpy(), pred.cpu().numpy(),
                    average="macro", labels=list(range(num_classes)),
                    zero_division=0) * 100


def per_class_acc(pred, y, num_classes):
    out = []
    for c in range(num_classes):
        m = y == c
        out.append((pred[m] == y[m]).float().mean().item() * 100 if m.any() else float("nan"))
    return out


@torch.no_grad()
def count_flops(fn):
    """返回单次调用的 MFLOPs"""
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return fc.get_total_flops() / 1e6


@torch.no_grad()
def latency_ms(fn, warmup=10, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def append_csv(path, rows, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header or list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
