"""冻结主干 + 训练投影层与分类头"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEV, macro_f1, per_class_acc, top1
from models import build_head, build_projection


def channel_stats(raw_tr, batch=512):
    """逐通道均值/标准差 (流式计算, 不接触标签)"""
    c = raw_tr.size(1)
    s = torch.zeros(c, dtype=torch.float64)
    ss = torch.zeros(c, dtype=torch.float64)
    n = 0
    for i in range(0, raw_tr.size(0), batch):
        f = raw_tr[i:i + batch].float()
        s += f.sum(dim=(0, 2, 3)).double()
        ss += f.pow(2).sum(dim=(0, 2, 3)).double()
        n += f.size(0) * f.size(2) * f.size(3)
    m = s / n
    v = (ss / n - m.pow(2)).clamp_min(1e-12)
    return (m.float().view(1, -1, 1, 1).to(DEV),
            v.sqrt().float().view(1, -1, 1, 1).to(DEV))


@torch.no_grad()
def evaluate(net, raw, y, mean=None, std=None, norm=True, batch=512, num_classes=None):
    net.eval()
    preds = []
    for i in range(0, raw.size(0), batch):
        f = raw[i:i + batch].to(DEV).float()
        if norm and mean is not None:
            f = (f - mean) / std
        preds.append(net(f).argmax(1).cpu())
    pred = torch.cat(preds)
    n_cls = num_classes or int(max(y.max().item(), pred.max().item())) + 1
    return {"acc": (pred == y).float().mean().item() * 100,
            "f1": macro_f1(pred, y, n_cls),
            "per_class": per_class_acc(pred, y, n_cls), "pred": pred}


def fit(raw_tr, y_tr, raw_va, y_va, mode, in_ch, head_type, seed,
        epochs=30, bs=256, lr=1e-3, wd=1e-4, norm=True, verbose=True):
    """训练 投影 + 分类头; 主干不在计算图内"""
    torch.manual_seed(seed)
    proj, in_dim = build_projection(mode, in_ch)
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    head = build_head(head_type, in_dim, num_classes)
    net = nn.Sequential(proj, head).to(DEV)

    mean, std = (channel_stats(raw_tr) if norm else (None, None))
    params = list(proj.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps_per_ep = int(np.ceil(raw_tr.size(0) / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_ep)

    t0 = time.time()
    n_b = 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(raw_tr.size(0))
        run = 0.0
        n_b = 0
        for i in range(0, raw_tr.size(0), bs):
            idx = perm[i:i + bs]
            f = raw_tr[idx].to(DEV).float()
            if norm:
                f = (f - mean) / std
            loss = F.cross_entropy(net(f), y_tr[idx].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            run += loss.item()
            n_b += 1
        if verbose and (ep + 1) % 10 == 0:
            v = evaluate(net, raw_va, y_va, mean, std, norm)
            print(f"    ep {ep+1}/{epochs} loss {run/max(1,n_b):.3f} "
                  f"val {v['acc']:.2f}% f1 {v['f1']:.2f}", flush=True)
    tr_sec = time.time() - t0
    v = evaluate(net, raw_va, y_va, mean, std, norm)
    return net, {"val_acc": v["acc"], "val_f1": v["f1"], "train_sec": tr_sec,
                 "mean": mean, "std": std}


def n_params(module):
    return sum(p.numel() for p in module.parameters())
