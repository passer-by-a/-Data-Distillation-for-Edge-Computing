"""云端分类器: Linear / MLP，从随机初始化开始训练

验证协议（仅两种）:
  external : 传入独立真实验证集，全部代表特征参与训练，按真实验证集选 epoch
  none     : 不验证，使用最后一个 epoch（例如 BDM 选点探针）

旧的内部分层 10% 验证（internal）已移除，不再减少实际参与训练的代表特征数。

训练预算:
  fixed_epochs : 固定 epoch（默认）
  fixed_steps  : 固定优化步数，循环读取训练样本
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import macro_f1


class LinearCloud(nn.Module):
    def __init__(self, dim=1024, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPCloud(nn.Module):
    def __init__(self, dim=1024, hidden=512, num_classes=10, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def build_cloud(head, dim, num_classes):
    if head == "linear":
        return LinearCloud(dim, num_classes)
    if head == "mlp":
        return MLPCloud(dim, 512, num_classes)
    raise ValueError(head)


def train_cloud_classifier(features, labels, head="mlp", num_classes=10,
                           seed=0, epochs=200, lr=1e-3, wd=1e-4, batch=256,
                           device="cuda", verbose=False,
                           val_features=None, val_labels=None,
                           budget_mode="fixed_epochs", steps=None):
    """训练云端分类器。

    val_features/val_labels 同时给出时进入 external 模式：
      - 全部 features 参与训练（n_train == len(features)）
      - 用独立真实验证集做 epoch 选择
    未提供时进入 none 模式：全部 features 参与训练，使用最后一个 epoch。
    """
    device = torch.device(device)
    x = features.float().cpu()
    y = labels.long().cpu()
    if x.dim() != 2:
        raise ValueError(f"features 必须是 [N,d]，实际 {tuple(x.shape)}")
    if y.dim() != 1 or len(y) != len(x):
        raise ValueError("labels 必须是一维且与 features 等长")
    if len(x) == 0:
        raise ValueError("features 为空")
    if not torch.isfinite(x).all():
        raise ValueError("features 含 NaN/Inf")
    if int(y.min()) < 0 or int(y.max()) >= num_classes:
        raise ValueError(f"labels 越界: [{int(y.min())}, {int(y.max())}]")

    # ---------- 验证模式 ----------
    external = val_features is not None or val_labels is not None
    if external and (val_features is None or val_labels is None):
        raise ValueError("val_features 与 val_labels 必须同时提供")
    if external:
        validation_mode = "external"
        xt, yt = x, y
        xv = val_features.float().cpu()
        yv = val_labels.long().cpu()
        if xv.dim() != 2 or xv.size(1) != x.size(1):
            raise ValueError(f"验证特征维度不一致: {tuple(xv.shape)} vs {tuple(x.shape)}")
        if yv.dim() != 1 or len(yv) != len(xv):
            raise ValueError("验证 labels 与 features 长度不一致")
        if not torch.isfinite(xv).all():
            raise ValueError("验证特征含 NaN/Inf")
        if len(yv) and (int(yv.min()) < 0 or int(yv.max()) >= num_classes):
            raise ValueError("验证 labels 越界")
    else:
        validation_mode = "none"
        xt, yt = x, y
        xv, yv = x[:0], y[:0]

    xt, yt = xt.to(device), yt.to(device)
    xv, yv = xv.to(device), yv.to(device)

    # ---------- 训练预算 ----------
    steps_per_epoch = max(1, math.ceil(len(xt) / batch))
    if budget_mode == "fixed_epochs":
        total_steps = int(epochs) * steps_per_epoch
        max_steps = total_steps
        n_epochs = int(epochs)
    elif budget_mode == "fixed_steps":
        if not steps or int(steps) <= 0:
            raise ValueError("fixed_steps 模式需要正整数 steps")
        total_steps = int(steps)
        max_steps = total_steps
        n_epochs = int(math.ceil(max_steps / steps_per_epoch))
    else:
        raise ValueError(f"未知 budget_mode: {budget_mode}")

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = build_cloud(head, x.size(1), num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, total_steps))
        best_acc, best_state, best_epoch, best_step = -1.0, None, 0, 0
        step_count = 0
        t0 = time.time()

        def _validate(ep, st):
            nonlocal best_acc, best_state, best_epoch, best_step
            if len(xv) == 0:
                return
            model.eval()
            with torch.no_grad():
                acc = (model(xv).argmax(1) == yv).float().mean().item() * 100
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
                best_epoch, best_step = ep, st

        for ep in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(len(xt), device=device)
            for i in range(0, len(xt), batch):
                if step_count >= max_steps:
                    break
                j = perm[i:i + batch]
                loss = F.cross_entropy(model(xt[j]), yt[j])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
                step_count += 1
            if len(xv) and (ep % 10 == 0 or step_count >= max_steps):
                _validate(ep, step_count)
            if step_count >= max_steps:
                break
        # 固定步数可能停在 epoch 中间，补一次最终验证
        if len(xv) and best_step != step_count:
            _validate(min(ep, n_epochs), step_count)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        info = {"val_acc": best_acc if best_acc >= 0 else None,
                "train_sec": time.time() - t0,
                "n_input_features": int(len(x)),
                "n_train": int(len(xt)), "n_val": int(len(xv)),
                "validation_mode": validation_mode,
                "best_epoch": int(best_epoch), "best_step": int(best_step),
                "optimizer_steps": int(step_count),
                "head": head, "budget_mode": budget_mode,
                "epochs": int(n_epochs), "steps_per_epoch": int(steps_per_epoch)}
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
    return model, info


@torch.no_grad()
def evaluate_cloud_classifier(model, features, labels, num_classes=10, batch=512):
    dev = next(model.parameters()).device
    model.eval()
    preds = []
    x = features.float()
    for i in range(0, x.size(0), batch):
        preds.append(model(x[i:i + batch].to(dev)).argmax(1).cpu())
    p = torch.cat(preds)
    y = labels.long().cpu()
    return {"acc": (p == y).float().mean().item() * 100,
            "f1": macro_f1(p, y, num_classes), "pred": p}


def quick_val_acc(syn, syn_y, val_x, val_y, num_classes=10, epochs=50,
                  seed=0, device="cuda", head="linear"):
    """DM checkpoint 选择: 合成特征训练一个小分类器，在真实验证特征上评估"""
    model, _ = train_cloud_classifier(
        syn, syn_y, head=head, num_classes=num_classes, seed=seed,
        epochs=epochs, lr=1e-3, wd=1e-4, batch=256, device=device)
    return evaluate_cloud_classifier(model, val_x, val_y, num_classes)["acc"]
