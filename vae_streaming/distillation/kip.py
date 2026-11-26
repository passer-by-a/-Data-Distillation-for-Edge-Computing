"""KIP: Kernel Inducing Points 式潜空间特征回归蒸馏

参考官方实现 `google-research/google-research/kip`（`KIP.ipynb`, ICLR 2021,
Nguyen et al.）: 在 support 集上做 RBF 核岭回归的闭式解

    alpha = (K_ss + λI)^{-1} Y_s,   pred_t = K_ts alpha

以 pred_t 与目标集标签的 MSE 为损失，对 support 特征反传。

与官方的差异: 官方在像素空间用 support 图像端到端重训网络；这里 Encoder
固定，只在 1024 维潜空间优化已有代表向量（预算、对齐、云端协议不变），
即 KIP 的潜特征版本。
"""
import math
import time

import torch

_EPS = 1e-8


def _sqdist(a, b):
    """成对平方欧氏距离，只用矩阵乘法避免 [N,M,D] 中间张量。"""
    a2 = a.pow(2).sum(1, keepdim=True)
    b2 = b.pow(2).sum(1, keepdim=True)
    d2 = a2 + b2.t() - 2.0 * (a @ b.t())
    return d2.clamp_min(0.0)


def _rbf(a, b, sigma2):
    return torch.exp(-_sqdist(a, b) / (2.0 * sigma2))


def _stride_sample(x, max_n):
    """按等距跨步抽取子样本，覆盖整个取值范围。

    代表特征与 replay 编码都按类别连续排列，直接取前 max_n 行只会覆盖
    前几个类，median heuristic 会明显低估带宽。跨步抽样保持类别覆盖。
    """
    n = len(x)
    if n <= max_n:
        return x
    step = int(math.ceil(n / float(max_n)))
    return x[::step][:max_n]


def _median_sqdist(a, b, max_n=512):
    """median heuristic：在跨类别等距子样本上估计 RBF 方差尺度。"""
    with torch.no_grad():
        d2 = _sqdist(_stride_sample(a, max_n), _stride_sample(b, max_n))
        med = float(d2.median().item())
    return max(med, _EPS)


def median_sqdist(a, b, max_n=512):
    """bandwidth heuristic 的公开入口（供只读诊断复用）。"""
    return _median_sqdist(a, b, max_n)


@torch.no_grad()
def krr_scores(support, support_labels, query, sigma2, ridge=0.01,
               num_classes=None):
    """RBF 核岭回归分数：alpha=(K_ss+λI)^-1 Y_s，返回 K_qs alpha。

    仅用于只读诊断；训练目标仍走 `_objective` 的内联实现，避免改变
    梯度路径与数值。
    """
    if num_classes is None:
        num_classes = int(support_labels.max().item()) + 1
    ys = torch.nn.functional.one_hot(
        support_labels.long(), int(num_classes)).float()
    k_ss = _rbf(support, support, sigma2) + (float(ridge) + 1e-6) * torch.eye(
        support.size(0), device=support.device, dtype=support.dtype)
    try:
        alpha = torch.linalg.solve(k_ss, ys)
    except RuntimeError:
        alpha = torch.linalg.lstsq(k_ss, ys).solution
    return _rbf(query, support, sigma2) @ alpha


class KIPDistiller:
    """在固定预算的代表特征上做 KIP 式核回归蒸馏。"""

    def __init__(self, features, labels, quotas, cfg, seed=42, device="cpu"):
        self.features = torch.as_tensor(features).float().cpu().clone()
        self.labels = torch.as_tensor(labels).long().cpu()
        self.quotas = [int(q) for q in quotas]
        self.num_classes = len(self.quotas)
        self.dim = self.features.size(1)
        self.cfg = cfg or {}
        self.device = torch.device(device)
        self.seed = int(seed)
        self.steps = int(self.cfg.get("kip_steps",
                                      self.cfg.get("updates_per_class", 20)))
        self.lr = float(self.cfg.get("kip_lr", self.cfg.get("lr", 0.01)))
        self.ridge = float(self.cfg.get("kip_ridge", 0.01))
        self.bw_mult = float(self.cfg.get("kip_bandwidth_mult", 1.0))
        self.anchor_w = float(self.cfg.get("kip_lambda_anchor", 0.0))
        self.bw_max_n = int(self.cfg.get("kip_bandwidth_samples", 512))

    # ------------------------------------------------------------------
    def update(self, allowed_idx=None, round_idx=0, real_features=None,
               real_labels=None, paired_features=None, target_scores=None,
               cross_weight=0.0):
        if real_features is None or real_labels is None:
            raise ValueError("KIPDistiller 必须提供 real_features/real_labels")
        t0 = time.time()
        rf = torch.as_tensor(real_features).float().cpu()
        ry = torch.as_tensor(real_labels).long().cpu()
        if len(rf) != len(ry):
            raise ValueError("real_features 与 real_labels 长度不一致")
        if len(rf) == 0:
            raise ValueError("KIP 目标集为空")

        dev = self.device
        num_classes = self.num_classes
        p0 = self.features.to(dev)
        ys = torch.nn.functional.one_hot(self.labels.to(dev),
                                         num_classes).float()
        if int(ry.min()) < 0 or int(ry.max()) >= num_classes:
            raise ValueError("目标标签超出类别数")
        yt = torch.nn.functional.one_hot(ry.to(dev), num_classes).float()
        rf = rf.to(dev)

        # 跨版本项：同一张历史图在新坐标下的旧版本完整类别分数。
        # 只在调用方显式给出常量配对集且权重 > 0 时启用；否则完全走原路径。
        paired = None
        targets = None
        cross_w = float(cross_weight)
        if (paired_features is not None and target_scores is not None
                and cross_w > 0):
            paired = torch.as_tensor(paired_features).float().to(dev)
            targets = torch.as_tensor(target_scores).float().to(dev)
            if len(paired) != len(targets):
                raise ValueError("paired_features 与 target_scores 长度不一致")
            if paired.size(0) > 0 and targets.size(1) != num_classes:
                raise ValueError("target_scores 类别数与代表集不一致")
            if paired.size(0) == 0:
                paired = targets = None

        # 带宽只依赖当前代表与目标特征，不参与梯度
        sigma2 = _median_sqdist(p0, rf, self.bw_max_n) * self.bw_mult
        sigma2 = max(sigma2, _EPS)

        allowed = torch.ones(len(self.features), dtype=torch.bool)
        if allowed_idx is not None:
            allowed = torch.zeros(len(self.features), dtype=torch.bool)
            allowed[torch.as_tensor(allowed_idx, dtype=torch.long)] = True
        allowed = allowed.to(dev)
        hold = ~allowed

        p = torch.nn.Parameter(p0.clone())
        anchor = p.detach().clone()
        opt = torch.optim.Adam([p], lr=self.lr)
        # loss 统一在参数更新后重新计算：首末口径一致，且末值是最终代表
        # 的真实目标值，而不是最后一次更新前的旧值
        with torch.no_grad():
            first = self._terms(p, ys, rf, yt, sigma2, anchor, paired,
                                targets, cross_w)
        n_done = 0
        for _ in range(max(0, self.steps)):
            loss = self._objective(p, ys, rf, yt, sigma2, anchor, paired,
                                   targets, cross_w)
            if not torch.isfinite(loss):
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if p.grad is not None and not torch.isfinite(p.grad).all():
                break
            opt.step()
            with torch.no_grad():
                # 未提交位置严格保持对齐后的旧值
                p[hold] = anchor[hold]
            n_done += 1

        last = first
        if n_done > 0:
            with torch.no_grad():
                last = self._terms(p, ys, rf, yt, sigma2, anchor, paired,
                                   targets, cross_w)

        feats = p.detach().cpu().clone()
        if not torch.isfinite(feats).all():
            feats = self.features.clone()
            n_done = 0
        info = {"method": "kip", "round": int(round_idx),
                "variant": ("kip_cross_version"
                            if paired is not None else "kip"),
                "steps": int(n_done),
                "loss_first": float(first[0]), "loss_last": float(last[0]),
                "loss_base_first": float(first[1]),
                "loss_base_last": float(last[1]),
                "loss_cross_first": (None if first[2] is None
                                     else float(first[2])),
                "loss_cross_last": (None if last[2] is None
                                    else float(last[2])),
                "cross_weight": float(cross_w),
                "n_paired": (0 if paired is None else int(paired.size(0))),
                "sigma2": float(sigma2),
                "ridge": self.ridge, "lr": self.lr,
                "n_support": int(len(self.features)),
                "n_target": int(len(rf)),
                "n_updated": (int(allowed.sum().item())
                              if n_done > 0 else 0),
                "sec": time.time() - t0}
        return {"features": feats, "info": info}

    def _terms(self, p, ys, rf, yt, sigma2, anchor, paired=None,
               targets=None, cross_w=0.0):
        """返回 (总损失, 基础 KIP 项, 跨版本项)；跨版本项未启用时为 None。"""
        k_ss = _rbf(p, p, sigma2) + (self.ridge + 1e-6) * torch.eye(
            p.size(0), device=p.device, dtype=p.dtype)
        try:
            alpha = torch.linalg.solve(k_ss, ys)
        except RuntimeError:
            # 极端病态时退化为最小二乘，保证不中断整轮
            alpha = torch.linalg.lstsq(k_ss, ys).solution
        base = torch.nn.functional.mse_loss(_rbf(rf, p, sigma2) @ alpha, yt)
        if self.anchor_w > 0:
            base = base + self.anchor_w * (p - anchor).pow(2).mean()
        if cross_w > 0 and paired is not None and paired.size(0) > 0:
            cross = torch.nn.functional.mse_loss(
                _rbf(paired, p, sigma2) @ alpha, targets)
            return base + cross_w * cross, base, cross
        return base, base, None

    def _objective(self, p, ys, rf, yt, sigma2, anchor, paired=None,
                   targets=None, cross_w=0.0):
        """KIP 闭式核回归预测与 one-hot 目标的 MSE（支持梯度与 no_grad）。"""
        return self._terms(p, ys, rf, yt, sigma2, anchor, paired, targets,
                           cross_w)[0]
