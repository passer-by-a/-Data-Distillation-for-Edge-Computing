"""Streaming Distribution Matching

每轮:
  1. 代表特征先由 FeatureAlignment 转换到当前 Encoder 版本。
  2. 目标均值/方差来自当前版本 Streaming Statistical Memory（已含本轮新样本）。
  3. 按类别独立优化，anchor 指向对齐后的上一轮代表特征。

支持全量更新与固定补丁子集更新（partial），后者保证未提交位置保持对齐值。
"""
import time

import torch
import torch.nn.functional as F


class StreamingDM:
    def __init__(self, features, labels, quotas, cfg, seed=42, device="cpu"):
        self.features = torch.as_tensor(features).float().cpu().clone()
        self.labels = torch.as_tensor(labels).long().cpu()
        self.quotas = [int(q) for q in quotas]
        self.num_classes = len(self.quotas)
        self.dim = self.features.size(1)
        self.cfg = cfg or {}
        self.device = torch.device(device)
        self.seed = int(seed)
        self.slices = {}
        off = 0
        for c, q in enumerate(self.quotas):
            self.slices[c] = (off, off + q)
            off += q
        if off != len(self.features):
            raise ValueError("quotas 与代表特征数量不一致")
        self.updates_per_class = int(self.cfg.get("updates_per_class", 20))
        self.lr = float(self.cfg.get("lr", 0.01))
        self.wd = float(self.cfg.get("weight_decay", 0.0))
        self.lm = float(self.cfg.get("lambda_mean", 1.0))
        self.lv = float(self.cfg.get("lambda_var", 1.0))
        self.la = float(self.cfg.get("lambda_anchor", 0.0))

    # ------------------------------------------------------------------
    def _class_loss(self, p, stats_c, anchor_c):
        _, mu_r, var_r = stats_c
        mu_s = p.mean(0)
        var_s = p.var(0, unbiased=False) if len(p) > 1 else \
            torch.zeros_like(mu_s)
        l_mean = (mu_s - mu_r).pow(2).mean()
        l_var = (var_r - var_s).pow(2).mean() if len(p) > 1 else \
            torch.zeros((), device=p.device)
        l_anchor = (p - anchor_c).pow(2).mean()
        loss = self.lm * l_mean + self.lv * l_var + self.la * l_anchor
        return loss, {"mean": float(l_mean.detach()),
                      "var": float(l_var.detach()),
                      "anchor": float(l_anchor.detach())}

    # ------------------------------------------------------------------
    def update(self, statistics, allowed_idx=None, round_idx=0, verbose=False):
        """statistics: StreamingStatistics（当前版本）"""
        t0 = time.time()
        base = self.features.clone()
        allowed = None
        if allowed_idx is not None:
            allowed = torch.zeros(len(base), dtype=torch.bool)
            allowed[torch.as_tensor(allowed_idx, dtype=torch.long)] = True

        per_class = []
        for c in range(self.num_classes):
            a, b = self.slices[c]
            idx = torch.arange(a, b)
            take = idx if allowed is None else idx[allowed[idx]]
            if len(take) == 0:
                continue
            if not statistics.has_class(c):
                continue
            p = torch.nn.Parameter(base[a:b].to(self.device).clone())
            anchor = base[a:b].to(self.device).clone()
            opt = torch.optim.AdamW([p], lr=self.lr, weight_decay=self.wd)
            stats_c = statistics.get(c)
            stats_c = (stats_c[0], stats_c[1].to(self.device),
                       stats_c[2].to(self.device))
            take_local = (take - a).to(self.device)
            mask = torch.zeros(b - a, dtype=torch.bool, device=self.device)
            mask[take_local] = True
            losses = {"mean": 0.0, "var": 0.0, "anchor": 0.0}
            last = None
            for _ in range(self.updates_per_class):
                loss, parts = self._class_loss(p[mask], stats_c, anchor[mask])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                before = p.detach().clone()
                opt.step()
                # 未允许更新的位置严格保持对齐后的旧值
                with torch.no_grad():
                    p[~mask] = before[~mask]
                last = float(loss.detach())
                for k in losses:
                    losses[k] += parts[k] / self.updates_per_class
            with torch.no_grad():
                base[a:b] = p.detach().cpu()
            mu_s = base[a:b].mean(0)
            var_s = base[a:b].var(0, unbiased=False)
            ratio = float(var_s.mean() / stats_c[2].mean().clamp_min(1e-8))
            per_class.append({"class": c, "loss": last, **losses,
                              "var_ratio": ratio,
                              "n_optimized": int(len(take)),
                              "n_total": int(b - a)})

        self.features = base
        return {"features": base, "info": {
            "method": "streaming_dm", "round": int(round_idx),
            "per_class": per_class,
            "n_optimized": int(allowed.sum()) if allowed is not None
            else int(len(base)),
            "mean_loss": float(sum(x["loss"] for x in per_class) /
                               max(1, len(per_class))),
            "mean_var_ratio": float(sum(x["var_ratio"] for x in per_class) /
                                    max(1, len(per_class))),
            "updates_per_class": self.updates_per_class,
            "sec": time.time() - t0}}
