"""Streaming Distribution Matching（原版 DM：固定随机特征空间均值匹配）

每轮:
  1. 代表特征先由 FeatureAlignment 转换到当前 Encoder 版本。
  2. 目标均值来自当前版本真实特征经固定随机网络后的逐类均值。
  3. 按类别独立优化，anchor 指向对齐后的上一轮代表特征。

支持全量更新与固定补丁子集更新（partial），后者保证未提交位置保持对齐值。
"""
import time

import torch

from random_feature import MultiRandomMLP, RandomMLP, ResidualMLP


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
        self.la = float(self.cfg.get("lambda_anchor", 0.0))
        self.latent_w = float(self.cfg.get("lambda_latent", 0.0))
        self.feature_type = str(self.cfg.get("feature_type", "mlp"))
        f_kwargs = dict(
            input_dim=self.dim,
            hidden_dim=int(self.cfg.get("feature_hidden", 2048)),
            output_dim=int(self.cfg.get("feature_out", 1024)),
            seed=int(self.cfg.get("feature_seed", 0)))
        # 固定随机网络：参数不训练，只对 synthetic 特征保留梯度
        if self.feature_type == "multi_mlp":
            self.n_mappings = int(self.cfg.get("num_mappings", 3))
            self.feature_net = MultiRandomMLP(
                n_mappings=self.n_mappings, **f_kwargs).to(self.device)
        elif self.feature_type == "residual_mlp":
            self.n_mappings = 1
            self.feature_net = ResidualMLP(**f_kwargs).to(self.device)
        else:
            self.n_mappings = 1
            self.feature_net = RandomMLP(**f_kwargs).to(self.device)
        self.objective = {
            "mlp": "random_feature_mean",
            "multi_mlp": "multi_random_feature_mean",
            "residual_mlp": "residual_feature_mean",
        }.get(self.feature_type, "random_feature_mean")
        if self.latent_w > 0:
            self.objective += "+latent_mean"

    # ------------------------------------------------------------------
    def _apply_feat(self, x):
        """统一成 [K, N, D]：单映射时 K=1"""
        out = self.feature_net(x)
        return out if out.dim() == 3 else out.unsqueeze(0)

    # ------------------------------------------------------------------
    def _batch_loss(self, p, sel, active, anchor, real_mean,
                    real_latent_mean=None):
        """按类别一次性计算损失（[C,qmax,D] 批量，避免逐类 Python 循环）。

        p: 全部代表特征；sel: 本步参与更新的位置；active: 本轮有真实特征的类别。
        每个类别独立优化，因此对类别求和等价于逐类单独 backward。
        real_mean: [K,C,Do] 各随机映射下的逐类真实均值。
        real_latent_mean: [C,D] 原始潜空间的逐类真实均值（可选）。
        """
        m3 = sel.unsqueeze(-1).to(p.dtype)                # [C,qmax,1]
        q = sel.sum(1, keepdim=True).clamp_min(1).to(p.dtype)
        feat = self._apply_feat(p.reshape(-1, self.dim))   # [K,C,qmax,Do]
        feat = feat.reshape(feat.size(0), p.size(0), p.size(1), -1)
        f_mu = (feat * m3.unsqueeze(0)).sum(2) / q.unsqueeze(0)
        l_mean_c = (f_mu - real_mean).pow(2).mean(dim=(0, 2))
        l_anchor_c = ((p - anchor).pow(2) * m3).sum((1, 2)) / (q[:, 0] * self.dim)
        loss = self.lm * l_mean_c + self.la * l_anchor_c
        l_latent_c = None
        if real_latent_mean is not None:
            p_mu = (p * m3).sum(1) / q                     # [C,D]
            l_latent_c = (p_mu - real_latent_mean).pow(2).mean(1)
            loss = loss + self.latent_w * l_latent_c
        return loss[active].sum(), l_mean_c, l_anchor_c, l_latent_c

    # ------------------------------------------------------------------
    def update(self, allowed_idx=None, round_idx=0, verbose=False,
               real_features=None, real_labels=None):
        """原版 DM：目标为当前版本真实特征经固定随机网络后的逐类均值。"""
        if real_features is None or real_labels is None:
            raise ValueError("StreamingDM 必须提供 real_features/real_labels")
        real_features = torch.as_tensor(real_features).float().cpu()
        real_labels = torch.as_tensor(real_labels).long().cpu()
        if len(real_features) != len(real_labels):
            raise ValueError("real_features 与 real_labels 长度不一致")
        t0 = time.time()
        num_classes, dim = self.num_classes, self.dim
        qmax = max(self.quotas)
        dev = self.device

        # ---------- 组装 [C,qmax,D] 的稠密参数（不同配额用 valid 掩码）----------
        slot = torch.full((num_classes, qmax), -1, dtype=torch.long)
        valid = torch.zeros(num_classes, qmax, dtype=torch.bool)
        for c, (a, b) in self.slices.items():
            slot[c, :b - a] = torch.arange(a, b)
            valid[c, :b - a] = True
        valid_dev = valid.to(dev)
        p = torch.nn.Parameter(torch.zeros(num_classes, qmax, dim, device=dev))
        with torch.no_grad():
            p[valid_dev] = self.features[slot[valid]].to(dev)
        anchor = p.detach().clone()

        # 参与更新的位置：full = 全部有效位置；patch = 有效位置 ∩ 允许位置
        sel = valid_dev.clone()
        if allowed_idx is not None:
            allowed = torch.zeros(len(self.features), dtype=torch.bool)
            allowed[torch.as_tensor(allowed_idx, dtype=torch.long)] = True
            sel &= allowed[slot].to(dev)

        # ---------- 原版 DM 目标：固定随机网络下的逐类真实均值 ----------
        rl = real_labels.to(dev)
        rf_dev = real_features.to(dev)
        with torch.no_grad():
            rf = self._apply_feat(rf_dev)                  # [K,N,Do]
            rsum = torch.zeros(rf.size(0), num_classes, rf.size(-1),
                               device=dev)
            for k in range(rf.size(0)):
                rsum[k].index_add_(0, rl, rf[k])
            rcnt = torch.bincount(rl, minlength=num_classes)
            real_mean = rsum / rcnt.clamp_min(1)[None, :, None].to(dev)
            real_latent_mean = None
            if self.latent_w > 0:
                lsum = torch.zeros(num_classes, dim, device=dev)
                lsum.index_add_(0, rl, rf_dev)
                real_latent_mean = lsum / rcnt.clamp_min(1)[:, None].to(dev)
        has = rcnt > 0
        train = sel & has[:, None]
        hold = ~train
        if not bool(has.any()):
            raise ValueError("本轮没有任何类别的真实特征")

        # ---------- 批量优化 ----------
        opt = torch.optim.AdamW([p], lr=self.lr, weight_decay=self.wd)
        p0 = p.detach().clone()
        n_steps = self.updates_per_class
        acc = {"mean": torch.zeros(num_classes, device=dev),
               "anchor": torch.zeros(num_classes, device=dev),
               "latent": torch.zeros(num_classes, device=dev)
               if real_latent_mean is not None else None}
        for _ in range(n_steps):
            loss, lm_c, la_c, ll_c = self._batch_loss(
                p, train, has, anchor, real_mean, real_latent_mean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc["mean"] += lm_c.detach()
            acc["anchor"] += la_c.detach()
            if ll_c is not None:
                acc["latent"] += ll_c.detach()
        with torch.no_grad():
            # 未参与更新的位置严格保持对齐后的旧值（含权重衰减影响）
            p[hold] = p0[hold]
        for k in acc:
            if acc[k] is not None:
                acc[k] /= max(1, n_steps)

        # ---------- 写回并汇报 ----------
        with torch.no_grad():
            out = self.features.clone()
            out[slot[valid]] = p.detach()[valid_dev].cpu()
            n_opt = train.sum(1).cpu()
            total_c = torch.tensor([b - a for a, b in
                                    (self.slices[c] for c in range(num_classes))])
        per_class = []
        for c in range(num_classes):
            if not bool(has[c]):
                continue
            lm = float(acc["mean"][c].cpu())
            la = float(acc["anchor"][c].cpu())
            rec = {"class": c,
                   "loss": self.lm * lm + self.la * la,
                   "mean": lm, "anchor": la,
                   "n_optimized": int(n_opt[c]),
                   "n_total": int(total_c[c])}
            if acc["latent"] is not None:
                ll = float(acc["latent"][c].cpu())
                rec["latent"] = ll
                rec["loss"] += self.latent_w * ll
            per_class.append(rec)

        self.features = out
        return {"features": out, "info": {
            "method": "streaming_dm", "round": int(round_idx),
            "objective": self.objective,
            "feature_type": self.feature_type,
            "num_mappings": int(self.n_mappings),
            "lambda_latent": self.latent_w,
            "real_samples": int(len(real_features)),
            "per_class": per_class,
            "n_optimized": int(n_opt.sum()),
            "mean_loss": float(sum(x["loss"] for x in per_class) /
                               max(1, len(per_class))),
            "updates_per_class": self.updates_per_class,
            "sec": time.time() - t0}}
