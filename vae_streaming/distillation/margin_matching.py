"""分类间隔分位数匹配（Plain DM 的附加项，实验用）

边缘 VAE 自带一个受标签监督的分类头，对标签为 c 的潜向量定义

    m(z, c) = h_c(z) - max_{k != c} h_k(z)

本模块负责三件事：
  1. 把归一化潜向量还原到分类头训练时的尺度（*std + mean）并打分；
  2. 在每类真实样本上取 (i-0.5)/q 分位数，作为该类 q 个代表的目标间隔；
  3. 对代表向量逐类排序后做分位数回归，只对代表反传。

分类头在此始终 eval() 且参数冻结，不参与任何梯度。
"""
import copy

import torch


class MarginQuantileMatcher:
    def __init__(self, classifier, mean, std, quotas, latent_dim=16,
                 spatial=8, device="cpu"):
        # 用深拷贝：不改变在线 VAE 分类头的 train/eval 状态与优化器状态
        self.classifier = copy.deepcopy(classifier).to(device)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.mean = torch.as_tensor(mean).float().to(device)
        self.std = torch.as_tensor(std).float().to(device).clamp_min(1e-6)
        if self.mean.dim() == 2:  # 兼容 [C,1,1]
            self.mean = self.mean.unsqueeze(0)
        if self.std.dim() == 2:
            self.std = self.std.unsqueeze(0)
        # mean/std 为逐通道 [1,C,1,1]，空间尺寸需显式给出
        self.shape = (int(latent_dim), int(spatial), int(spatial))
        self.quotas = [int(q) for q in quotas]
        self.num_classes = len(self.quotas)
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    def _raw(self, feats):
        return feats.reshape(-1, *self.shape) * self.std + self.mean

    def margins(self, feats, labels):
        """feats: [N, D] 归一化潜向量（可求导）；labels: [N] 真实类标签。"""
        logits = self.classifier(self._raw(feats))
        lab = labels.reshape(-1, 1)
        hc = logits.gather(1, lab).squeeze(1)
        other = logits.scatter(1, lab, float("-inf"))
        return hc - other.max(1).values

    # ------------------------------------------------------------------
    @torch.no_grad()
    def class_targets(self, real_feats, real_labels):
        """返回 (targets [C, qmax], mask [C, qmax])。

        第 i 个代表（0-based）匹配该类真实间隔分布的第 (i+0.5)/q 分位数。
        """
        rf = torch.as_tensor(real_feats).float().to(self.device)
        rl = torch.as_tensor(real_labels).long().to(self.device)
        m = self.margins(rf, rl)
        qmax = max(self.quotas)
        targets = torch.zeros(self.num_classes, qmax, device=self.device)
        mask = torch.zeros(self.num_classes, qmax, dtype=torch.bool,
                           device=self.device)
        for c, q in enumerate(self.quotas):
            if q == 0:
                continue
            mc = torch.sort(m[rl == c]).values
            n = int(mc.numel())
            if n == 0:
                continue
            if n == 1:
                t = mc.expand(q).clone()
            else:
                pos = (torch.arange(q, device=self.device, dtype=mc.dtype)
                       + 0.5) / q * (n - 1)
                lo = pos.floor().long().clamp_(0, n - 1)
                hi = (lo + 1).clamp_(max=n - 1)
                w = pos - lo.to(pos.dtype)
                t = mc[lo] * (1.0 - w) + mc[hi] * w
            targets[c, :q] = t
            mask[c, :q] = True
        return targets, mask

    def class_loss(self, reps, targets, mask):
        """reps: [C, qmax, D] 代表向量；返回逐类分位数回归损失 [C]。"""
        losses = reps.new_zeros(self.num_classes)
        for c, q in enumerate(self.quotas):
            if q == 0 or not bool(mask[c, :q].any()):
                continue
            lab = torch.full((q,), c, dtype=torch.long, device=reps.device)
            m = torch.sort(self.margins(reps[c, :q], lab)).values
            losses[c] = (m - targets[c, :q]).pow(2).mean()
        return losses
