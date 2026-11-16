"""Streaming Statistical Memory: 按类别的在线 count / mean / M2

约束:
  - 只接收"当前轮新到达"的特征；Replay 回放样本不得重复计入。
  - 不同 Encoder 版本的统计量禁止直接合并，必须先 align_to 新版本。
  - 对齐后的历史统计量标记 estimated=True，表示它是版本转换估计值。
"""
import copy

import numpy as np
import torch


class StreamingStatistics:
    def __init__(self, num_classes, dim, version=0):
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.version = int(version)
        self.count = np.zeros(self.num_classes, dtype=np.int64)
        self.mean = torch.zeros(self.num_classes, self.dim, dtype=torch.float64)
        self.M2 = torch.zeros(self.num_classes, self.dim, dtype=torch.float64)
        self.estimated = np.zeros(self.num_classes, dtype=bool)
        self.total_updates = 0
        self.log = []

    # ------------------------------------------------------------------
    def reset(self, version):
        self.count[:] = 0
        self.mean.zero_()
        self.M2.zero_()
        self.estimated[:] = False
        self.version = int(version)

    def align_to(self, version, alignment):
        """把全部类别统计量从当前版本转换到 version（对角仿射近似）"""
        if int(version) == self.version:
            return
        if alignment is None:
            raise ValueError("跨版本统计合并必须提供 alignment")
        if int(alignment.version_from) != self.version or \
                int(alignment.version_to) != int(version):
            raise ValueError(
                f"对齐版本不匹配: 统计 v{self.version} → v{version}, "
                f"alignment v{alignment.version_from} → v{alignment.version_to}")
        a = alignment.scale.double()
        b = alignment.bias.double()
        for c in range(self.num_classes):
            if self.count[c] == 0:
                continue
            var = self.M2[c] / self.count[c]
            new_mean = a * self.mean[c] + b
            new_var = a.pow(2) * var
            self.mean[c] = new_mean
            self.M2[c] = new_var * self.count[c]
            self.estimated[c] = True
        self.version = int(version)

    # ------------------------------------------------------------------
    def update(self, features, labels, version, round_idx=None):
        """合并当前轮新样本。features: [n,D]；labels: [n]"""
        if int(version) != self.version:
            raise ValueError(
                f"统计量版本 v{self.version} 与新特征版本 v{version} 不一致；"
                f"必须先 align_to 再 update（禁止跨版本直接混用）")
        features = torch.as_tensor(features).double().cpu()
        labels = torch.as_tensor(labels).long().cpu()
        if features.dim() != 2 or features.size(1) != self.dim:
            raise ValueError(f"特征形状错误: {tuple(features.shape)}")
        if len(features) != len(labels):
            raise ValueError("features 与 labels 长度不一致")
        batch_counts = []
        for c in torch.unique(labels).tolist():
            f = features[labels == c]
            n = len(f)
            batch_counts.append(int(n))
            bm = f.mean(0)
            bM2 = (f - bm).pow(2).sum(0)
            old_n = int(self.count[c])
            new_n = old_n + n
            if old_n == 0:
                self.mean[c] = bm
                self.M2[c] = bM2
            else:
                delta = bm - self.mean[c]
                self.mean[c] = self.mean[c] + delta * (n / new_n)
                self.M2[c] = self.M2[c] + bM2 + \
                    delta.pow(2) * (old_n * n / new_n)
            self.count[c] = new_n
        self.total_updates += 1
        self.log.append({"version": int(version), "round": round_idx,
                         "n": int(len(features)),
                         "batch_class_counts": batch_counts})
        return self

    # ------------------------------------------------------------------
    def get(self, c):
        if self.count[c] == 0:
            raise ValueError(f"类别 {c} 没有统计量")
        var = self.M2[c] / self.count[c]
        return int(self.count[c]), self.mean[c].float(), var.float()

    def stats_dict(self):
        return {c: (self.mean[c].float(), (self.M2[c] / self.count[c]).float())
                for c in range(self.num_classes) if self.count[c] > 0}

    def has_class(self, c):
        return self.count[c] > 0

    @property
    def n_seen(self):
        return int(self.count.sum())

    def summary(self):
        return {"version": int(self.version),
                "n_seen": self.n_seen,
                "per_class_count": self.count.tolist(),
                "n_classes_seen": int((self.count > 0).sum()),
                "estimated_classes": [int(c) for c in np.where(self.estimated)[0]],
                "total_updates": self.total_updates}

    # ------------------------------------------------------------------
    def state_dict(self):
        return {"num_classes": self.num_classes, "dim": self.dim,
                "version": self.version, "count": self.count.copy(),
                "mean": self.mean.clone(), "M2": self.M2.clone(),
                "estimated": self.estimated.copy(),
                "total_updates": self.total_updates,
                "log": copy.deepcopy(self.log)}

    @classmethod
    def from_state_dict(cls, st):
        obj = cls(st["num_classes"], st["dim"], st["version"])
        obj.count = np.asarray(st["count"]).copy()
        obj.mean = st["mean"].clone()
        obj.M2 = st["M2"].clone()
        obj.estimated = np.asarray(st["estimated"]).copy()
        obj.total_updates = int(st["total_updates"])
        obj.log = copy.deepcopy(st["log"])
        return obj
