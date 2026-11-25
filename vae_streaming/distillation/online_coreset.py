"""Online Coreset 对照: Online K-means

与 Streaming DM 共享同一代表预算、同一固定补丁更新接口。
"""
import time

import numpy as np
import torch


class _BaseCoreset:
    def __init__(self, features, labels, quotas, dim=1024, seed=0):
        self.features = torch.as_tensor(features).float().cpu().clone()
        self.labels = torch.as_tensor(labels).long().cpu()
        self.quotas = [int(q) for q in quotas]
        self.dim = int(dim)
        self.seed = int(seed)
        self.num_classes = len(self.quotas)
        self.slices = {}
        off = 0
        for c, q in enumerate(self.quotas):
            self.slices[c] = (off, off + q)
            off += q
        self.rng = np.random.RandomState(seed)

    def align(self, alignment):
        self.features = alignment.apply(self.features).float()
        return self

    def update(self, features, labels, round_idx=0, allowed_idx=None):
        raise NotImplementedError

    def get_features(self):
        return self.features

    def _allowed_mask(self, allowed_idx):
        if allowed_idx is None:
            return None
        m = torch.zeros(len(self.features), dtype=torch.bool)
        m[torch.as_tensor(allowed_idx, dtype=torch.long)] = True
        return m

    def state_dict(self):
        return {"features": self.features.clone(), "labels": self.labels.clone(),
                "quotas": list(self.quotas), "dim": self.dim,
                "seed": self.seed, "rng_state": self.rng.get_state()}

    @classmethod
    def from_state_dict(cls, st, kind="kmeans"):
        cls_map = {"kmeans": OnlineKMeans, "exstream": ExStreamCoreset}
        if kind not in cls_map:
            raise ValueError(f"未知 coreset: {kind}")
        obj = cls_map[kind](st["features"], st["labels"], st["quotas"],
                            dim=st["dim"], seed=st["seed"])
        obj.features = st["features"].clone()
        obj.rng.set_state(st["rng_state"])
        return obj


class OnlineKMeans(_BaseCoreset):
    """Sculley 流式 k-means: 代表特征是逐类质心（与静态 K-means 口径一致）。"""

    def __init__(self, features, labels, quotas, dim=1024, seed=0,
                 initial_counts=None):
        super().__init__(features, labels, quotas, dim, seed)
        # 每个质心独立计数（Sculley 在线均值的正确形式）。旧实现每类只存
        # 一个计数，同类其它质心收到的样本也会压低该质心的学习率
        # （CIFAR-10 约 98 个质心/类，步长偏小约 78 倍；CIFAR-100 约 8 倍）。
        # 初始计数应继承 Base 聚类的实际簇大小；若从 0 开始，首个流式样本
        # 会以 lr=1 直接覆盖质心。
        total = int(self.features.size(0))
        if initial_counts is None:
            self.counts = np.ones(total, dtype=np.int64)
        else:
            counts = np.asarray(initial_counts, dtype=np.int64).flatten()
            if counts.shape == (self.num_classes,):
                # 兼容每类一个平均簇大小的旧接口：广播到该类每个质心
                counts = np.concatenate(
                    [np.full(int(q), counts[c], dtype=np.int64)
                     for c, q in enumerate(self.quotas)])
            elif counts.shape != (total,):
                raise ValueError(
                    f"initial_counts 需为每质心 {total} 维或每类 "
                    f"{self.num_classes} 维")
            self.counts = np.maximum(counts, 1)

    def align(self, alignment):
        super().align(alignment)
        return self

    def update(self, features, labels, round_idx=0, allowed_idx=None):
        t0 = time.time()
        features = torch.as_tensor(features).float().cpu()
        labels = torch.as_tensor(labels).long().cpu()
        allowed = self._allowed_mask(allowed_idx)
        n_assigned, n_skipped = 0, 0
        for c in range(self.num_classes):
            a, b = self.slices[c]
            z = features[labels == c]
            if len(z) == 0:
                continue
            if allowed is None:
                cand = torch.arange(b - a)
            else:
                cand = torch.where(allowed[a:b])[0]
            if len(cand) == 0:
                n_skipped += len(z)
                continue
            cent = self.features[a:b][cand]
            for j in range(len(z)):
                d = (cent - z[j]).pow(2).sum(1)
                k = int(d.argmin())
                g = a + int(cand[k])
                self.counts[g] += 1
                lr = 1.0 / float(self.counts[g])
                cent[k] = cent[k] + lr * (z[j] - cent[k])
                self.features[g] = cent[k]
                n_assigned += 1
        return {"features": self.features.clone(),
                "info": {"method": "online_kmeans", "round": int(round_idx),
                         "n_assigned": int(n_assigned),
                         "n_skipped": int(n_skipped),
                         "sec": time.time() - t0}}

    def state_dict(self):
        st = super().state_dict()
        st["counts"] = self.counts.copy()
        return st

    @classmethod
    def from_state_dict(cls, st, kind="kmeans"):
        obj = super().from_state_dict(st, kind="kmeans")
        obj.counts = np.asarray(st["counts"]).copy()
        return obj


class ExStreamCoreset(_BaseCoreset):
    """ExStream 式类缓冲代表集（Hayes et al., 2019）。

    与在线 K-means 的区别：新样本不再吸收到最近质心，而是占用一个独立
    槽位；当该类缓冲已满时，合并类内最相近的两个旧代表（按累计样本数
    加权平均），把腾出的槽位留给新样本。每槽需额外保存计数。

    官方实现: `tyler-hayes/ExStream` 的 `_stream_clustering_class_buffers`
    与 `_l2_dist_metric`（类内成对平方距离，对角线置 inf）。
    """

    def __init__(self, features, labels, quotas, dim=1024, seed=0,
                 initial_counts=None):
        super().__init__(features, labels, quotas, dim, seed)
        total = int(self.features.size(0))
        if initial_counts is None:
            self.counts = np.ones(total, dtype=np.int64)
        else:
            counts = np.asarray(initial_counts, dtype=np.int64).flatten()
            if counts.shape == (self.num_classes,):
                counts = np.concatenate(
                    [np.full(int(q), counts[c], dtype=np.int64)
                     for c, q in enumerate(self.quotas)])
            elif counts.shape != (total,):
                raise ValueError(
                    f"initial_counts 需为每槽 {total} 维或每类 "
                    f"{self.num_classes} 维")
            self.counts = np.maximum(counts, 0)

    # ------------------------------------------------------------------
    @staticmethod
    def _pair_sqdist(f):
        d = torch.cdist(f, f, p=2).pow(2)
        d.fill_diagonal_(float("inf"))
        return d

    def update(self, features, labels, round_idx=0, allowed_idx=None):
        t0 = time.time()
        features = torch.as_tensor(features).float().cpu()
        labels = torch.as_tensor(labels).long().cpu()
        allowed = self._allowed_mask(allowed_idx)
        n_inserted, n_merged, n_skipped = 0, 0, 0
        for c in range(self.num_classes):
            a, b = self.slices[c]
            if allowed is None:
                idx = torch.arange(a, b)
            else:
                idx = torch.where(allowed[a:b])[0] + a
            z = features[labels == c]
            if len(idx) == 0:
                n_skipped += len(z)
                continue
            for j in range(len(z)):
                # 优先占用空槽（初始未填满时）
                cnts = torch.as_tensor(self.counts[idx.numpy()])
                empty = idx[cnts == 0]
                if len(empty) > 0:
                    g = int(empty[0])
                    self.features[g] = z[j]
                    self.counts[g] = 1
                    n_inserted += 1
                    continue
                # 类缓冲已满：合并类内最近的两个旧代表
                if len(idx) < 2:
                    # 单槽类（或部分更新只放开 1 个槽）没有可合并的对；
                    # 若直接走下面的最近对逻辑，gi == gj 会让新样本先写入
                    # 又被 merged 覆盖而丢失。这里按累计样本数加权吸收。
                    g = int(idx[0])
                    cg = int(self.counts[g])
                    self.features[g] = (self.features[g] * cg + z[j]) \
                        / float(cg + 1)
                    self.counts[g] = cg + 1
                    n_merged += 1
                    continue
                d2 = self._pair_sqdist(self.features[idx])
                k = int(d2.argmin())
                i_loc = k // len(idx)
                j_loc = k % len(idx)
                gi, gj = int(idx[i_loc]), int(idx[j_loc])
                ci, cj = int(self.counts[gi]), int(self.counts[gj])
                merged = (self.features[gi] * ci +
                          self.features[gj] * cj) / float(ci + cj)
                # 官方逻辑：idx0 槽放新样本（count=1），idx1 槽放合并代表
                self.features[gi] = z[j]
                self.counts[gi] = 1
                self.features[gj] = merged
                self.counts[gj] = ci + cj
                n_merged += 1
        return {"features": self.features.clone(),
                "info": {"method": "exstream", "round": int(round_idx),
                         "n_assigned": int(n_inserted + n_merged),
                         "n_inserted": int(n_inserted),
                         "n_merged": int(n_merged),
                         "n_skipped": int(n_skipped),
                         "n_slots": int(len(self.features)),
                         "counts_sum": int(self.counts.sum()),
                         "sec": time.time() - t0}}

    def state_dict(self):
        st = super().state_dict()
        st["counts"] = self.counts.copy()
        return st

    @classmethod
    def from_state_dict(cls, st, kind="exstream"):
        obj = super().from_state_dict(st, kind="exstream")
        obj.counts = np.asarray(st["counts"]).copy()
        return obj


def build_coreset(kind, features, labels, quotas, dim=1024, seed=0,
                  initial_counts=None):
    if kind == "kmeans":
        return OnlineKMeans(features, labels, quotas, dim, seed,
                            initial_counts=initial_counts)
    if kind == "exstream":
        return ExStreamCoreset(features, labels, quotas, dim, seed,
                               initial_counts=initial_counts)
    raise ValueError(f"未知 coreset: {kind}")


def coreset_from_state_dict(st, kind):
    """断点恢复：按方法名选择对应的 coreset 类。"""
    if kind == "kmeans":
        return OnlineKMeans.from_state_dict(st, kind="kmeans")
    if kind == "exstream":
        return ExStreamCoreset.from_state_dict(st, kind="exstream")
    raise ValueError(f"未知 coreset: {kind}")
