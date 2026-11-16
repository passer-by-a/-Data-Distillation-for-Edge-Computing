"""Online Coreset 对照: Reservoir / Online K-means / Online Herding

三者与 Streaming DM 共享同一代表预算、同一固定补丁更新接口。
"""
import time

import numpy as np
import torch

from vae_bdm.baselines import herding_select


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
    def from_state_dict(cls, st, kind="reservoir"):
        cls_map = {"reservoir": ReservoirCoreset,
                   "kmeans": OnlineKMeans, "herding": OnlineHerding}
        obj = cls_map[kind](st["features"], st["labels"], st["quotas"],
                            dim=st["dim"], seed=st["seed"])
        obj.features = st["features"].clone()
        obj.rng.set_state(st["rng_state"])
        return obj


class ReservoirCoreset(_BaseCoreset):
    """逐类 reservoir sampling；代表特征是真实历史特征。"""

    def __init__(self, features, labels, quotas, dim=1024, seed=0):
        super().__init__(features, labels, quotas, dim, seed)
        self.seen = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, features, labels, round_idx=0, allowed_idx=None):
        t0 = time.time()
        features = torch.as_tensor(features).float().cpu()
        labels = torch.as_tensor(labels).long().cpu()
        allowed = self._allowed_mask(allowed_idx)
        n_replaced = 0
        for c in range(self.num_classes):
            a, b = self.slices[c]
            z = features[labels == c]
            if len(z) == 0:
                continue
            if allowed is None:
                slots = torch.arange(b - a)
            else:
                slots = torch.where(allowed[a:b])[0]
            if len(slots) == 0:
                continue
            for j in range(len(z)):
                self.seen[c] += 1
                if self.seen[c] <= len(slots):
                    pos = int(slots[int(self.seen[c]) - 1])
                else:
                    k = int(self.rng.randint(0, int(self.seen[c])))
                    if k >= len(slots):
                        continue
                    pos = int(slots[k])
                self.features[a + pos] = z[j]
                n_replaced += 1
        return {"features": self.features.clone(),
                "info": {"method": "reservoir", "round": int(round_idx),
                         "n_replaced": int(n_replaced),
                         "sec": time.time() - t0}}

    def state_dict(self):
        st = super().state_dict()
        st["seen"] = self.seen.copy()
        return st

    @classmethod
    def from_state_dict(cls, st, kind="reservoir"):
        obj = super().from_state_dict(st, kind="reservoir")
        obj.seen = np.asarray(st["seen"]).copy()
        return obj


class OnlineKMeans(_BaseCoreset):
    """Sculley 流式 k-means: 代表特征是逐类质心（与静态 K-means 口径一致）。"""

    def __init__(self, features, labels, quotas, dim=1024, seed=0):
        super().__init__(features, labels, quotas, dim, seed)
        self.counts = np.zeros(self.num_classes, dtype=np.int64)

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
                self.counts[c] += 1
                lr = 1.0 / float(self.counts[c])
                cent[k] = cent[k] + lr * (z[j] - cent[k])
                self.features[a + int(cand[k])] = cent[k]
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


class OnlineHerding(_BaseCoreset):
    """有界候选池 + 每轮 herding 重选；候选池容量 = quota × pool_factor。"""

    def __init__(self, features, labels, quotas, dim=1024, seed=0,
                 pool_factor=4):
        super().__init__(features, labels, quotas, dim, seed)
        self.pool_factor = int(pool_factor)
        self.pools = [[] for _ in range(self.num_classes)]
        for c, (a, b) in self.slices.items():
            self.pools[c] = [self.features[i].clone() for i in range(a, b)]
        self.pool_seen = np.zeros(self.num_classes, dtype=np.int64)
        self.replacements = 0

    def align(self, alignment):
        for c in range(self.num_classes):
            if self.pools[c]:
                self.pools[c] = [alignment.apply(z.unsqueeze(0))[0]
                                 for z in self.pools[c]]
        super().align(alignment)
        return self

    def update(self, features, labels, round_idx=0, allowed_idx=None):
        t0 = time.time()
        features = torch.as_tensor(features).float().cpu()
        labels = torch.as_tensor(labels).long().cpu()
        allowed = self._allowed_mask(allowed_idx)
        for c in range(self.num_classes):
            z = features[labels == c]
            if len(z) == 0:
                continue
            cap = max(self.quotas[c], self.quotas[c] * self.pool_factor)
            pool = self.pools[c]
            for j in range(len(z)):
                self.pool_seen[c] += 1
                if len(pool) < cap:
                    pool.append(z[j])
                else:
                    k = int(self.rng.randint(0, int(self.pool_seen[c])))
                    if k < cap:
                        pool[k] = z[j]
                        self.replacements += 1
        for c in range(self.num_classes):
            a, b = self.slices[c]
            pool = self.pools[c]
            if not pool:
                continue
            sel = herding_select(torch.stack(pool).numpy(), b - a)
            new = torch.stack([pool[i] for i in sel], 0)
            if allowed is not None:
                m = allowed[a:b]
                self.features[a:b][m] = new[m]
            else:
                self.features[a:b] = new
        return {"features": self.features.clone(),
                "info": {"method": "online_herding", "round": int(round_idx),
                         "pool_sizes": [len(p) for p in self.pools],
                         "replacements": int(self.replacements),
                         "sec": time.time() - t0}}

    def state_dict(self):
        st = super().state_dict()
        st["pools"] = [[z.clone() for z in p] for p in self.pools]
        st["pool_factor"] = self.pool_factor
        st["pool_seen"] = self.pool_seen.copy()
        st["replacements"] = self.replacements
        return st

    @classmethod
    def from_state_dict(cls, st, kind="herding"):
        obj = cls(st["features"], st["labels"], st["quotas"], dim=st["dim"],
                  seed=st["seed"], pool_factor=st["pool_factor"])
        obj.features = st["features"].clone()
        obj.pools = [[z.clone() for z in p] for p in st["pools"]]
        obj.pool_seen = np.asarray(st["pool_seen"]).copy()
        obj.replacements = int(st["replacements"])
        obj.rng.set_state(st["rng_state"])
        return obj


def build_coreset(kind, features, labels, quotas, dim=1024, seed=0):
    if kind == "reservoir":
        return ReservoirCoreset(features, labels, quotas, dim, seed)
    if kind == "kmeans":
        return OnlineKMeans(features, labels, quotas, dim, seed)
    if kind == "herding":
        return OnlineHerding(features, labels, quotas, dim, seed)
    raise ValueError(f"未知 coreset: {kind}")
