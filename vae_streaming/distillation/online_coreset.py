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


class GCRCoreset(_BaseCoreset):
    """GCR（Gradient Coreset）潜特征版本。

    候选池用 reservoir sampling 保存每类 pool_mult × quota 个真实潜特征；
    每轮在当前池上拟合 ridge 线性代理头（one-hot 回归），样本梯度内积为
        <g_i, g_j> = (h_i · h_j) · (r_i · r_j),
    按类做贪心 facility-location 覆盖，选出 quota 个真实样本作为代表。
    代表特征始终是池中的真实潜特征，不做合成或梯度优化。

    论文原方法面向持续学习；此处接入的是潜特征版本（无原始图像/像素梯度）。
    """

    def __init__(self, features, labels, quotas, dim=1024, seed=0,
                 pool_mult=4.0, ridge=1.0, pool=None, pool_seen=None,
                 select=True):
        quotas = [int(q) for q in quotas]
        dim = int(dim)
        rep_labels = torch.cat(
            [torch.full((q,), c, dtype=torch.long)
             for c, q in enumerate(quotas)])
        # 先占位初始化基类字段（代表特征随后由梯度覆盖选择给出）
        super().__init__(torch.zeros(int(sum(quotas)), dim), rep_labels,
                         quotas, dim, seed)
        self.pool_mult = float(pool_mult)
        self.ridge = float(ridge)
        self.pool_caps = [max(1, int(round(self.pool_mult * q)))
                          for q in self.quotas]
        if pool is None:
            self.pool = [torch.zeros(0, dim) for _ in self.quotas]
            self.pool_seen = np.zeros(self.num_classes, dtype=np.int64)
            self._offer_pool(torch.as_tensor(features).float().cpu(),
                             torch.as_tensor(labels).long().cpu())
        else:
            if len(pool) != self.num_classes:
                raise ValueError("候选池类别数与配额不一致")
            self.pool = [torch.as_tensor(p).float().cpu().clone()
                         for p in pool]
            self.pool_seen = np.asarray(pool_seen, dtype=np.int64).copy()
        if select:
            self._reselect()

    # ------------------------------------------------------------------
    def _pool_add(self, c, f):
        cap = self.pool_caps[c]
        self.pool_seen[c] += 1
        buf = self.pool[c]
        if len(buf) < cap:
            self.pool[c] = torch.cat([buf, f.view(1, -1)], 0)
            return
        j = int(self.rng.randint(0, int(self.pool_seen[c])))
        if j < cap:
            buf[j] = f

    def _offer_pool(self, features, labels):
        for f, y in zip(features, labels):
            self._pool_add(int(y), f)

    def _grad_similarity(self):
        """返回 (S, blocks)：S 为样本梯度内积，blocks=[(类, 池大小)]。"""
        blocks = [(c, len(p)) for c, p in enumerate(self.pool) if len(p) > 0]
        H = torch.cat([self.pool[c] for c, _ in blocks], 0)
        y = torch.cat([torch.full((m,), c, dtype=torch.long)
                       for c, m in blocks], 0)
        Y = torch.nn.functional.one_hot(y, self.num_classes).float()
        n, d = H.shape
        lam = max(float(self.ridge), 1e-6)
        Ht = H.t()
        if d <= n:
            A = H.t() @ H + lam * torch.eye(d)
            W = torch.linalg.solve(A, H.t() @ Y)
            P = H @ W
        else:
            K = H @ H.t()
            alpha = torch.linalg.solve(K + lam * torch.eye(n), Y)
            P = K @ alpha
            del K
        R = P - Y
        # 分块计算 S = K ⊙ (R Rᵀ)，避免同时物化两个 n×n 矩阵
        # （全池 n=9800 时旧写法峰值约 6GB，分块后只需一份 S）
        Rt = R.t()
        S = torch.empty((n, n), dtype=H.dtype, device=H.device)
        step = max(1, min(n, 1024))
        for i in range(0, n, step):
            j = min(i + step, n)
            S[i:j] = (H[i:j] @ Ht) * (R[i:j] @ Rt)
        return S, blocks

    @staticmethod
    def _facility_location(S, k):
        m = int(S.size(0))
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        if k >= m:
            return torch.arange(m)
        # 标准 facility location 从零覆盖开始；用 -inf 会让首个点退化为
        # “按行内积和选取”，而非最大增益
        cov = torch.zeros(m, dtype=S.dtype, device=S.device)
        chosen = torch.zeros(m, dtype=torch.bool)
        sel = []
        for _ in range(k):
            gains = torch.maximum(cov.unsqueeze(0), S).sum(1)
            gains[chosen] = -float("inf")
            i = int(torch.argmax(gains))
            sel.append(i)
            chosen[i] = True
            cov = torch.maximum(cov, S[i])
        return torch.tensor(sel, dtype=torch.long)

    def _reselect(self, allowed_idx=None, classes=None):
        S, blocks = self._grad_similarity()
        allowed = self._allowed_mask(allowed_idx) \
            if allowed_idx is not None else None
        off, n_selected = 0, 0
        for c, m in blocks:
            a, b = self.slices[c]
            if classes is not None and c not in classes:
                off += m
                continue
            if allowed is None:
                slots = torch.arange(a, b)
            else:
                slots = torch.where(allowed[a:b])[0] + a
            if len(slots) == 0:
                off += m
                continue
            k = min(int(len(slots)), m)
            sel = self._facility_location(S[off:off + m, off:off + m], k)
            self.features[slots[:k]] = self.pool[c][sel]
            n_selected += k
            off += m
        return n_selected

    # ------------------------------------------------------------------
    def align(self, alignment):
        super().align(alignment)
        self.pool = [alignment.apply(p).float().cpu() if len(p) > 0 else p
                     for p in self.pool]
        return self

    def update(self, features, labels, round_idx=0, allowed_idx=None):
        t0 = time.time()
        features = torch.as_tensor(features).float().cpu()
        labels = torch.as_tensor(labels).long().cpu()
        for f, y in zip(features, labels):
            self._pool_add(int(y), f)
        present = sorted(set(int(y) for y in labels))
        n_selected = self._reselect(allowed_idx, classes=set(present))
        return {"features": self.features.clone(),
                "info": {"method": "gcr", "round": int(round_idx),
                         "n_new": int(len(labels)),
                         "n_pool": int(sum(len(p) for p in self.pool)),
                         "pool_sizes": [len(p) for p in self.pool],
                         "n_selected": int(n_selected),
                         "sec": time.time() - t0}}

    # ------------------------------------------------------------------
    def state_dict(self):
        st = super().state_dict()
        st.update({"pool": [p.clone() for p in self.pool],
                   "pool_seen": self.pool_seen.copy(),
                   "pool_caps": list(self.pool_caps),
                   "pool_mult": self.pool_mult, "ridge": self.ridge})
        return st

    @classmethod
    def from_state_dict(cls, st, kind="gcr"):
        obj = cls(torch.zeros(0), torch.zeros(0, dtype=torch.long),
                  st["quotas"], dim=st["dim"], seed=st["seed"],
                  pool_mult=st.get("pool_mult", 4.0),
                  ridge=st.get("ridge", 1.0),
                  pool=st["pool"], pool_seen=st["pool_seen"], select=False)
        if "pool_caps" in st:
            obj.pool_caps = [int(x) for x in st["pool_caps"]]
        obj.features = st["features"].clone()
        obj.labels = st["labels"].clone()
        obj.rng.set_state(st["rng_state"])
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
    if kind == "gcr":
        return GCRCoreset.from_state_dict(st, kind="gcr")
    raise ValueError(f"未知 coreset: {kind}")
