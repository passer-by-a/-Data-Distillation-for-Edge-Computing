"""Streaming Representative Memory: 有界 1024 维代表特征集

每个代表特征携带: 稳定 ID / 类别 / 特征 / Encoder 版本 / 最近更新轮次。
云端同步视图单独保存为 submitted，用于逐轮校验边缘—云一致。
"""
import copy

import numpy as np
import torch
from sklearn.cluster import KMeans

from vae_bdm.baselines import allocate_quotas


def _select_class(z, q, method, seed, cfg):
    """返回 (代表特征, 每个代表对应的 Base 簇大小)。

    簇大小用于 Online K-means 的逐质心计数初始化；random 初始化没有聚类
    语义，每个代表视为来自 1 个样本。
    """
    z = z.float().cpu()
    if q > len(z):
        raise ValueError(f"类别样本不足: 需要 {q}, 实际 {len(z)}")
    if method == "random":
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(z), q, replace=False)
        return z[sel], np.ones(q, dtype=np.int64)
    if method == "kmeans":
        km = KMeans(n_clusters=q, random_state=seed, n_init=1, max_iter=100)
        km.fit(z.numpy())
        # 保留真实簇大小：空簇记为 0，由下游按空槽处理（ExStream 优先填空槽，
        # Online K-means 的 1/count 学习率在内部对 0 取 max(·,1)）。
        sizes = np.bincount(km.labels_, minlength=q)
        return (torch.from_numpy(km.cluster_centers_).float(),
                sizes.astype(np.int64))
    raise ValueError(f"未知初始化方法: {method}")


def initialize_representatives(x, y, quotas, method="kmeans", seed=42,
                               device="cpu", cfg=None):
    """在 Initial Base Set 特征上初始化代表集。

    返回 (features [M,D], labels [M], cluster_counts [M])，类别顺序与
    quotas 一致；cluster_counts 是该代表在 Base 聚类中聚合的样本数。
    """
    x = torch.as_tensor(x).float().cpu()
    y = torch.as_tensor(y).long().cpu()
    quotas = [int(q) for q in quotas]

    if method == "bdm":
        from vae_bdm.bdm import BatchDM
        dcfg = dict(cfg or {})
        dcfg.setdefault("feature_dim", x.size(1))
        dcfg.setdefault("updates_per_class", 20)
        dcfg.setdefault("lr", 0.01)
        dcfg.setdefault("weight_decay", 0.0)
        dcfg.setdefault("lambda_mean", 1.0)
        dcfg.setdefault("selection", "final")
        dcfg.setdefault("log_every", 1000)
        dcfg.setdefault("eval_every", 1000)
        dcfg.setdefault("kmeans_seed", seed)
        dm = BatchDM(x, y, quotas, len(quotas), dcfg, seed=seed,
                     init="kmeans", device=device)
        out = dm.fit(verbose=False)
        reps, labels = out["features"], out["labels"]
        order = torch.argsort(labels, stable=True)
        # BDM 初始化的簇大小未知，按每个代表 1 个样本计
        return (reps[order].float().cpu(), labels[order].long(),
                np.ones(len(reps), dtype=np.int64))

    reps, labels, counts = [], [], []
    for c, q in enumerate(quotas):
        z = x[y == c]
        rep, cnt = _select_class(z, q, method, seed + c, cfg)
        reps.append(rep)
        counts.append(cnt)
        labels += [c] * q
    return (torch.cat(reps, 0).float(), torch.tensor(labels, dtype=torch.long),
            np.concatenate(counts).astype(np.int64))


class RepresentativeMemory:
    def __init__(self, budget, num_classes, quotas=None, dim=1024,
                 base_labels=None):
        self.budget = int(budget)
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        if quotas is None:
            if base_labels is None:
                raise ValueError("需要 quotas 或 base_labels")
            quotas = allocate_quotas(np.asarray(base_labels), self.budget,
                                     self.num_classes)
        quotas = np.asarray(quotas, dtype=np.int64)
        if int(quotas.sum()) != self.budget:
            raise ValueError(f"配额总和 {quotas.sum()} != 预算 {self.budget}")
        self.quotas = quotas
        self.slices = {}
        off = 0
        for c, q in enumerate(quotas):
            self.slices[c] = (off, off + int(q))
            off += int(q)
        self.labels = torch.cat([torch.full((int(q),), c, dtype=torch.long)
                                 for c, q in enumerate(quotas)])
        self.ids = torch.arange(self.budget, dtype=torch.long)
        self.features = torch.zeros(self.budget, self.dim)
        self.submitted = torch.zeros(self.budget, self.dim)
        self.versions = torch.full((self.budget,), -1, dtype=torch.long)
        self.submitted_versions = torch.full((self.budget,), -1, dtype=torch.long)
        self.last_round = torch.full((self.budget,), -1, dtype=torch.long)
        # 每个代表在 Base 聚类中聚合的样本数（Online K-means 逐质心计数用）
        self.cluster_counts = np.ones(self.budget, dtype=np.int64)
        self.initialized = False

    # ------------------------------------------------------------------
    def initialize_from_features(self, x, y, method="kmeans", version=0,
                                 seed=42, device="cpu", cfg=None):
        feats, labels, counts = initialize_representatives(
            x, y, self.quotas, method=method, seed=seed, device=device, cfg=cfg)
        if not torch.equal(labels, self.labels):
            raise RuntimeError("初始化代表集的类别顺序与配额不一致")
        self.features = feats.float().cpu()
        self.cluster_counts = np.asarray(counts, dtype=np.int64)
        self.submitted = self.features.clone()
        self.versions.fill_(int(version))
        self.submitted_versions.fill_(int(version))
        self.last_round.fill_(-1)
        self.initialized = True
        return self

    def reinitialize_from_features(self, x, y, method="kmeans", version=0,
                                   round_idx=0, seed=42, device="cpu", cfg=None):
        """Re-encoding Replay 模式: 用当前版本特征重建整个代表集"""
        self.initialize_from_features(x, y, method=method, version=version,
                                      seed=seed, device=device, cfg=cfg)
        self.last_round.fill_(int(round_idx))
        return self

    # ------------------------------------------------------------------
    def align(self, alignment, round_idx=None):
        """旧代表特征与云端已同步视图一起转换到新版本"""
        if not self.initialized:
            raise RuntimeError("代表集尚未初始化")
        if int(alignment.version_to) <= int(self.versions.max()):
            raise ValueError("对齐目标版本必须大于当前代表特征版本")
        f = alignment.apply(self.features)
        s = alignment.apply(self.submitted)
        self.features = f.float()
        self.submitted = s.float()
        self.versions.fill_(int(alignment.version_to))
        self.submitted_versions.fill_(int(alignment.version_to))
        if round_idx is not None:
            self.last_round.fill_(int(round_idx))
        return self

    def update_features(self, idx, values, version, round_idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        values = torch.as_tensor(values).float().cpu()
        if values.shape != (len(idx), self.dim):
            raise ValueError(f"更新特征形状错误: {tuple(values.shape)}")
        self.features[idx] = values
        self.versions[idx] = int(version)
        self.last_round[idx] = int(round_idx)
        return self

    def set_submitted(self, idx, values, version):
        idx = torch.as_tensor(idx, dtype=torch.long)
        self.submitted[idx] = torch.as_tensor(values).float().cpu()
        self.submitted_versions[idx] = int(version)
        return self

    # ------------------------------------------------------------------
    def class_indices(self, c):
        a, b = self.slices[c]
        return torch.arange(a, b)

    def cloud_view(self):
        return {"features": self.submitted.clone(),
                "labels": self.labels.clone(), "ids": self.ids.clone(),
                "versions": self.submitted_versions.clone()}

    def assert_budget(self):
        assert len(self.features) == self.budget
        assert len(self.labels) == self.budget
        # 维度由构造时的 dim 决定，不写死 1024（CIFAR-10/100 当前都是 1024）
        assert self.features.size(1) == self.dim
        return True

    def summary(self, stats=None):
        out = {"budget": self.budget, "dim": self.dim,
               "n_features": int(len(self.features)),
               "quotas": self.quotas.tolist(),
               "versions": sorted(set(self.versions.tolist())),
               "last_round": sorted(set(self.last_round.tolist())),
               "n_updated_last_round": None}
        if stats is not None:
            ratios, mean_d = [], []
            for c, (a, b) in self.slices.items():
                if stats.count[c] == 0 or b - a < 2:
                    continue
                v_rep = self.features[a:b].var(0, unbiased=False)
                _, mu_r, v_r = stats.get(c)
                ratios.append(float(v_rep.mean() / v_r.mean().clamp_min(1e-8)))
                mean_d.append(float((self.features[a:b].mean(0) - mu_r)
                                    .pow(2).mean().sqrt()))
            out["group_single_var_ratio"] = float(np.mean(ratios)) if ratios else None
            out["class_mean_distance"] = float(np.mean(mean_d)) if mean_d else None
        return out

    # ------------------------------------------------------------------
    def state_dict(self):
        return {"budget": self.budget, "num_classes": self.num_classes,
                "dim": self.dim, "quotas": self.quotas.copy(),
                "features": self.features.clone(),
                "submitted": self.submitted.clone(),
                "labels": self.labels.clone(), "ids": self.ids.clone(),
                "versions": self.versions.clone(),
                "submitted_versions": self.submitted_versions.clone(),
                "last_round": self.last_round.clone(),
                "cluster_counts": self.cluster_counts.copy(),
                "initialized": self.initialized, "slices": copy.deepcopy(self.slices)}

    @classmethod
    def from_state_dict(cls, st):
        obj = cls(st["budget"], st["num_classes"], quotas=st["quotas"],
                  dim=st["dim"])
        obj.features = st["features"].clone()
        obj.submitted = st["submitted"].clone()
        obj.labels = st["labels"].clone()
        obj.ids = st["ids"].clone()
        obj.versions = st["versions"].clone()
        obj.submitted_versions = st["submitted_versions"].clone()
        obj.last_round = st["last_round"].clone()
        # 旧 checkpoint 没有 cluster_counts：退回每代表 1 个样本
        obj.cluster_counts = np.asarray(
            st.get("cluster_counts", np.ones(obj.budget, dtype=np.int64)),
            dtype=np.int64)
        obj.initialized = bool(st["initialized"])
        obj.slices = copy.deepcopy(st["slices"])
        return obj
