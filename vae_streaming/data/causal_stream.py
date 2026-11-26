"""严格因果的数据流: Initial Base Set + 逐轮 Streaming Set

约定:
  - 轮次编号 t = 0..T-1；base 为 t = -1。
  - 第 t 轮只能访问 base 与第 0..t 轮的数据。
  - 访问未来轮次会直接抛错，而不是静默返回。
  - 所有切分只使用训练集；验证集/测试集不进入任何轮次。

两种标签协议（label_mode）:
  supervised_stream     原始全监督流式协议；round(t) 返回 RoundBatch（含真实标签）
  unlabeled_after_base  仅 Initial Base 有真实标签；round(t) 返回 UnlabeledBatch，
                        不携带任何标签字段，流式真实标签在本对象上被密封为 -1
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from data import load_split

from .stream_scenarios import (allocate_round_counts, apply_covariate_shift,
                               covariate_levels, label_shift_blocks,
                               round_weights)

LABEL_MODES = ("supervised_stream", "unlabeled_after_base")


@dataclass
class RoundBatch:
    round: int
    images: torch.Tensor        # uint8 [n,3,32,32]
    labels: torch.Tensor        # long [n]
    indices: np.ndarray         # 原始训练集索引（审计用）
    domain_level: float = 0.0

    def __len__(self):
        return len(self.labels)


@dataclass
class LabeledBatch:
    """真实有标签数据：只允许用于 Initial Base 及其 Replay。"""
    round: int
    images: torch.Tensor        # uint8 [n,3,32,32]
    labels: torch.Tensor        # long [n] 真实标签
    indices: np.ndarray
    source: str = "initial_base"
    label_source: str = "ground_truth"

    def __len__(self):
        return len(self.labels)


@dataclass
class UnlabeledBatch:
    """流式无标签数据：没有、也不允许新增任何标签字段。

    sample_ids 只是稳定样本标识（原始训练集索引），不得用于回查真实类别；
    审计测试会校验对应位置的真实标签已被密封。
    """
    round: int
    images: torch.Tensor        # uint8 [n,3,32,32]
    sample_ids: np.ndarray
    domain_level: float = 0.0

    def __len__(self):
        return len(self.images)


class CausalStream:
    def __init__(self, dataset="cifar100", initial_fraction=0.20,
                 num_rounds=20, seed=42, class_balanced_initial=True,
                 scenario="iid", val_ratio=0.02, split_seed=42,
                 device="cpu", class_quota=None,
                 label_mode="supervised_stream"):
        if label_mode not in LABEL_MODES:
            raise ValueError(f"未知标签协议: {label_mode}")
        if label_mode == "unlabeled_after_base":
            if scenario != "iid":
                raise ValueError(
                    "unlabeled_after_base 只支持 iid（无标签流式协议）")
        if not 0.0 < initial_fraction < 1.0:
            raise ValueError("initial_fraction 必须在 (0,1)")
        if num_rounds < 1:
            raise ValueError("num_rounds 必须 >= 1")
        if class_quota is not None:
            if scenario != "iid":
                raise ValueError("class_quota 只支持 iid 场景")
            if not class_balanced_initial:
                raise ValueError("class_quota 需要 class_balanced_initial=True")
            class_quota = int(class_quota)
            if class_quota < 1:
                raise ValueError("class_quota 必须 >= 1")
        self.class_quota = class_quota
        self.dataset = dataset
        self.initial_fraction = float(initial_fraction)
        self.num_rounds = int(num_rounds)
        self.seed = int(seed)
        self.scenario = scenario
        self.device = device
        self.label_mode = label_mode

        split = load_split(dataset, val_ratio=val_ratio, seed=split_seed)
        self.train_images, self.train_labels = split["train"]
        self.val_images, self.val_labels = split["val"]
        self.test_images, self.test_labels = split["test"]
        self.num_classes = int(self.train_labels.max().item()) + 1
        self.split_seed = split_seed

        if label_mode == "unlabeled_after_base":
            self._build_unlabeled_stream(class_balanced_initial)
            # 审计用的原始类别计数在密封前缓存
            self._class_counts_total = np.bincount(
                self.train_labels.numpy(), minlength=self.num_classes)
            self._seal_stream_labels()
        else:
            self._build(class_balanced_initial)
            self._class_counts_total = np.bincount(
                self.train_labels.numpy(), minlength=self.num_classes)
        self._max_accessed = -1        # -1 = 只访问了 base
        self._base_accessed = False

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    def _build(self, class_balanced_initial):
        rng = np.random.RandomState(self.seed)
        labels = self.train_labels.numpy()
        self.class_indices = []
        self.base_indices = []
        rest_indices = []

        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]
            idx = idx[rng.permutation(len(idx))]
            if self.class_quota is not None:
                need = self.class_quota * self.num_rounds
                if len(idx) <= need:
                    raise ValueError(
                        f"类别 {c} 仅 {len(idx)} 张，无法满足 "
                        f"{self.num_rounds} 轮 × 每轮 {self.class_quota} 张")
                # 固定每类每轮配额：base 吸收类别总数之间的差异
                n0 = len(idx) - need
            elif class_balanced_initial:
                n0 = int(round(len(idx) * self.initial_fraction))
            else:
                n0 = None
            if n0 is None:
                # 非类平衡: 全局随机取前 initial_fraction
                n0 = 0
            self.class_indices.append(idx)
            self.base_indices.append(idx[:n0])
            rest_indices.append(idx[n0:])

        if not class_balanced_initial:
            total = len(labels)
            n0 = int(round(total * self.initial_fraction))
            perm = rng.permutation(total)
            base_set = set(perm[:n0].tolist())
            self.base_indices = [np.array([i for i in idx if i in base_set])
                                 for idx in self.class_indices]
            rest_indices = [np.array([i for i in idx if i not in base_set])
                            for idx in self.class_indices]

        self.base_indices = np.sort(np.concatenate(self.base_indices))

        if self.class_quota is None and self.scenario in ("iid", "covariate_shift"):
            # IID 下所有类的小数配额相同，按固定类顺序补足名额会让低编号类
            # 每轮多拿 1 张，末期提前耗尽（CIFAR-100 r79 只剩类 90–99）。
            # 改为：保留 base，剩余样本固定种子整体打乱一次后等分各轮。
            rest = np.concatenate(rest_indices)
            self.round_indices = self._even_round_split(rest, rng)
            self.round_class_counts = np.stack([
                np.bincount(labels[idx], minlength=self.num_classes)
                for idx in self.round_indices])
        else:
            sizes = np.array([len(x) for x in rest_indices], dtype=np.int64)
            weights = round_weights(self.num_classes, self.num_rounds,
                                    self.scenario, seed=self.seed)
            alloc = allocate_round_counts(sizes, self.num_rounds, weights)
            self.round_class_counts = alloc

            self.round_indices = []
            cursor = [0] * self.num_classes
            for r in range(self.num_rounds):
                parts = []
                for c in range(self.num_classes):
                    k = int(alloc[r, c])
                    if k:
                        parts.append(rest_indices[c][cursor[c]:cursor[c] + k])
                        cursor[c] += k
                self.round_indices.append(np.sort(np.concatenate(parts)))
            for c in range(self.num_classes):
                assert cursor[c] == len(rest_indices[c]), "类别剩余样本未耗尽"

        all_stream = np.concatenate(self.round_indices)
        assert len(all_stream) == len(set(all_stream.tolist())), "轮次间存在重复样本"
        assert len(set(self.base_indices.tolist()) & set(all_stream.tolist())) == 0

        self.levels = covariate_levels(self.num_rounds) \
            if self.scenario == "covariate_shift" else [0.0] * self.num_rounds
        self.blocks = label_shift_blocks(self.num_classes, self.num_rounds,
                                         self.seed) \
            if self.scenario == "label_shift" else []

        self._rest_indices = rest_indices

    def _even_round_split(self, rest, rng):
        """固定种子整体打乱剩余索引后等分为 num_rounds 轮。

        正式协议 39200 = 80 × 490（rem=0）；余数只在非整除的小配置下出现，
        平摊到前 rem 轮，保证总量与顺序不变。
        """
        stream_order = rest[rng.permutation(len(rest))]
        per = len(stream_order) // self.num_rounds
        rem = len(stream_order) - per * self.num_rounds
        sizes = [per + (1 if i < rem else 0) for i in range(self.num_rounds)]
        rounds, cursor = [], 0
        for k in sizes:
            rounds.append(stream_order[cursor:cursor + k].copy())
            cursor += k
        assert cursor == len(stream_order)
        return rounds

    def _build_unlabeled_stream(self, class_balanced_initial):
        """无标签协议的数据流构造。

        - base 仍按真实类别做类平衡抽样（base 允许使用真实标签）。
        - 流式部分只由固定随机种子对"剩余样本索引"做全局排列，再等分为
          各轮；不按真实类别分桶，顺序与流式真实标签无关。
        """
        rng = np.random.RandomState(self.seed)
        labels = self.train_labels.numpy()
        self.class_indices = []
        self.base_indices = []
        rest_indices = []

        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]
            idx = idx[rng.permutation(len(idx))]
            if self.class_quota is not None:
                need = self.class_quota * self.num_rounds
                if len(idx) <= need:
                    raise ValueError(
                        f"类别 {c} 仅 {len(idx)} 张，无法满足 "
                        f"{self.num_rounds} 轮 × 每轮 {self.class_quota} 张")
                n0 = len(idx) - need
            elif class_balanced_initial:
                n0 = int(round(len(idx) * self.initial_fraction))
            else:
                n0 = 0
            self.class_indices.append(idx)
            self.base_indices.append(idx[:n0])
            rest_indices.append(idx[n0:])

        if not class_balanced_initial:
            total = len(labels)
            n0 = int(round(total * self.initial_fraction))
            perm = rng.permutation(total)
            base_set = set(perm[:n0].tolist())
            self.base_indices = [np.array([i for i in idx if i in base_set])
                                 for idx in self.class_indices]
            rest_indices = [np.array([i for i in idx if i not in base_set])
                            for idx in self.class_indices]

        self.base_indices = np.sort(np.concatenate(self.base_indices))
        rest = np.concatenate(rest_indices)
        # 唯一决定流式顺序的随机源：固定种子 + 样本索引
        self.round_indices = self._even_round_split(rest, rng)
        # 流式轮次不按真实类别统计（避免任何类别分桶语义）
        self.round_class_counts = np.zeros(
            (self.num_rounds, self.num_classes), dtype=np.int64)

        all_stream = np.concatenate(self.round_indices)
        assert len(all_stream) == len(set(all_stream.tolist())), "轮次间存在重复样本"
        assert len(set(self.base_indices.tolist()) & set(all_stream.tolist())) == 0

        self.levels = [0.0] * self.num_rounds
        self.blocks = []
        self._rest_indices = rest_indices

    def _seal_stream_labels(self):
        """密封流式真实标签：base 之外的位置一律置为 -1。

        训练接口 round(t) 返回的 UnlabeledBatch 不含任何标签字段；此处的
        密封是第二道防线，保证即使误用 train_labels[stream_idx] 也拿不到真值。
        """
        stream_idx = np.concatenate(self.round_indices)
        self.base_labels = self.train_labels[self.base_indices].clone()
        sealed = torch.full_like(self.train_labels, -1)
        sealed[self.base_indices] = self.base_labels
        self.train_labels = sealed
        self._sealed_stream_indices = int(len(stream_idx))

    # ------------------------------------------------------------------
    # 访问
    # ------------------------------------------------------------------
    def _check(self, t):
        if t < 0 or t >= self.num_rounds:
            raise IndexError(f"轮次 {t} 越界 [0,{self.num_rounds})")
        if not self._base_accessed:
            raise RuntimeError("必须先访问 Initial Base Set 才能读取流式轮次")
        if t > self._max_accessed + 1:
            raise RuntimeError(
                f"因果性违规: 已访问到第 {self._max_accessed} 轮，"
                f"不能直接读取第 {t} 轮（禁止跳读未来数据）")
        self._max_accessed = max(self._max_accessed, t)

    def _pack(self, t, indices, level=0.0):
        idx = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        images = self.train_images[idx]
        if level > 0:
            images = apply_covariate_shift(images, level, device=self.device)
        if self.label_mode == "unlabeled_after_base":
            # 流式数据：只返回图像、样本ID、轮次；不存在 labels 字段
            return UnlabeledBatch(round=t, images=images,
                                  sample_ids=np.asarray(indices, dtype=np.int64),
                                  domain_level=float(level))
        return RoundBatch(round=t, images=images,
                          labels=self.train_labels[idx].clone(),
                          indices=np.asarray(indices, dtype=np.int64),
                          domain_level=float(level))

    @property
    def base(self):
        self._base_accessed = True
        if self.label_mode == "unlabeled_after_base":
            idx = np.asarray(self.base_indices, dtype=np.int64)
            return LabeledBatch(round=-1, images=self.train_images[
                torch.from_numpy(idx)], labels=self.base_labels.clone(),
                indices=idx, source="initial_base",
                label_source="ground_truth")
        return self._pack(-1, self.base_indices)

    def round(self, t):
        self._check(t)
        return self._pack(t, self.round_indices[t], self.levels[t])

    # ------------------------------------------------------------------
    # 评估域
    # ------------------------------------------------------------------
    def test_domain(self, level: float = 0.0):
        """测试集的域变换版本（仅供评估，绝不进入训练/记忆/蒸馏）"""
        images = self.test_images
        if level > 0:
            images = apply_covariate_shift(images, level, device=self.device)
        return images, self.test_labels

    def round_domain_level(self, t):
        return self.levels[t]

    def scenario_groups(self, t):
        """评估分组: label shift 下返回当前强调类 / 历史已强调类"""
        labels = self.test_labels.numpy()
        if self.scenario != "label_shift":
            return {"all": np.ones(len(labels), dtype=bool)}
        nb = len(self.blocks)
        cur = self.blocks[t % nb]
        m_cur = np.isin(labels, cur)
        if t == 0:
            m_hist = np.zeros(len(labels), dtype=bool)
        else:
            # 历史组 = 之前轮次强调过、且不属于当前块的类别；
            # 旧写法未排除当前块，t≥块数后 historical 退化为全测试集
            prev = np.setdiff1d(np.unique(np.concatenate(
                [self.blocks[r % nb] for r in range(t)])), cur)
            m_hist = np.isin(labels, prev)
        return {"current": m_cur, "historical": m_hist}

    # ------------------------------------------------------------------
    # 审计
    # ------------------------------------------------------------------
    def stream_indices_sha1(self):
        """流式数据流指纹（按轮次、按顺序）"""
        h = hashlib.sha1()
        for r, idx in enumerate(self.round_indices):
            h.update(np.asarray(idx, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def base_indices_sha1(self):
        return hashlib.sha1(
            np.asarray(self.base_indices, dtype=np.int64).tobytes()
        ).hexdigest()[:16]

    def summary(self):
        counts = self._class_counts_total
        return {
            "dataset": self.dataset,
            "label_mode": self.label_mode,
            "streaming_labels_sealed": self.label_mode ==
            "unlabeled_after_base",
            "stream_indices_sha1": self.stream_indices_sha1(),
            "base_indices_sha1": self.base_indices_sha1(),
            "scenario": self.scenario,
            "num_classes": self.num_classes,
            "n_train": int(len(self.train_labels)),
            "n_val": int(len(self.val_labels)),
            "n_test": int(len(self.test_labels)),
            "initial_fraction": self.initial_fraction,
            "n_base": int(len(self.base_indices)),
            "num_rounds": self.num_rounds,
            "class_quota": self.class_quota,
            "round_sizes": [int(len(x)) for x in self.round_indices],
            "class_counts_total": counts.tolist(),
            "base_class_counts": np.bincount(
                self.train_labels[self.base_indices].numpy(),
                minlength=self.num_classes).tolist(),
            "domain_levels": list(self.levels),
            "label_shift_blocks": [b.tolist() for b in self.blocks],
            "split_seed": self.split_seed,
            "shuffle_seed": self.seed,
        }

    def assert_no_leakage(self, t, indices):
        """校验给定索引只来自 base 与第 0..t 轮"""
        allowed = set(self.base_indices.tolist())
        for r in range(t + 1):
            allowed |= set(self.round_indices[r].tolist())
        bad = set(np.asarray(indices).tolist()) - allowed
        if bad:
            raise RuntimeError(f"检测到 {len(bad)} 个越界训练索引")
        return True


def regroup_stream_rounds(stream, num_rounds):
    """把已有流式序列按连续样本重新等分，保持数据、顺序与总量完全不变。

    用于"同一批流式数据、不同轮粒度"的对照实验（如 392 轮 100 张/轮 →
    40 轮 980 张/轮）。base 划分、验证/测试集不受影响。
    """
    flat = np.concatenate(stream.round_indices)
    per = len(flat) // int(num_rounds)
    if per * int(num_rounds) != len(flat):
        raise ValueError(
            f"流式样本数 {len(flat)} 无法等分为 {num_rounds} 轮")
    stream.round_indices = [flat[i * per:(i + 1) * per].copy()
                            for i in range(int(num_rounds))]
    stream.num_rounds = int(num_rounds)
    stream.levels = [0.0] * int(num_rounds)
    return stream
