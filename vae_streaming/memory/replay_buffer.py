"""边缘 Raw Replay Memory: 有界、类平衡 Reservoir Sampling

只保存原始图像、标签与所属轮次；不上传，只用于 VAE 持续训练、特征重编码与锚点对齐。
回放样本不会被重新计入统计量（统计接口只接收当前轮特征）。
"""
import numpy as np
import torch


class ClassBalancedReservoir:
    def __init__(self, capacity, num_classes, class_balanced=True, seed=0):
        if capacity <= 0:
            raise ValueError("capacity 必须为正")
        self.capacity = int(capacity)
        self.num_classes = int(num_classes)
        self.class_balanced = bool(class_balanced)
        self.rng = np.random.RandomState(seed)
        if class_balanced:
            base = capacity // num_classes
            rem = capacity % num_classes
            self.caps = [base + (1 if c < rem else 0)
                         for c in range(num_classes)]
        else:
            self.caps = [capacity] + [0] * (num_classes - 1)
        self.buffers = [[] for _ in range(num_classes)]
        self.seen = np.zeros(num_classes, dtype=np.int64)
        self.total_added = 0
        self.total_replaced = 0
        # 来源审计：base=Initial Base 真实标签；stream=流式样本（仅允许伪标签）
        self.sources = {"base": 0, "stream": 0}
        self.label_sources = {"ground_truth": 0, "initial_teacher_pseudo": 0}

    # ------------------------------------------------------------------
    def _bucket(self, c):
        return c if self.class_balanced else 0

    def add_batch(self, images, labels, round_idx=-1, source="base",
                  label_source="ground_truth"):
        """images: uint8 [n,3,32,32]（CPU）；labels: long [n]

        同时记录样本所属轮次，供 `get(exclude_rounds=...)` 排除仍留在
        VAE pending 中的轮次，避免同一批样本重复进入同一个训练 batch。

        source: "base"（Initial Base 真实标签）或 "stream"（流式样本）。
        流式样本只允许携带伪标签，写入真实标签会直接报错。
        """
        if source == "stream" and label_source == "ground_truth":
            raise RuntimeError(
                "禁止把流式真实标签写入 Replay（必须使用 initial_teacher_pseudo）")
        if source not in self.sources:
            raise ValueError(f"未知来源: {source}")
        images = images.cpu()
        labels = labels.cpu()
        round_idx = int(round_idx)
        self.sources[source] += int(len(labels))
        self.label_sources[label_source] = \
            self.label_sources.get(label_source, 0) + int(len(labels))
        for i in range(len(labels)):
            c = int(labels[i])
            if c < 0 or c >= self.num_classes:
                raise ValueError(f"标签越界: {c}")
            b = self._bucket(c)
            cap = self.caps[b]
            if cap <= 0:
                continue
            self.seen[b] += 1
            self.total_added += 1
            buf = self.buffers[b]
            if len(buf) < cap:
                buf.append((images[i].clone(), c, round_idx))
            else:
                j = int(self.rng.randint(0, int(self.seen[b])))
                if j < cap:
                    buf[j] = (images[i].clone(), c, round_idx)
                    self.total_replaced += 1

    # ------------------------------------------------------------------
    def get(self, exclude_rounds=None):
        """返回 (images, labels)；exclude_rounds 中的轮次不计入。"""
        excl = set(int(r) for r in exclude_rounds) if exclude_rounds else set()
        imgs, ys = [], []
        for buf in self.buffers:
            for img, y, r in buf:
                if r in excl:
                    continue
                imgs.append(img)
                ys.append(y)
        if not imgs:
            return (torch.empty(0, 3, 32, 32, dtype=torch.uint8),
                    torch.empty(0, dtype=torch.long))
        return torch.stack(imgs, 0), torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return sum(len(b) for b in self.buffers)

    def nbytes(self):
        imgs, ys = self.get()
        return int(imgs.numel() * imgs.element_size() +
                   ys.numel() * ys.element_size())

    def per_class_counts(self):
        return [len(b) for b in self.buffers]

    def assert_capacity(self):
        assert sum(len(b) for b in self.buffers) <= self.capacity
        for b, buf in enumerate(self.buffers):
            assert len(buf) <= self.caps[b]
        return True

    def provenance(self):
        return {"capacity": self.capacity, "size": len(self),
                "class_balanced": self.class_balanced,
                "per_class": self.per_class_counts(),
                "added_by_source": dict(self.sources),
                "added_by_label_source": dict(self.label_sources),
                "n_replaced": int(self.total_replaced)}

    def state_dict(self):
        imgs, ys = self.get()
        return {"images": imgs, "labels": ys, "seen": self.seen.copy(),
                "rng_state": self.rng.get_state(),
                "caps": list(self.caps), "capacity": self.capacity,
                "num_classes": self.num_classes,
                "class_balanced": self.class_balanced,
                "total_added": self.total_added,
                "total_replaced": self.total_replaced,
                "sources": dict(self.sources),
                "label_sources": dict(self.label_sources),
                "per_class": list(self.buffers)}

    @classmethod
    def from_state_dict(cls, st):
        obj = cls(st["capacity"], st["num_classes"],
                  class_balanced=st["class_balanced"], seed=0)
        obj.caps = list(st["caps"])
        obj.seen = np.asarray(st["seen"]).copy()
        obj.rng.set_state(st["rng_state"])
        obj.total_added = int(st["total_added"])
        obj.total_replaced = int(st["total_replaced"])
        obj.sources = dict(st.get("sources", {"base": obj.total_added,
                                              "stream": 0}))
        obj.label_sources = dict(st.get("label_sources",
                                        {"ground_truth": obj.total_added,
                                         "initial_teacher_pseudo": 0}))
        obj.buffers = []
        for b in st["per_class"]:
            # 兼容旧断点：早期条目只有 (image, label)，缺少轮次记为 -1
            obj.buffers.append([
                (e[0], int(e[1]), int(e[2]) if len(e) > 2 else -1) for e in b])
        return obj
