"""边缘 Raw Replay Memory: 有界、类平衡 Reservoir Sampling

只保存原始图像与标签；不上传，只用于 VAE 持续训练、特征重编码与锚点对齐。
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

    # ------------------------------------------------------------------
    def _bucket(self, c):
        return c if self.class_balanced else 0

    def add_batch(self, images, labels):
        """images: uint8 [n,3,32,32]（CPU）；labels: long [n]"""
        images = images.cpu()
        labels = labels.cpu()
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
                buf.append((images[i].clone(), c))
            else:
                j = int(self.rng.randint(0, int(self.seen[b])))
                if j < cap:
                    buf[j] = (images[i].clone(), c)
                    self.total_replaced += 1

    # ------------------------------------------------------------------
    def get(self):
        imgs, ys = [], []
        for buf in self.buffers:
            for img, y in buf:
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

    def state_dict(self):
        imgs, ys = self.get()
        return {"images": imgs, "labels": ys, "seen": self.seen.copy(),
                "rng_state": self.rng.get_state(),
                "caps": list(self.caps), "capacity": self.capacity,
                "num_classes": self.num_classes,
                "class_balanced": self.class_balanced,
                "total_added": self.total_added,
                "total_replaced": self.total_replaced,
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
        obj.buffers = [list(b) for b in st["per_class"]]
        return obj
