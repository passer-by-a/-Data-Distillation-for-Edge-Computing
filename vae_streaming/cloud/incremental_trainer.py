"""云端持续训练器

- incremental: 保留上一轮模型与优化器状态，每轮重置余弦退火调度
- scratch    : 每轮从随机初始化重训（同预算参考）

云端只读取已同步的代表特征；不访问边缘原始图像、未来轮次或测试集标签。
"""
import copy
import time

import torch
import torch.nn.functional as F

from vae_bdm.cloud import build_cloud, evaluate_cloud_classifier


class IncrementalCloudLearner:
    def __init__(self, cfg, dim=1024, num_classes=10, device="cuda"):
        self.cfg = cfg["cloud"]
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.seed = int(self.cfg.get("seed", 0))
        self.head = self.cfg.get("head", "mlp")
        self.lr = float(self.cfg.get("lr", 1e-3))
        self.wd = float(self.cfg.get("wd", 1e-4))
        self.batch = int(self.cfg.get("batch", 256))
        self.steps_per_round = int(self.cfg.get("steps_per_round", 200))
        # >0 时按"每轮固定 epoch 数"训练，步数随当前同步集大小自动换算
        self.epochs_per_round = int(self.cfg.get("epochs_per_round", 0))
        self.model = None
        self.opt = None
        self.steps_done = 0
        self.total_train_sec = 0.0
        self.history = []
        self.round = -1

    # ------------------------------------------------------------------
    def _build(self, seed):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() \
            if torch.cuda.is_available() else None
        try:
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            model = build_cloud(self.head, self.dim, self.num_classes).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                    weight_decay=self.wd)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
        return model, opt

    def _ensure(self):
        if self.model is None:
            self.model, self.opt = self._build(self.seed)

    # ------------------------------------------------------------------
    def update(self, features, labels, val_features=None, val_labels=None,
               round_idx=0, mode="incremental", on_epoch_end=None):
        self._ensure()
        if mode == "scratch":
            self.model, self.opt = self._build(self.seed + int(round_idx))
        elif mode != "incremental":
            raise ValueError(f"未知云端训练模式: {mode}")

        x = torch.as_tensor(features).float().cpu()
        y = torch.as_tensor(labels).long().cpu()
        if len(x) == 0:
            raise ValueError("云端代表集为空")
        xv = None if val_features is None else \
            torch.as_tensor(val_features).float().cpu()
        yv = None if val_labels is None else \
            torch.as_tensor(val_labels).long().cpu()

        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() \
            if torch.cuda.is_available() else None
        try:
            # RNG 隔离: Dropout 等随机性只由 (seed, round) 决定，便于断点续跑
            torch.manual_seed(self.seed + 104729 * int(round_idx))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed + 104729 * int(round_idx))

            g = torch.Generator().manual_seed(self.seed + 7919 * int(round_idx))
            # 同步集足够小时整体驻留 GPU，避免每步重复 H2D 拷贝
            dev_x = dev_y = dev_xv = None
            if self.device.type == "cuda":
                max_bytes = 256 * 1024 ** 2
                if x.numel() * x.element_size() <= max_bytes:
                    dev_x, dev_y = x.to(self.device), y.to(self.device)
                if (xv is not None and xv.numel() * xv.element_size()
                        <= max_bytes):
                    dev_xv = xv.to(self.device)
            batches, epoch_ends = None, []
            if self.epochs_per_round > 0:
                # 固定 5 个 epoch：每轮完整遍历当前同步集，最后不足一个
                # batch 的尾批也参与更新；epoch_ends 记录每个 epoch 的结束步
                batches = []
                for _ in range(self.epochs_per_round):
                    perm = torch.randperm(len(x), generator=g)
                    for i in range(0, len(x), self.batch):
                        batches.append(perm[i:i + self.batch])
                    epoch_ends.append(len(batches))
                steps = len(batches)
            else:
                steps = self.steps_per_round
            # 重置学习率调度（余弦退火到 0），但保留模型与优化器状态
            for pg in self.opt.param_groups:
                pg["lr"] = self.lr
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,
                                                             max(1, steps))
            best_acc, best_state, best_step = -1.0, None, 0
            t0 = time.time()
            val_every = max(1, steps // 5)

            def _validate(step):
                nonlocal best_acc, best_state, best_step
                if xv is None or len(xv) == 0:
                    return
                self.model.eval()
                with torch.no_grad():
                    pred = []
                    for i in range(0, len(xv), 512):
                        xvb = (dev_xv[i:i + 512] if dev_xv is not None
                               else xv[i:i + 512].to(self.device))
                        pred.append(self.model(xvb).argmax(1).cpu())
                    acc = (torch.cat(pred) == yv).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_step = int(step)

            self.model.train()
            loss_sum, n_loss = 0.0, 0
            ep_sum, ep_n = 0.0, 0
            epoch_losses = []
            ep_pos = 0
            next_ep_end = epoch_ends[0] if epoch_ends else -1
            for step in range(1, steps + 1):
                if batches is None:
                    idx = torch.randint(0, len(x), (min(self.batch, len(x)),),
                                        generator=g)
                else:
                    idx = batches[step - 1]
                xb = dev_x[idx] if dev_x is not None else x[idx].to(self.device)
                yb = dev_y[idx] if dev_y is not None else y[idx].to(self.device)
                loss = F.cross_entropy(self.model(xb), yb)
                loss_sum += float(loss.item()) * len(xb)
                n_loss += len(xb)
                ep_sum += float(loss.item()) * len(xb)
                ep_n += len(xb)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                sch.step()
                if step == next_ep_end:
                    # 该 epoch 的训练 CE（按样本加权）；回调可做逐 epoch 只读评估
                    epoch_losses.append(ep_sum / max(1, ep_n))
                    if on_epoch_end is not None:
                        on_epoch_end(int(ep_pos), float(epoch_losses[-1]))
                        self.model.train()
                    ep_sum, ep_n = 0.0, 0
                    ep_pos += 1
                    next_ep_end = (epoch_ends[ep_pos]
                                   if ep_pos < len(epoch_ends) else -1)
                if step % val_every == 0 or step == steps:
                    _validate(step)
                    self.model.train()
            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.model.eval()
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)

        sec = time.time() - t0
        self.total_train_sec += sec
        self.steps_done += steps
        self.round = int(round_idx)
        info = {"mode": mode, "round": int(round_idx),
                "n_train": int(len(x)),
                "steps": int(steps),
                "epochs_per_round": int(self.epochs_per_round),
                "train_sec": sec,
                "total_train_sec": self.total_train_sec,
                "best_val_acc": None if best_acc < 0 else float(best_acc),
                "best_step": int(best_step),
                "train_loss": float(loss_sum / max(1, n_loss)),
                "epoch_train_loss": epoch_losses,
                "optimizer_steps_total": int(self.steps_done),
                "head": self.head,
                "param_norm": float(sum(p.detach().pow(2).sum()
                                        for p in self.model.parameters()) ** 0.5)}
        self.history.append(info)
        return info

    # ------------------------------------------------------------------
    def evaluate(self, features, labels):
        out = evaluate_cloud_classifier(self.model, features, labels,
                                        self.num_classes)
        out["loss"] = self._cross_entropy(features, labels)
        return out

    @torch.no_grad()
    def _cross_entropy(self, features, labels, batch=512):
        """当前云端分类器在给定特征上的平均交叉熵（评估用）。"""
        dev = next(self.model.parameters()).device
        x = features.float()
        y = labels.long()
        total, n = 0.0, 0
        self.model.eval()
        for i in range(0, x.size(0), batch):
            xb = x[i:i + batch].to(dev)
            yb = y[i:i + batch].to(dev)
            total += float(F.cross_entropy(self.model(xb), yb,
                                           reduction="sum").item())
            n += int(yb.numel())
        return total / max(1, n)

    def state_dict(self):
        return {"model": copy.deepcopy(self.model.state_dict())
                if self.model is not None else None,
                "optimizer": copy.deepcopy(self.opt.state_dict())
                if self.opt is not None else None,
                "steps_done": self.steps_done,
                "total_train_sec": self.total_train_sec,
                "history": copy.deepcopy(self.history),
                "round": self.round, "seed": self.seed}

    def load_state_dict(self, st):
        self.model, self.opt = self._build(self.seed)
        if st["model"] is not None:
            self.model.load_state_dict(st["model"])
        if st["optimizer"] is not None:
            self.opt.load_state_dict(st["optimizer"])
        self.steps_done = int(st["steps_done"])
        self.total_train_sec = float(st["total_train_sec"])
        self.history = copy.deepcopy(st["history"])
        self.round = int(st["round"])
        return self
