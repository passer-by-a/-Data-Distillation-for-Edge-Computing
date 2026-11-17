"""Batch Distribution Matching (VAE-BDM, v2)

原版 DM：固定随机 MLP 特征空间中的逐类均值匹配。

固定协议：
  - 每类一个 nn.Parameter + 每类独立 AdamW；非当前类完全不更新
  - balanced 更新：每轮随机排列全部类别，每个类更新次数严格等于 updates_per_class
  - 不使用 random_class / 单张量优化器 / 全局 anchor

S 是可直接优化的张量参数；标签固定，VAE 冻结。
"""
import copy
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from random_feature import RandomMLP
from .cloud import quick_val_acc


class BatchDM:
    def __init__(self, x, y, quotas, num_classes, cfg, seed=42,
                 init="kmeans", device="cuda", val=None):
        # ---------- 输入检查 ----------
        if x.dim() != 2:
            raise ValueError(f"x 必须是 [N,d]，实际 {tuple(x.shape)}")
        dim = int(cfg.get("feature_dim", 1024))
        if x.size(1) != dim:
            raise ValueError(f"特征维度必须是 {dim}，实际 {x.size(1)}")
        if len(y) != len(x):
            raise ValueError("x 与 y 长度不一致")
        if not torch.isfinite(x).all():
            raise ValueError("x 含 NaN/Inf")
        if int(y.min()) < 0 or int(y.max()) >= num_classes:
            raise ValueError("y 标签越界")
        if "updates_per_class" not in cfg:
            raise ValueError("v2 协议必须提供 updates_per_class（旧 random_class 已移除）")
        quotas = [int(q) for q in quotas]
        if len(quotas) != num_classes:
            raise ValueError("quotas 长度必须等于 num_classes")
        if any(q <= 0 for q in quotas):
            raise ValueError("每个类别的配额必须为正")
        for c in range(num_classes):
            n_c = int((y == c).sum())
            if n_c == 0:
                raise ValueError(f"类别 {c} 没有训练样本")
            if quotas[c] > n_c:
                raise ValueError(f"类别 {c} 配额 {quotas[c]} 超过样本数 {n_c}")

        self.x = x.float().cpu()
        self.y = y.long().cpu().numpy()
        self.dim = dim
        self.quotas = quotas
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = torch.device(device)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.val = val
        self.cls_idx = [np.where(self.y == c)[0] for c in range(num_classes)]
        self.slices = {}

        self.updates_per_class = int(cfg["updates_per_class"])
        self.log_every = max(1, int(cfg.get("log_every", 50)))
        self.val_head = cfg.get("val_head", "mlp")
        self.val_epochs = int(cfg.get("val_epochs", 50))
        self.val_seed = int(cfg.get("val_seed", 0))
        self.selection = cfg.get("selection", "val")  # val / final
        self.save_every = int(cfg.get("save_every", 0))
        # 原版 DM：固定随机 MLP 特征空间中的均值匹配
        self.feature_net = RandomMLP(
            self.dim,
            hidden_dim=int(cfg.get("feature_hidden", 2048)),
            output_dim=int(cfg.get("feature_out", 1024)),
            seed=int(cfg.get("feature_seed", 0))).to(self.device)
        self.feat_cache = {}

        # ---------- 初始化代表特征 ----------
        reps, labels = [], []
        off = 0
        t0 = time.time()
        for c in range(num_classes):
            q = self.quotas[c]
            idx = self.cls_idx[c]
            if init == "random":
                sel = self.rng.choice(idx, q, replace=False)
                z = self.x[sel].clone()
            elif init == "kmeans":
                km = KMeans(n_clusters=q,
                            random_state=cfg.get("kmeans_seed", 42) + seed,
                            n_init=cfg.get("kmeans_n_init", 1),
                            max_iter=cfg.get("kmeans_max_iter", 100))
                km.fit(self.x[idx].numpy())
                z = torch.from_numpy(km.cluster_centers_).float()
            else:
                raise ValueError(init)
            reps.append(z)
            labels += [c] * q
            self.slices[c] = (off, off + q)
            off += q
        self.init_sec = time.time() - t0
        self.S0_by_class = [z.to(self.device).float() for z in reps]
        self.y_syn = torch.tensor(labels, dtype=torch.long, device=self.device)

        self.S_by_class = torch.nn.ParameterList(
            [torch.nn.Parameter(z.clone()) for z in self.S0_by_class])
        self.optimizers = [
            torch.optim.AdamW([p], lr=float(cfg.get("lr", 0.01)),
                              weight_decay=float(cfg.get("weight_decay", 0.0)))
            for p in self.S_by_class]

        # ---------- 审计状态 ----------
        self.logs = []
        self.class_update_counts = np.zeros(num_classes, dtype=np.int64)
        self.total_optimizer_steps = 0
        self._bal_order = None
        self._bal_pos = 0

    # ============================================================
    # 参数访问
    # ============================================================
    def get_all_features(self):
        """统一输出接口: [M,1024]"""
        return torch.cat([p for p in self.S_by_class], 0).detach()

    # ============================================================
    # 单类损失
    # ============================================================
    def _class_feat(self, c):
        """整类样本经固定随机网络后的特征，惰性缓存一次。

        feature_net 无梯度且逐样本独立，因此对批次做前向与先整类前向再取子集等价，
        可把每个 step 的 256 次前向降为每类一次。
        """
        if c not in self.feat_cache:
            with torch.no_grad():
                xc = self.x[self.cls_idx[c]].to(self.device).float()
                self.feat_cache[c] = self.feature_net(xc)
        return self.feat_cache[c]

    def _losses(self, c):
        idx = self.cls_idx[c]
        n = min(int(self.cfg.get("real_batch_size", 256)), len(idx))
        pos = self.rng.choice(len(idx), n, replace=False)
        zs = self.S_by_class[c]

        # 原版 DM：固定随机特征空间中的均值匹配
        real_feat = self._class_feat(c)[torch.as_tensor(
            pos, device=self.device)]
        l_mean = (real_feat.mean(0) - self.feature_net(zs).mean(0)
                  ).pow(2).mean()

        loss = self.cfg.get("lambda_mean", 1.0) * l_mean
        stats = {"mean": l_mean.detach(),
                 "syn_norm": zs.detach().norm(dim=1).mean()}
        return loss, stats

    # ============================================================
    # 优化一步
    # ============================================================
    def _step(self, c):
        loss, stats = self._losses(c)
        opt = self.optimizers[c]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.S_by_class[c]], 1.0)
        opt.step()
        self.class_update_counts[c] += 1
        self.total_optimizer_steps += 1
        return loss, stats

    # ============================================================
    # 训练
    # ============================================================
    def fit(self, val_x=None, val_y=None, verbose=True):
        cfg = self.cfg
        eval_every = max(1, int(cfg.get("eval_every", 500)))
        best_acc, best_S, best_step = -1.0, None, 0
        peak = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()

        def _current_features():
            return self.get_all_features().clone()

        def _maybe_validate(step):
            nonlocal best_acc, best_S, best_step
            if self.selection == "final" or val_x is None:
                return
            acc = quick_val_acc(_current_features(), self.y_syn, val_x, val_y,
                                num_classes=self.num_classes,
                                epochs=self.val_epochs, seed=self.val_seed,
                                device=str(self.device), head=self.val_head)
            if acc > best_acc:
                best_acc, best_step = float(acc), step
                best_S = _current_features()
            if verbose:
                print(f"    [val] step {step} acc {acc:.2f}% (best {best_acc:.2f})",
                      flush=True)

        step = int(self.total_optimizer_steps)

        def _register(loss, stats):
            need_record = (step % self.log_every == 0 or step == 1)
            need_print = verbose and (step % 200 == 0 or step == 1)
            if need_record or need_print:
                rec = self._record(step, loss, stats)
                if need_record:
                    self.logs.append(rec)
                if need_print:
                    print(f"  iter {step:>4} loss {rec['loss']:.5f} "
                          f"mean {rec['mean']:.5f}", flush=True)

        # balanced：每轮随机排列全部类别；断点续训时按已完成的每类次数继续
        while int(self.class_update_counts.min()) < self.updates_per_class:
            if self._bal_order is None or self._bal_pos >= self.num_classes:
                self._bal_order = self.rng.permutation(self.num_classes)
                self._bal_pos = 0
            c = int(self._bal_order[self._bal_pos])
            self._bal_pos += 1
            if self.class_update_counts[c] >= self.updates_per_class:
                continue
            loss, stats = self._step(c)
            step += 1
            _register(loss, stats)
            if self.device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(self.device) / 1e9)
            if (self.selection != "final" and val_x is not None
                    and step % eval_every == 0):
                _maybe_validate(step)
            if self.save_every > 0 and step % self.save_every == 0:
                self.save_checkpoint(f"dm_ckpt_step{step}.pt")

        if self.selection != "final" and val_x is not None and best_S is None:
            _maybe_validate(step)
        if best_S is None:
            best_S = _current_features()
            if best_step == 0:
                best_step = step

        return {"features": best_S.cpu().float(), "labels": self.y_syn.cpu(),
                "init_sec": self.init_sec, "train_sec": time.time() - t0,
                "best_val_acc": best_acc, "best_step": int(best_step),
                "peak_gpu_gb": float(peak), "logs": self.logs,
                "class_update_counts": self.class_update_counts.tolist(),
                "total_optimizer_steps": int(self.total_optimizer_steps),
                "update_schedule": "balanced",
                "updates_per_class": self.updates_per_class,
                "per_class_optimizer": True,
                "val_head": self.val_head, "val_epochs": self.val_epochs,
                "val_seed": self.val_seed, "selection": self.selection}

    def _record(self, step, loss, stats):
        rec = {"iter": int(step), "loss": float(loss.detach())}
        rec.update({k: float(v) for k, v in stats.items()})
        return rec

    # ============================================================
    # checkpoint / 断点续训
    # ============================================================
    def state_dict(self):
        return {
            "params": [p.detach().cpu() for p in self.S_by_class],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "class_update_counts": self.class_update_counts.copy(),
            "total_optimizer_steps": int(self.total_optimizer_steps),
            "rng_state": self.rng.get_state(),
            "bal_order": (None if self._bal_order is None
                          else np.asarray(self._bal_order)),
            "bal_pos": int(self._bal_pos),
            "logs": copy.deepcopy(self.logs),
            "updates_per_class": self.updates_per_class,
        }

    def load_state_dict(self, sd):
        with torch.no_grad():
            for p, q in zip(self.S_by_class, sd["params"]):
                p.copy_(q.to(self.device).float())
        for o, osd in zip(self.optimizers, sd["optimizers"]):
            o.load_state_dict(osd)
        self.class_update_counts = np.asarray(sd["class_update_counts"]).copy()
        self.total_optimizer_steps = int(sd["total_optimizer_steps"])
        self.rng.set_state(sd["rng_state"])
        self._bal_order = (None if sd["bal_order"] is None
                           else np.asarray(sd["bal_order"]))
        self._bal_pos = int(sd["bal_pos"])
        self.logs = copy.deepcopy(sd.get("logs", []))

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        return path

    def load_checkpoint(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location,
                                        weights_only=False))
