"""四个云端分类器共享同一条边缘轨迹（在线配对实验）

每轮同步后，同一份 cloud.features / cloud.labels 同时交给四条分支：
  inc200     : 继承自身上一轮权重 + AdamW，训练 200 步（原协议基线）
  inc20      : 继承自身上一轮权重 + AdamW，训练 20 步
  scratch200 : 每轮随机初始化 MLP + 新 AdamW，训练 200 步
  scratch20  : 每轮随机初始化 MLP + 新 AdamW，训练 20 步

Base 只训练一次（200 步），两个增量分支从同一份模型与优化器状态分叉。
两个从头分支每轮由相同 seed 构建，初始权重逐位相同，轮次采样与 Dropout
随机序列也相同；四组均使用完整的余弦计划。
"""
import copy
import hashlib
import time

import torch
import torch.nn.functional as F

from vae_bdm.cloud import evaluate_cloud_classifier

from .incremental_trainer import IncrementalCloudLearner


BRANCHES = ("inc200", "inc20", "scratch200", "scratch20")


def _module_sha1(module):
    h = hashlib.sha1()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _metrics(model, features, labels, num_classes, device):
    out = evaluate_cloud_classifier(model, features, labels, num_classes)
    model.eval()
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(features), 512):
            xb = torch.as_tensor(features[i:i + 512]).float().to(device)
            yb = torch.as_tensor(labels[i:i + 512]).long().to(device)
            nll_sum += F.cross_entropy(model(xb), yb,
                                       reduction="sum").item()
            n += int(len(yb))
    return {"acc": float(out["acc"]), "f1": float(out["f1"]),
            "nll": float(nll_sum / max(n, 1))}


class MultiBranchCloudLearner:
    def __init__(self, cfg, dim=1024, num_classes=10, device="cuda"):
        cloud = cfg["cloud"]
        b = cloud.get("branches", {}) or {}
        self.steps = {
            "inc200": int(b.get("inc200_steps",
                                cloud.get("steps_per_round", 200))),
            "inc20": int(b.get("inc20_steps",
                               cloud.get("short_steps", 20))),
            "scratch200": int(b.get("scratch200_steps",
                                    b.get("inc200_steps",
                                          cloud.get("steps_per_round", 200)))),
            "scratch20": int(b.get("scratch20_steps",
                                   b.get("inc20_steps",
                                         cloud.get("short_steps", 20)))),
        }
        self.cfgs = {}
        for name, steps in self.steps.items():
            c = copy.deepcopy(cfg)
            c["cloud"]["steps_per_round"] = int(steps)
            # 多分支显式按步数区分分支，不能被 epochs 模式覆盖
            c["cloud"].pop("epochs_per_round", None)
            c["cloud"]["incremental_training"] = True
            self.cfgs[name] = c

        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.inc200 = IncrementalCloudLearner(
            self.cfgs["inc200"], dim, num_classes, device=device)
        self.inc20 = IncrementalCloudLearner(
            self.cfgs["inc20"], dim, num_classes, device=device)
        self.scratch200 = None
        self.scratch20 = None
        self._base_ready = False
        self._init_same = True
        self._init_checked = False

        self.history = []
        self.round = -1
        self.steps_done = 0
        self.total_train_sec = 0.0
        self.last_branch_metrics = None

    @property
    def model(self):
        return self.inc200.model

    # ------------------------------------------------------------------
    def update(self, features, labels, val_features=None, val_labels=None,
               round_idx=0, mode="incremental"):
        if val_features is not None or val_labels is not None:
            raise ValueError(
                "multi_branch 固定 select_best_on_val=false，禁止轮内选优")
        if mode != "incremental":
            raise ValueError("multi_branch 仅支持 incremental Base 模式")
        x = torch.as_tensor(features).float().cpu()
        y = torch.as_tensor(labels).long().cpu()

        if int(round_idx) < 0:
            info200 = self.inc200.update(x, y, None, None,
                                         round_idx=round_idx,
                                         mode="incremental")
            # 共享 Base：inc20 从同一份模型与优化器状态分叉
            self.inc20._ensure()
            self.inc20.model.load_state_dict(self.inc200.model.state_dict())
            self.inc20.opt.load_state_dict(self.inc200.opt.state_dict())
            self.inc20.steps_done = self.inc200.steps_done
            self.inc20.round = self.inc200.round
            self.inc20.total_train_sec = 0.0
            self.inc20.history = []
            self._base_ready = True
            self.round = int(round_idx)
            self.steps_done = int(self.inc200.steps_done)
            self.total_train_sec += float(info200["train_sec"])
            info = {"mode": mode, "round": int(round_idx), "base": True,
                    "steps": self.steps, "branches": {"inc200": info200},
                    "best_val_acc": info200.get("best_val_acc"),
                    "n_train": info200.get("n_train"),
                    "train_sec": info200.get("train_sec"),
                    "total_train_sec": self.total_train_sec}
            self.history.append(info)
            return info

        t0 = time.time()
        branch_wall = {}
        w0 = time.perf_counter()
        info200 = self.inc200.update(x, y, None, None, round_idx=round_idx,
                                     mode="incremental")
        branch_wall["inc200"] = time.perf_counter() - w0
        w0 = time.perf_counter()
        info20 = self.inc20.update(x, y, None, None, round_idx=round_idx,
                                   mode="incremental")
        branch_wall["inc20"] = time.perf_counter() - w0

        self.scratch200 = IncrementalCloudLearner(
            self.cfgs["scratch200"], self.dim, self.num_classes,
            device=self.device)
        self.scratch20 = IncrementalCloudLearner(
            self.cfgs["scratch20"], self.dim, self.num_classes,
            device=self.device)
        # 两个从头组必须逐位相同的初始权重；该性质只由 (seed, 结构) 决定，
        # 每次 update 都重建校验模型纯属浪费，只校验一次
        if not self._init_checked:
            m200, _ = self.scratch200._build(
                self.scratch200.seed + int(round_idx))
            m20, _ = self.scratch20._build(
                self.scratch20.seed + int(round_idx))
            self._init_same = _module_sha1(m200) == _module_sha1(m20)
            del m200, m20
            self._init_checked = True
        w0 = time.perf_counter()
        info_s200 = self.scratch200.update(x, y, None, None,
                                           round_idx=round_idx,
                                           mode="scratch")
        branch_wall["scratch200"] = time.perf_counter() - w0
        w0 = time.perf_counter()
        info_s20 = self.scratch20.update(x, y, None, None,
                                         round_idx=round_idx,
                                         mode="scratch")
        branch_wall["scratch20"] = time.perf_counter() - w0

        self.round = int(round_idx)
        self.steps_done += sum(int(i["steps"])
                               for i in (info200, info20, info_s200, info_s20))
        self.total_train_sec += sum(
            float(i["train_sec"])
            for i in (info200, info20, info_s200, info_s20))
        info = {"mode": mode, "round": int(round_idx),
                "init_same": bool(self._init_same),
                "steps": self.steps,
                "branches": {"inc200": info200, "inc20": info20,
                             "scratch200": info_s200,
                             "scratch20": info_s20},
                "best_val_acc": info200.get("best_val_acc"),
                "n_train": info200.get("n_train"),
                "train_sec": float(sum(
                    i["train_sec"] for i in
                    (info200, info20, info_s200, info_s20))),
                "branch_wall_sec": branch_wall,
                "total_train_sec": self.total_train_sec,
                "wall_sec": float(time.time() - t0)}
        self.history.append(info)
        return info

    # ------------------------------------------------------------------
    def evaluate(self, features, labels):
        return self.inc200.evaluate(features, labels)

    def evaluate_branches(self, val_features, val_labels, test_features,
                          test_labels, test_features_int8=None):
        models = {"inc200": self.inc200.model, "inc20": self.inc20.model}
        shared_base = False
        if self.scratch200 is None or self.scratch20 is None:
            models["scratch200"] = self.inc200.model
            models["scratch20"] = self.inc200.model
            shared_base = True
        else:
            models["scratch200"] = self.scratch200.model
            models["scratch20"] = self.scratch20.model
        out = {}
        for name in BRANCHES:
            out[name] = {
                "val": _metrics(models[name], val_features, val_labels,
                                self.num_classes, self.device),
                "test": _metrics(models[name], test_features, test_labels,
                                 self.num_classes, self.device),
                "test_int8": (None if test_features_int8 is None else
                              _metrics(models[name], test_features_int8,
                                       test_labels, self.num_classes,
                                       self.device)),
            }
        out["round"] = int(self.round)
        out["shared_base"] = shared_base
        out["init_same"] = bool(self._init_same)
        self.last_branch_metrics = out
        return out

    # ------------------------------------------------------------------
    def state_dict(self):
        return {"multi_branch": True,
                "inc200": self.inc200.state_dict(),
                "inc20": self.inc20.state_dict(),
                "round": int(self.round),
                "steps_done": int(self.steps_done),
                "total_train_sec": float(self.total_train_sec),
                "base_ready": bool(self._base_ready),
                "history": copy.deepcopy(self.history)}

    def load_state_dict(self, st):
        if not st.get("multi_branch"):
            raise ValueError("断点不是 multi_branch 状态")
        self.inc200.load_state_dict(st["inc200"])
        self.inc20.load_state_dict(st["inc20"])
        self.round = int(st["round"])
        self.steps_done = int(st["steps_done"])
        self.total_train_sec = float(st["total_train_sec"])
        self._base_ready = bool(st["base_ready"])
        self.history = copy.deepcopy(st["history"])
        return self
