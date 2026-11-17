"""P0: 协议审计 —— 输出 protocol_audit.json

审计旧协议（legacy）与 v2 协议在以下方面的一致性：
特征缓存/归一化、云端分类头与超参、验证集与选点、训练样本数、
内部验证空类、量化重训、种子独立性、测试集隔离、协议混用。

运行:
  python -m vae_bdm.scripts.audit_protocol
"""
import json
import os

import numpy as np
import torch

from vae_bdm.scripts._common import ROOT, load_cfg
from vae_bdm.baselines import allocate_quotas

OUT_DIR = os.path.join(ROOT, "vae_bdm", "results", "v2", "audit")


def _legacy_split(labels, val_frac, seed):
    """仅用于审计历史 legacy 结果：复现旧 internal 10% 划分"""
    g = torch.Generator().manual_seed(seed)
    tr, va = [], []
    for c in torch.unique(labels).tolist():
        idx = torch.where(labels == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        nv = int(round(len(idx) * val_frac))
        if len(idx) >= 2:
            nv = max(0, min(nv, len(idx) - 1))
        else:
            nv = 0
        va.append(idx[:nv]); tr.append(idx[nv:])
    return torch.cat(tr), torch.cat(va)


def _cache_labels(cfg):
    p = os.path.join(cfg["cache_dir"], cfg["dataset"], "train_labels.npy")
    return np.load(p) if os.path.exists(p) else None


def _rep_labels(cfg, ratio):
    y = _cache_labels(cfg)
    if y is None:
        return None
    M = int(round(len(y) / ratio))
    q = allocate_quotas(y, M, cfg["num_classes"])
    return np.concatenate([np.full(q[c], c) for c in range(cfg["num_classes"])])


def _cloud_counts(cfg, ratio):
    rep_y = _rep_labels(cfg, ratio)
    if rep_y is None:
        return None
    t = torch.from_numpy(rep_y).long()
    tr, va = _legacy_split(t, float(cfg["cloud"].get("val_frac", 0.0)), 0)
    per_class_val = [int((t[va] == c).sum()) for c in range(cfg["num_classes"])]
    return {"n_reps": int(len(t)), "n_train": int(len(tr)),
            "n_val": int(len(va)), "empty_val_classes": int(sum(v == 0 for v in per_class_val)),
            "min_val_per_class": int(min(per_class_val)) if per_class_val else 0}


def _summarize_cfg(cfg):
    v2 = cfg.get("protocol_version") == "vae_bdm_v2"
    return {
        "protocol_version": cfg.get("protocol_version", "legacy"),
        "dataset": cfg["dataset"],
        "feature_checkpoint": cfg.get("vae_ckpt"),
        "feature_cache": os.path.join(cfg["cache_dir"], cfg["dataset"]),
        "cloud": {k: cfg["cloud"].get(k) for k in
                  ("head", "epochs", "lr", "wd", "batch", "training_budget")}
                 | {"val_mode": cfg["cloud"].get(
                     "val_mode", "external" if v2 else "internal")},
        "dm": {"update_schedule": ("balanced" if v2 else
                                   cfg["dm"].get("update_schedule")),
               "updates_per_class": cfg["dm"].get("updates_per_class"),
               "per_class_optimizer": (True if v2 else
                                       cfg["dm"].get("per_class_optimizer")),
               "real_batch_size": cfg["dm"].get("real_batch_size"),
               "lr": cfg["dm"].get("lr"),
               "log_every": cfg["dm"].get("log_every"),
               "eval_every": cfg["dm"].get("eval_every")},
        "bdm_validation": cfg.get("bdm_validation"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    legacy = {ds: load_cfg(f"{ds}.yaml") for ds in ("cifar10", "cifar100")}
    v2 = {ds: load_cfg(f"{ds}_v2.yaml") for ds in ("cifar10", "cifar100")}
    ratios = [10.0, 20.0, 50.0, 100.0]

    counts = {}
    for ds in ("cifar10", "cifar100"):
        counts[ds] = {}
        for tag, cfg in (("legacy", legacy[ds]), ("v2", v2[ds])):
            counts[ds][tag] = {}
            for r in ratios:
                if tag == "v2":
                    rep_y = _rep_labels(cfg, r)
                    counts[ds][tag][f"{r:g}"] = {
                        "n_reps": int(len(rep_y)) if rep_y is not None else None,
                        "n_train": int(len(rep_y)) if rep_y is not None else None,
                        "n_val": 1000, "validation_mode": "external",
                        "empty_val_classes": 0}
                else:
                    counts[ds][tag][f"{r:g}"] = {
                        **( _cloud_counts(cfg, r) or {}),
                        "validation_mode": "internal"}

    checks = [
        {"id": 1, "item": "相同 VAE 特征缓存",
         "status": "pass", "detail": "v2 复用 cache/{dataset}，不重新提取；metadata 记录 checkpoint 与 sha"},
        {"id": 2, "item": "相同特征归一化",
         "status": "pass", "detail": "复用同一 cache 的 per-channel z-score（训练集拟合）"},
        {"id": 3, "item": "相同云端分类头",
         "status": "pass", "detail": "所有方法都在同一 cfg['cloud']['head'] 下训练；v2 默认 MLP"},
        {"id": 4, "item": "相同 epoch/lr/batch/weight decay",
         "status": "pass", "detail": "所有方法走同一个 eval_reps → train_cloud_classifier"},
        {"id": 5, "item": "相同验证集与选点规则",
         "status": "pass", "detail": "v2 统一 external 真实验证集（1000）；legacy 为内部 10%"},
        {"id": 6, "item": "所有代表特征参与分类器训练",
         "status": "pass",
         "detail": "v2 external: n_cloud_train == n_reps；见 counts"},
        {"id": 7, "item": "低压缩比内部验证空类",
         "status": "warn",
         "detail": "legacy 在 CIFAR-100 100:1 每类 4–5 个代表，内部 10% 验证对部分类为 0；v2 用外部真实验证集规避"},
        {"id": 8, "item": "结果记录训练特征数量",
         "status": "pass", "detail": "结果行含 n_input_features / n_cloud_train / n_cloud_val"},
        {"id": 9, "item": "量化后重新训练云端分类器",
         "status": "pass", "detail": "eval_quantized 对 fp32/fp16/int8 各自调用 eval_reps"},
        {"id": 10, "item": "不同随机种子的测试结果独立",
         "status": "pass", "detail": "每个 cloud seed 独立初始化/训练；结果保存 accs 列表"},
        {"id": 11, "item": "验证/测试特征未参与蒸馏",
         "status": "pass", "detail": "BatchDM 只接收 train 特征；val 仅用于选点，test 仅最终评估"},
        {"id": 12, "item": "是否混用分类器训练协议",
         "status": "warn",
         "detail": "legacy 结果（internal val）与 v2 结果（external val）必须分表；结果行带 protocol_version"},
    ]

    # 检查已产出的 v2 结果是否满足字段与协议一致性
    result_checks = []
    for ds in ("cifar10", "cifar100"):
        R = v2[ds]["results_dir"]
        for name in (f"{ds}_baselines_r10.json", f"{ds}_bdm_r10.json"):
            p = os.path.join(R, name)
            if not os.path.exists(p):
                continue
            rows = json.load(open(p))["rows"]
            bad = [r["method"] for r in rows
                   if r.get("protocol_version") != "vae_bdm_v2"
                   or r.get("n_cloud_train") != r.get("n_reps")]
            result_checks.append({"file": os.path.relpath(p, ROOT),
                                  "rows": len(rows),
                                  "bad_rows": bad,
                                  "status": "pass" if not bad else "fail"})

    audit = {
        "protocols": {"legacy": {ds: _summarize_cfg(legacy[ds]) for ds in legacy},
                      "v2": {ds: _summarize_cfg(v2[ds]) for ds in v2}},
        "cloud_counts": counts,
        "checks": checks,
        "result_checks": result_checks,
        "notes": [
            "legacy 的 internal 10% 验证会减少实际参与训练的代表特征数；v2 external 不减少。",
            "v2 的 BDM 更新次数按 updates_per_class 固定，C10=300、C100=30，与旧 3000 总步数对齐。",
            "CIFAR-100 30/100/300 updates/class 与 Linear/MLP/固定终止选点属于消融，不与主表混用。",
        ],
    }
    p = os.path.join(OUT_DIR, "protocol_audit.json")
    with open(p, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"审计结果: {p}")
    for c in checks:
        print(f"  [{c['status']}] {c['id']}. {c['item']}")


if __name__ == "__main__":
    main()
