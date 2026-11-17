"""统一评估: 基线/BDM → 云端分类 → 量化 → payload"""
import copy
import json
import os
import time

import numpy as np
import torch

from .baselines import allocate_quotas, build_baseline
from .bdm import BatchDM
from .cloud import evaluate_cloud_classifier, train_cloud_classifier
from .config_utils import result_path
from .quant import payload_table, quantize_pair, save_payload


VARIANTS = {
    # 原版 DM：固定随机 MLP 特征空间中的逐类均值匹配
    "random_feature": {},
    # 每类更新次数消融
    "upd30": {"updates_per_class": 30},
    "upd100": {"updates_per_class": 100},
    "upd300": {"updates_per_class": 300},
    # 选点协议消融
    "sel_linear50": {"val_head": "linear", "val_epochs": 50},
    "sel_mlp50": {"val_head": "mlp", "val_epochs": 50},
    "sel_mlp100": {"val_head": "mlp", "val_epochs": 100},
    "sel_final": {"selection": "final"},
}


def eval_reps(reps, rep_y, test_x, test_y, cfg, seeds, val=None):
    """在代表特征上训练云端分类器并评估。

    cfg["cloud"]["val_mode"]:
      external -> 用传入的真实验证集选 epoch（全部代表特征参与训练）
      none     -> 不验证，全部代表特征参与训练
    """
    accs, f1s, infos = [], [], []
    cloud = cfg["cloud"]
    val_mode = cloud.get("val_mode")
    if val_mode is None:
        val_mode = "external" if val is not None else "none"
    if val_mode == "internal":
        raise ValueError("internal 10% 验证协议已移除；请使用 external/none")
    if val_mode not in ("external", "none"):
        raise ValueError(f"未知 cloud.val_mode: {val_mode}")
    tb = cloud.get("training_budget", {})
    budget_mode = tb.get("mode", "fixed_epochs")
    budget_steps = tb.get("steps")
    epochs = int(tb.get("epochs", cloud.get("epochs", 200)))
    use_external = val_mode == "external"
    if use_external and val is None:
        raise ValueError("cloud.val_mode=external 但未提供真实验证集")
    for s in seeds:
        model, info = train_cloud_classifier(
            reps, rep_y, head=cloud["head"],
            num_classes=cfg["num_classes"], seed=s, epochs=epochs,
            lr=cloud["lr"], wd=cloud["wd"], batch=cloud["batch"],
            device="cuda",
            val_features=val["x"] if use_external else None,
            val_labels=val["y"] if use_external else None,
            budget_mode=budget_mode, steps=budget_steps)
        m = evaluate_cloud_classifier(model, test_x, test_y,
                                      cfg["num_classes"])
        accs.append(m["acc"]); f1s.append(m["f1"]); infos.append(info)

    def _avg(key):
        vals = [i[key] for i in infos if i.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "accs": accs, "f1s": f1s, "cloud": infos,
            "validation_mode": val_mode,
            "cloud_head": cloud["head"],
            "n_input_features": int(len(reps)),
            "n_cloud_train": _avg("n_train"),
            "n_cloud_val": _avg("n_val"),
            "cloud_optimizer_steps": _avg("optimizer_steps"),
            "best_cloud_epoch": _avg("best_epoch"),
            "cloud_train_sec": float(np.sum([i["train_sec"] for i in infos])),
            "training_budget_mode": budget_mode}


def reps_path(cfg, method, ratio):
    d = os.path.join(cfg["results_dir"], "reps")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cfg['dataset']}_{method}_r{ratio:g}.npz")


def run_method(cfg, feats, method, ratio, variant=None, seeds=(0, 1, 2),
               save_reps=True, verbose=True, budget_mode=None, budget_steps=None):
    x = feats["train"]["x"]; y = feats["train"]["y"]
    test_x = feats["test"]["x"]; test_y = feats["test"]["y"]
    t0 = time.time()
    dm_seed = None
    if method == "full":
        reps, rep_y = x, y
        info = {"method": "full", "ratio": 1.0, "n_reps": len(y),
                "distill_sec": 0.0}
        tag = "full"
    elif method in ("random", "kmeans", "kmeans_moment"):
        reps, rep_y, info = build_baseline(
            x, y, ratio, method, cfg["num_classes"], cfg["baseline"],
            seed=cfg["baseline"]["seed"], verbose=verbose)
        tag = method
    elif method == "bdm":
        variant = variant or "random_feature"
        dcfg = copy.deepcopy(cfg["dm"])
        bv = cfg.get("bdm_validation", {})
        dcfg["val_head"] = bv.get("head", dcfg.get("val_head", "linear"))
        dcfg["val_epochs"] = int(bv.get("epochs", dcfg.get("val_epochs", 50)))
        dcfg["val_seed"] = int(bv.get("seed", dcfg.get("val_seed", 0)))
        dcfg["selection"] = bv.get("selection", dcfg.get("selection", "val"))
        if "eval_every" in bv:
            dcfg["eval_every"] = int(bv["eval_every"])
        dcfg.update(VARIANTS.get(variant, {}))
        dcfg["kmeans_seed"] = cfg["baseline"]["seed"]
        dcfg["kmeans_n_init"] = cfg["baseline"]["kmeans_n_init"]
        dcfg["kmeans_max_iter"] = cfg["baseline"]["kmeans_max_iter"]
        dm_seed = dcfg.get("seed", 42)
        quotas = allocate_quotas(y.numpy(), int(round(len(y) / ratio)),
                                 cfg["num_classes"])
        dm = BatchDM(x, y, quotas, cfg["num_classes"], dcfg,
                     seed=dm_seed, init=dcfg.get("init", "kmeans"),
                     device="cuda", val=feats["val"])
        out = dm.fit(feats["val"]["x"], feats["val"]["y"], verbose=verbose)
        reps, rep_y = out["features"], out["labels"]
        info = {"method": f"bdm_{variant}", "ratio": ratio,
                "n_reps": len(reps), "distill_sec": out["train_sec"] + out["init_sec"],
                "init_sec": out["init_sec"], "best_val_acc": out["best_val_acc"],
                "best_step": out["best_step"], "peak_gpu_gb": out["peak_gpu_gb"],
                "logs": out["logs"], "variant": variant,
                "class_update_counts": out.get("class_update_counts"),
                "total_optimizer_steps": out.get("total_optimizer_steps"),
                "updates_per_class": out.get("updates_per_class"),
                "per_class_optimizer": out.get("per_class_optimizer"),
                "val_head": out.get("val_head"), "val_epochs": out.get("val_epochs"),
                "val_seed": out.get("val_seed"), "selection": out.get("selection")}
        tag = f"bdm_{variant}"
    else:
        raise ValueError(method)

    ecfg = cfg
    if budget_mode:
        ecfg = copy.deepcopy(cfg)
        ecfg["cloud"]["training_budget"] = {
            "mode": budget_mode, "steps": budget_steps,
            "epochs": int(cfg["cloud"].get("epochs", 200))}
    metrics = eval_reps(reps, rep_y, test_x, test_y, ecfg, seeds,
                        val=feats["val"])
    pt = payload_table(len(y), len(reps), cfg["payload"]["label_bytes"],
                       cfg["payload"]["bandwidth_mbps"])
    rec = {"dataset": cfg["dataset"], "method": tag, "ratio": ratio,
           "protocol_version": cfg.get("protocol_version", "legacy"),
           "feature_checkpoint": feats["meta"].get("vae_ckpt"),
           "distillation_seed": dm_seed, "cloud_seeds": list(seeds),
           "cloud_seed": (int(seeds[0]) if len(seeds) == 1 else None),
           "n_train": len(y), "n_reps": len(reps),
           "n_input_features": metrics["n_input_features"],
           "n_cloud_train": metrics["n_cloud_train"],
           "n_cloud_val": metrics["n_cloud_val"],
           "cloud_head": metrics["cloud_head"],
           "validation_mode": metrics["validation_mode"],
           "validation_head": info.get("val_head"),
           "validation_epochs": info.get("val_epochs"),
           "validation_seed": info.get("val_seed"),
           "selection": info.get("selection"),
           "training_budget_mode": metrics["training_budget_mode"],
           "cloud_optimizer_steps": metrics["cloud_optimizer_steps"],
           "best_cloud_epoch": metrics["best_cloud_epoch"],
           "actual_ratio": len(y) / max(1, len(reps)),
           "distill_sec": info.get("distill_sec", 0.0),
           "distillation_seconds": info.get("distill_sec", 0.0),
           "cloud_train_sec": metrics["cloud_train_sec"],
           "cloud_training_seconds": metrics["cloud_train_sec"],
           "best_distillation_step": info.get("best_step"),
           "total_sec": time.time() - t0,
           **{k: metrics[k] for k in ("acc_mean", "acc_std", "f1_mean", "f1_std")},
           "accuracy": metrics["acc_mean"], "macro_f1": metrics["f1_mean"],
           "upload_bytes": pt["int8"]["distill_bytes"],
           "payload": pt, "feature_meta": feats["meta"],
           "cloud": metrics["cloud"]}
    if "moment_calibration" in info:
        rec["moment_calibration"] = info["moment_calibration"]
        rec["method_base"] = info.get("base")
    if "logs" in info:
        rec["dm_logs"] = info["logs"]
        rec["best_val_acc"] = info.get("best_val_acc")
        rec["best_step"] = info.get("best_step")
        rec["peak_gpu_gb"] = info.get("peak_gpu_gb")
        for k in ("class_update_counts", "total_optimizer_steps",
                  "updates_per_class", "per_class_optimizer"):
            if k in info:
                rec[k] = info[k]
    if save_reps:
        p = reps_path(cfg, tag, ratio)
        np.savez_compressed(p, features=reps.float().numpy(),
                            labels=rep_y.long().numpy(),
                            meta=json.dumps({k: v for k, v in rec.items()
                                             if k not in ("dm_logs",)},
                                            ensure_ascii=False))
        rec["reps_path"] = p
    if verbose:
        print(f"[{tag} r={ratio:g}] M={len(reps)} "
              f"acc {metrics['acc_mean']:.2f}±{metrics['acc_std']:.2f} "
              f"f1 {metrics['f1_mean']:.2f} | {rec['distill_sec']:.1f}s",
              flush=True)
    return rec


def eval_quantized(cfg, feats, method, ratio, variant=None, modes=("fp32", "fp16", "int8"),
                   seeds=(0, 1, 2), out_dir=None, tag=None):
    p = reps_path(cfg, method if method != "bdm" else f"bdm_{variant or 'random_feature'}", ratio)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    obj = np.load(p, allow_pickle=True)
    syn = torch.from_numpy(obj["features"]).float()
    syn_y = torch.from_numpy(obj["labels"]).long()
    test_x = feats["test"]["x"]; test_y = feats["test"]["y"]
    out = {}
    for mode in modes:
        syn_dq, test_dq, q = quantize_pair(syn, test_x, mode)
        m = eval_reps(syn_dq, syn_y, test_dq, test_y, cfg, seeds,
                      val=feats["val"])
        sizes = None
        if out_dir:
            sizes = save_payload(syn, syn_y, q, out_dir,
                                 f"{cfg['dataset']}_{tag or method}_r{ratio:g}_{mode}")
        out[mode] = {**m, "bits": q.bits(), "quant": q.meta(),
                     "payload": sizes}
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"结果: {path}")
