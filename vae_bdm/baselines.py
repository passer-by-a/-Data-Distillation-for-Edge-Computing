"""确定性配额与静态 Coreset 基线: Full / Random / K-means

另含解析式矩校正基线（无需梯度迭代）: kmeans_moment
"""
import time

import numpy as np
import torch
from sklearn.cluster import KMeans


def class_stats(x, y, num_classes):
    """完整训练集的逐类均值/逐维方差（仅训练集）"""
    x = torch.as_tensor(x).float().cpu()
    y = torch.as_tensor(y).long().cpu().numpy()
    stats = {}
    for c in range(num_classes):
        z = x[y == c]
        stats[c] = (z.mean(0), z.var(0, unbiased=False))
    return stats


def moment_calibration(reps, rep_y, real_stats, eps=1e-6, max_scale=3.0):
    """解析式矩校正: s' = mu_r + sqrt((v_r+eps)/(v_s+eps)) * (s - mu_s)

    逐类、逐维；缩放系数截断到 [1/max_scale, max_scale]，因此不保证
    所有维度方差完全匹配。代表特征少于 2 个的类跳过并记录。
    """
    t0 = time.time()
    out = reps.clone().float()
    rep_y = rep_y.long()
    info = {"eps": eps, "max_scale": max_scale, "classes": {},
            "skipped": [], "clamped_frac": 0.0, "pre_mean_mse": [],
            "post_mean_mse": [], "pre_var_mse": [], "post_var_mse": []}
    n_clamped = 0
    n_dims = 0
    for c in sorted(real_stats):
        idx = torch.where(rep_y == c)[0]
        if len(idx) < 2:
            info["skipped"].append(c)
            continue
        mu_r, var_r = real_stats[c]
        mu_s = out[idx].mean(0)
        var_s = out[idx].var(0, unbiased=False)
        scale = ((var_r + eps) / (var_s + eps)).sqrt()
        lo, hi = 1.0 / max_scale, max_scale
        n_clamped += int(((scale < lo) | (scale > hi)).sum())
        n_dims += int(scale.numel())
        scale = scale.clamp(lo, hi)
        out[idx] = mu_r + (out[idx] - mu_s) * scale
        mu_p = out[idx].mean(0)
        var_p = out[idx].var(0, unbiased=False)
        info["pre_mean_mse"].append(float((mu_s - mu_r).pow(2).mean()))
        info["post_mean_mse"].append(float((mu_p - mu_r).pow(2).mean()))
        info["pre_var_mse"].append(float((var_s - var_r).pow(2).mean()))
        info["post_var_mse"].append(float((var_p - var_r).pow(2).mean()))
        info["classes"][str(c)] = {"pre_mean_mse": info["pre_mean_mse"][-1],
                                   "post_mean_mse": info["post_mean_mse"][-1],
                                   "pre_var_mse": info["pre_var_mse"][-1],
                                   "post_var_mse": info["post_var_mse"][-1]}
    for k in ("pre_mean_mse", "post_mean_mse", "pre_var_mse", "post_var_mse"):
        info[k] = float(np.mean(info[k])) if info[k] else float("nan")
    info["clamped_frac"] = float(n_clamped / max(1, n_dims))
    info["elapsed_sec"] = time.time() - t0
    info["n_reps"] = int(len(out))
    info["n_classes"] = int(len(info["classes"]))
    assert torch.isfinite(out).all(), "moment calibration 产生 NaN/Inf"
    return out, info


def allocate_quotas(labels, M, num_classes):
    """按类别样本数比例分配，最大余数法；保证精确总和 M 且每类 ≥1"""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.int64)
    raw = M * counts / counts.sum()
    q = np.floor(raw).astype(np.int64)
    q = np.maximum(q, 1)
    # 先处理超出（极少见）
    while q.sum() > M:
        excess = np.where(q > 1)[0]
        j = excess[np.argmin(raw[excess] - np.floor(raw[excess]))]
        q[j] -= 1
    # 再补足：余数大的优先，平手取 class id 小
    rem = raw - np.floor(raw)
    order = sorted(range(num_classes), key=lambda c: (-rem[c], c))
    i = 0
    while q.sum() < M:
        c = order[i % num_classes]
        if q[c] < counts[c]:
            q[c] += 1
        i += 1
        if i > 10 * num_classes:
            raise RuntimeError("配额分配失败")
    return q


def build_baseline(x, y, ratio, method, num_classes, cfg, seed=42, verbose=True):
    """返回 reps [M,1024], rep_labels [M], info"""
    x = torch.as_tensor(x).float().cpu()
    y = torch.as_tensor(y).long().cpu().numpy()
    M = int(round(len(y) / float(ratio)))
    quotas = allocate_quotas(y, M, num_classes)
    t0 = time.time()

    if method == "full":
        return x, torch.from_numpy(y).long(), {
            "method": "full", "ratio": 1.0, "n_reps": len(y),
            "quotas": np.bincount(y, minlength=num_classes).tolist(),
            "distill_sec": 0.0}

    if method == "random":
        rng = np.random.RandomState(seed)
        reps, labels = [], []
        for c in range(num_classes):
            idx = np.where(y == c)[0]
            sel = rng.choice(idx, quotas[c], replace=False)
            reps.append(x[sel])
            labels += [c] * quotas[c]
        reps = torch.cat(reps, 0)
    elif method == "kmeans":
        reps, labels = [], []
        for c in range(num_classes):
            idx = np.where(y == c)[0]
            z = x[idx].numpy()
            km = KMeans(n_clusters=quotas[c], random_state=seed,
                        n_init=cfg.get("kmeans_n_init", 1),
                        max_iter=cfg.get("kmeans_max_iter", 100))
            km.fit(z)
            reps.append(torch.from_numpy(km.cluster_centers_).float())
            labels += [c] * quotas[c]
        reps = torch.cat(reps, 0)
    elif method == "kmeans_moment":
        reps, rep_y, base_info = build_baseline(
            x, y, ratio, "kmeans", num_classes, cfg, seed=seed, verbose=False)
        stats = class_stats(x, y, num_classes)
        reps, cinfo = moment_calibration(
            reps, rep_y, stats,
            eps=float(cfg.get("moment_eps", 1e-6)),
            max_scale=float(cfg.get("moment_max_scale", 3.0)))
        info = {**base_info, "method": method, "base": "kmeans",
                "moment_calibration": cinfo,
                "distill_sec": base_info["distill_sec"] + cinfo["elapsed_sec"]}
        if verbose:
            print(f"[{method} r={ratio:g}] M={len(reps)} | "
                  f"{info['distill_sec']:.1f}s", flush=True)
        return reps.float(), rep_y.long(), info
    else:
        raise ValueError(f"unknown baseline: {method}")

    reps = torch.stack(reps, 0) if isinstance(reps, list) else reps
    reps = reps.float()
    rep_labels = torch.tensor(labels, dtype=torch.long)
    info = {"method": method, "ratio": ratio, "n_reps": len(reps),
            "quotas": quotas.tolist(), "distill_sec": time.time() - t0}
    if verbose:
        print(f"[{method} r={ratio:g}] M={len(reps)} | {info['distill_sec']:.1f}s",
              flush=True)
    return reps, rep_labels, info
