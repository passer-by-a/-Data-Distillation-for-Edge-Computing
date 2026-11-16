"""解析式矩校正（Streaming Moment Calibration）

低成本对照: 不做梯度优化，直接把代表特征逐类、逐维缩放到当前统计记忆的
均值/方差。缩放系数截断，避免异常放大。
"""
import time

import torch

from vae_bdm.baselines import moment_calibration


def streaming_moment_calibrate(features, labels, statistics, allowed_idx=None,
                               eps=1e-6, max_scale=3.0, round_idx=0):
    t0 = time.time()
    feats = torch.as_tensor(features).float().cpu()
    ys = torch.as_tensor(labels).long().cpu()
    out, info = moment_calibration(feats, ys, statistics.stats_dict(),
                                   eps=eps, max_scale=max_scale)
    if allowed_idx is not None:
        allowed = torch.zeros(len(out), dtype=torch.bool)
        allowed[torch.as_tensor(allowed_idx, dtype=torch.long)] = True
        out[~allowed] = feats[~allowed]
        info["n_optimized"] = int(allowed.sum())
    else:
        info["n_optimized"] = int(len(out))
    info["round"] = int(round_idx)
    info["method"] = "moment_calibration"
    info["sec"] = time.time() - t0
    return out, info
