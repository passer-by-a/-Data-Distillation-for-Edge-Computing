"""两阶段共用的特征统计工具"""

import torch

from common import DEV


def channel_stats(raw_tr, batch=512):
    """逐通道均值/标准差 (流式计算, 不接触标签)"""
    c = raw_tr.size(1)
    s = torch.zeros(c, dtype=torch.float64)
    ss = torch.zeros(c, dtype=torch.float64)
    n = 0
    for i in range(0, raw_tr.size(0), batch):
        f = raw_tr[i:i + batch].float()
        s += f.sum(dim=(0, 2, 3)).double()
        ss += f.pow(2).sum(dim=(0, 2, 3)).double()
        n += f.size(0) * f.size(2) * f.size(3)
    m = s / n
    v = (ss / n - m.pow(2)).clamp_min(1e-12)
    return (m.float().view(1, -1, 1, 1).to(DEV),
            v.sqrt().float().view(1, -1, 1, 1).to(DEV))
