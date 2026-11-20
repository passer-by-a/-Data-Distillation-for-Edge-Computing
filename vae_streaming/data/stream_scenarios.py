"""流式场景构造: IID / Label Shift / Covariate Shift

所有分组/切分只使用训练集索引与训练集标签；测试集只用于评估。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def label_shift_blocks(num_classes, num_rounds, seed=0):
    """固定类别分块；每轮强调一个块，但所有类别仍然可见"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_classes)
    num_blocks = min(num_rounds, num_classes)
    return [np.sort(b) for b in np.array_split(perm, num_blocks)]


def round_weights(num_classes, num_rounds, scenario, seed=0):
    """返回 [num_rounds, num_classes] 的类别采样权重（未归一化）。

    IID / Covariate Shift : 全 1（域变化由 apply_covariate_shift 施加）
    Label Shift           : 每轮强调一个类别块（所有类别仍可见，只改变比例）
    """
    if scenario in ("iid", "covariate_shift"):
        return np.ones((num_rounds, num_classes), dtype=np.float64)
    if scenario != "label_shift":
        raise ValueError(f"未知场景: {scenario}")
    blocks = label_shift_blocks(num_classes, num_rounds, seed)
    w = np.ones((num_rounds, num_classes), dtype=np.float64)
    for r in range(num_rounds):
        w[r, blocks[r % len(blocks)]] = 3.0
    return w


def allocate_round_counts(class_sizes, num_rounds, weights):
    """按权重把每类剩余样本分配到各轮，严格遵守每类容量。

    返回 int 数组 [num_rounds, num_classes]，行和 = 各轮预算。
    """
    class_sizes = np.asarray(class_sizes, dtype=np.int64)
    total = int(class_sizes.sum())
    base = total // num_rounds
    rem = total % num_rounds
    sizes = np.array([base + (1 if r < rem else 0) for r in range(num_rounds)],
                     dtype=np.int64)
    alloc = np.zeros((num_rounds, len(class_sizes)), dtype=np.int64)
    remaining = class_sizes.copy()

    for r in range(num_rounds):
        budget = int(sizes[r])
        cap = remaining.copy()
        w = np.asarray(weights[r], dtype=np.float64) * (cap > 0)
        if w.sum() <= 0:
            w = (cap > 0).astype(np.float64)
        target = budget * w / w.sum()
        take = np.minimum(np.floor(target).astype(np.int64), cap)
        left = budget - int(take.sum())
        frac_order = np.argsort(-(target - np.floor(target)))
        while left > 0:
            progressed = False
            for c in frac_order:
                if left == 0:
                    break
                if take[c] < cap[c]:
                    take[c] += 1
                    left -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("轮次配额分配失败：容量不足")
        alloc[r] = take
        remaining -= take

    if remaining.sum() != 0:
        raise RuntimeError(f"轮次分配未耗尽剩余样本: {remaining.sum()}")
    return alloc


def covariate_levels(num_rounds):
    """协变量漂移强度: 第 1 轮起单调增强，最后一轮达到 1.0"""
    return [float(i + 1) / num_rounds for i in range(num_rounds)]


def _gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(math.ceil(2 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k


def apply_covariate_shift(images_u8, level, device="cpu"):
    """确定性域变换（亮度 / 对比度 / 模糊 / 颜色风格），输入输出均为 uint8。

    level ∈ [0,1]；0 表示原图。测试域可用相同函数构造，但绝不参与训练。
    """
    if level <= 0:
        return images_u8.clone()
    x = images_u8.to(device).float().div(127.5).sub(1.0)

    # 亮度
    x = x * (1.0 + 0.35 * level)
    # 对比度（逐图）
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * (1.0 + 0.35 * level) + m
    # 颜色风格
    gains = torch.tensor([1.0 + 0.15 * level, 1.0, 1.0 - 0.15 * level],
                         device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = x * gains
    # 模糊（可分离高斯）
    if level > 1e-3:
        sigma = 1.2 * level
        k = _gaussian_kernel(sigma, device, x.dtype)
        r = (len(k) - 1) // 2
        w = torch.outer(k, k).view(1, 1, len(k), len(k)).repeat(3, 1, 1, 1)
        x = F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), w, groups=3)

    x = x.clamp(-1, 1)
    out = x.mul(127.5).add(127.5).round().clamp(0, 255).to(torch.uint8)
    return out.cpu() if device != "cpu" else out
