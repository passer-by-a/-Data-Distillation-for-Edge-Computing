"""Pixel 表示：CIFAR 图像 → 标准化展平特征

仅用 Initial Base 拟合 RGB 三通道均值/标准差，之后冻结。Base、流式
批次、历史回放、验证集与测试集共用同一变换。标准化后把 [3,32,32]
展平为 3072 维，作为代表集与云端分类器的输入坐标。
"""
import torch

_EPS = 1e-6


class PixelFeatureProtocol:
    """固定像素坐标：x/255 → 通道标准化 → flatten(1)。"""

    dim = 3 * 32 * 32

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    @classmethod
    def fit(cls, images):
        """在 Initial Base 图像上估计三通道均值/标准差（[0,1] 浮点）。"""
        x = torch.as_tensor(images).float().div(255.0)
        mean = x.mean(dim=(0, 2, 3))
        std = x.std(dim=(0, 2, 3), unbiased=False).clamp_min(_EPS)
        return cls(mean, std)

    def __call__(self, images):
        x = torch.as_tensor(images).float().div(255.0)
        return ((x - self.mean) / self.std).flatten(1)

    # 与 FeatureProtocol 的同名接口，便于引擎统一调用
    def normalize(self, images):
        return self(images)

    def meta(self):
        return {"method": "pixel_standardize", "source": "initial_base",
                "dim": int(self.dim), "layout": "3x32x32_flat",
                "channel_mean": [float(v) for v in self.mean.flatten()],
                "channel_std": [float(v) for v in self.std.flatten()]}
