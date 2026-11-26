"""Pixel 表示单元测试（CPU 可运行）

覆盖 PixelFeatureProtocol 的拟合/标准化/冻结语义，以及 Base 代表初始化的
确定性（同一输入跨调用逐位一致）。运行:
  python -m pytest vae_streaming/tests/test_pixel_protocol.py -q
"""
import numpy as np
import torch

from vae_streaming.memory.representative_memory import \
    initialize_representatives
from vae_streaming.models.pixel_protocol import PixelFeatureProtocol


def _images(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, 32, 32, generator=g) * 255).byte()


def test_fit_matches_manual_channel_stats():
    x = _images()
    p = PixelFeatureProtocol.fit(x)
    f = x.float().div(255.0)
    assert torch.allclose(p.mean.flatten(), f.mean(dim=(0, 2, 3)), atol=1e-6)
    assert torch.allclose(p.std.flatten(), f.std(dim=(0, 2, 3), unbiased=False),
                          atol=1e-6)


def test_transform_shape_and_frozen_coordinates():
    base = _images(seed=0)
    p = PixelFeatureProtocol.fit(base)
    mean0, std0 = p.mean.clone(), p.std.clone()
    # 变换后为 3072 维；Base 上逐通道（跨样本与空间）均值≈0、标准差≈1
    z = p(base)
    assert z.shape == (len(base), 3 * 32 * 32)
    zc = z.view(len(base), 3, 32, 32)
    assert zc.mean(dim=(0, 2, 3)).abs().max() < 1e-5
    assert (zc.std(dim=(0, 2, 3), unbiased=False) - 1).abs().max() < 1e-5
    # 作用在另一批数据上不得更新统计量（只标准化，不重新拟合）
    _ = p(_images(seed=1))
    assert torch.equal(p.mean, mean0) and torch.equal(p.std, std0)


def test_reconstruct_from_saved_stats_is_identical():
    base = _images(seed=0)
    p = PixelFeatureProtocol.fit(base)
    q = PixelFeatureProtocol(p.mean.clone(), p.std.clone())
    x = _images(seed=2)
    assert torch.equal(p(x), q(x))


def test_meta_reports_layout_and_dim():
    p = PixelFeatureProtocol.fit(_images())
    meta = p.meta()
    assert meta["dim"] == 3072
    assert meta["layout"] == "3x32x32_flat"
    assert len(meta["channel_mean"]) == 3 and len(meta["channel_std"]) == 3


def test_kmeans_initialization_is_deterministic():
    """同一 Base 特征/seed 两次初始化必须逐位一致（自检 #2 的前提）。"""
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(200, 3072, generator=g)
    labels = torch.arange(200) % 4
    a = initialize_representatives(feats, labels, [5, 5, 5, 5], method="kmeans",
                                   seed=42)
    b = initialize_representatives(feats, labels, [5, 5, 5, 5], method="kmeans",
                                   seed=42)
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[2], b[2])
