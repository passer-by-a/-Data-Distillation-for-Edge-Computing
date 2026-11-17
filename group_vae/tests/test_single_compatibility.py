"""K=1 组编码与单图编码在初始化时一致"""
import torch

from group_vae.tests.test_shapes import make


def test_single_compatibility():
    x = torch.randn(3, 3, 32, 32)
    for fusion in ("mean", "attention", "film"):
        m = make(fusion)
        with torch.no_grad():
            single = m.encode_single(x)
            group = m.encode_group(x[:, None], torch.ones(3, 1, dtype=torch.bool))
        assert torch.allclose(group, single, atol=1e-5, rtol=1e-4), fusion
