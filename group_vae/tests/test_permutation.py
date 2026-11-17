"""置换不变性与 padding 不变性"""
import torch

from group_vae.tests.test_shapes import make


def test_permutation():
    for fusion in ("mean", "attention", "film"):
        m = make(fusion)
        x = torch.randn(2, 6, 3, 32, 32)
        mask = torch.ones(2, 6, dtype=torch.bool)
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        with torch.no_grad():
            a = m.encode_group(x, mask)
            b = m.encode_group(x[:, perm], mask)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_padding():
    for fusion in ("mean", "attention", "film"):
        m = make(fusion)
        x = torch.randn(2, 5, 3, 32, 32)
        mask = torch.ones(2, 5, dtype=torch.bool)
        pad = torch.randn(2, 3, 3, 32, 32)
        xp = torch.cat([x, pad], 1)
        mp = torch.cat([mask, torch.zeros(2, 3, dtype=torch.bool)], 1)
        with torch.no_grad():
            a = m.encode_group(x, mask)
            b = m.encode_group(xp, mp)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
