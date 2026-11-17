"""组级梯度可回流到 Set Attention / FiLM / 共享 Encoder"""
import torch

from group_vae.tests.test_shapes import make


def test_group_gradients_frozen_vae():
    m = make("film")
    for p in m.vae.parameters():
        p.requires_grad = False
    x = torch.randn(2, 6, 3, 32, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = m.forward_group(x, mask, with_single_ref=False)
    loss = out["group_mu"].sum() + out["group_logits"].sum()
    m.zero_grad(set_to_none=True)
    loss.backward()
    for name, p in m.named_parameters():
        if name.startswith("vae."):
            assert p.grad is None
        elif p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_joint_updates_encoder():
    m = make("attention")
    x = torch.randn(1, 4, 3, 32, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = m.forward_joint(x, mask)
    loss = out["group_mu"].sum()
    m.zero_grad(set_to_none=True)
    loss.backward()
    assert m.vae.encoder.conv1[0].weight.grad is not None
    assert m.vae.encoder.conv1[0].weight.grad.abs().sum() > 0
