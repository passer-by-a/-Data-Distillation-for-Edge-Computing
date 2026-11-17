"""形状与无潜变量聚合检查"""
import torch

from group_vae.models.group_vae import GroupVAE
from vae import EdgeVAE


def make(fusion="film"):
    torch.manual_seed(0)
    vae = EdgeVAE(latent_dim=16, num_classes=3).eval()
    cfg = {"fusion": fusion, "fusion_stage": "8x8", "attn_dim": 32,
           "attn_heads": 4, "attn_layers": 1, "film_hidden": 32}
    return GroupVAE(vae, 3, cfg).eval()


def test_group_shape():
    for fusion in ("mean", "attention", "film"):
        m = make(fusion)
        x = torch.randn(2, 10, 3, 32, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        f = m.encode_group(x, mask)
        assert f.shape == (2, 1024)
        out = m.forward_group(x, mask, with_single_ref=True)
        assert out["group_logits"].shape == (2, 3)
        assert out["single_mu"].shape == (2, 10, 1024)


def test_no_latent_aggregation():
    """组路径不能调用 encode_single（否则说明先生成 10 个潜变量再聚合）"""
    m = make("attention")
    m.encode_single = lambda x: (_ for _ in ()).throw(RuntimeError("called"))
    x = torch.randn(1, 4, 3, 32, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    f = m.encode_group(x, mask)
    assert f.shape == (1, 1024)
