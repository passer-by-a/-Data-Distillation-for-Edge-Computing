"""DDPM U-Net 双路径特征提取 + 投影 + 分类头

主干来自 checkpoints/ddpm_cifar10.pt (UNet2DModel, 100k 步, EMA)。
该主干是 4 级下采样 (128/256/256/256)，与规格的 3 级参考结构存在映射关系：

  forward_features(layer="encoder16") -> 256×16×16
      conv_in -> down_blocks[0] -> down_blocks[1]，第二次下采样之后立即返回
  forward_features(layer="middle")    -> 256×8×8
      conv_in -> down_blocks[0..2]，8×8 最深层（规格中 Middle 的位置）
  forward_features(layer="middle4")   -> 256×4×4
      再加上 down_blocks[3] 与真正的 mid_block（含 Self-Attention），作补充对照

偏差说明: 现有权重是 4 级 UNet，8×8 处通道为 256 而非规格的 256(同)、
16×16 处为 256 而非 128。因此投影层输入通道按实际 256 配置，
输出形状严格满足 16×8×8 与 4×16×16（均为 1024 元素）。

所有特征路径都是显式提前返回，不会执行后续 down_blocks[3]、mid_block 或 Decoder。
"""

import os

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel

ROOT = os.path.dirname(os.path.abspath(__file__))

LAYERS = {"A": "middle", "A_raw": "middle", "A_gap": "middle",
          "B1": "encoder16", "B2": "encoder16", "B_raw": "encoder16", "B_gap": "encoder16",
          "A_mid4": "middle4"}


class DDPMBackbone(nn.Module):
    """冻结的无条件 DDPM U-Net (EMA 权重)"""

    def __init__(self, ckpt_path, device, use_ema=True):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.unet = UNet2DModel(**ck["arch"]).to(device)
        sd = ck["ema_state_dict"] if use_ema and "ema_state_dict" in ck else ck["model_state_dict"]
        self.unet.load_state_dict(sd)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False
        self.scheduler = DDPMScheduler(**ck["scheduler"])
        self.pretrain_step = int(ck.get("step", 0))
        self.pretrain_loss = float(ck.get("loss", float("nan")))
        self.n_params = sum(p.numel() for p in self.unet.parameters())
        self.channels = tuple(ck["arch"]["block_out_channels"])

    def _temb(self, t, batch, dtype):
        if t.dim() == 0:
            t = t.repeat(batch)
        emb = self.unet.time_proj(t)
        emb = self.unet.time_embedding(emb, None)
        return emb.to(dtype)

    def forward(self, x_t, t):
        return self.unet(x_t, t).sample

    def forward_features(self, x_t, t, layer="middle"):
        """提前返回指定层特征, 不执行后续 Encoder/Middle/Decoder

        注意: DownBlock 的返回值是"下采样之后"的张量, 而 res_samples 保存的是
        每个 ResBlock 的输出(末尾再追加下采样结果)。规格要求的 encoder16 与
        middle 都是下采样前的特征, 因此取 _pre_down(res_samples)。
        """
        temb = self._temb(t, x_t.size(0), x_t.dtype)
        h = self.unet.conv_in(x_t)
        h, res0 = self.unet.down_blocks[0](h, temb)
        h, res1 = self.unet.down_blocks[1](h, temb)
        if layer == "encoder16":
            return self._pre_down(res1)                   # 256×16×16
        h, res2 = self.unet.down_blocks[2](h, temb)
        if layer == "middle":
            return self._pre_down(res2)                   # 256×8×8
        if layer == "middle4":
            h, res3 = self.unet.down_blocks[3](h, temb)
            h = self.unet.mid_block(h, temb)              # 256×4×4 (真 Middle)
            return h
        raise ValueError(layer)

    @staticmethod
    def _pre_down(res):
        """res_samples 末尾追加了下采样后的张量; 若末两项分辨率不同, 取倒数第二项"""
        return res[-1] if res[-1].shape[-1] == res[-2].shape[-1] else res[-2]


class VAEBackbone(nn.Module):
    """原有 EdgeVAE 编码器, 输出确定性 mu (16×8×8)"""

    def __init__(self, num_classes, device, ckpt_path):
        super().__init__()
        from vae import EdgeVAE
        self.ae = EdgeVAE(latent_dim=16, num_classes=num_classes).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.ae.load_state_dict(ck["model_state_dict"])
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False
        self.n_params = sum(p.numel() for p in self.ae.parameters())
        self.channels = (16,)

    def forward_features(self, x, t=None, layer=None):
        return self.ae.encode_mu(x)


class ConvProjection(nn.Module):
    """1×1 或 3×3/s2 卷积投影"""

    def __init__(self, in_ch, out_ch, kernel=1, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad)

    def forward(self, f):
        return self.conv(f)


class IdentityProjection(nn.Module):
    def forward(self, f):
        return f


class GAPProjection(nn.Module):
    def forward(self, f):
        return f.mean(dim=(2, 3))


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(z.flatten(1))


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, z):
        return self.net(z.flatten(1))


def build_projection(mode, in_ch):
    """按方案返回 (projection, 分类输入维度)"""
    if mode == "A":
        return ConvProjection(in_ch, 16, 1, 1), 16 * 8 * 8
    if mode == "B1":
        return ConvProjection(in_ch, 16, 3, 2), 16 * 8 * 8
    if mode == "B2":
        return ConvProjection(in_ch, 4, 1, 1), 4 * 16 * 16
    if mode == "A_raw":
        return IdentityProjection(), in_ch * 8 * 8
    if mode == "B_raw":
        return IdentityProjection(), in_ch * 16 * 16
    if mode in ("A_gap", "B_gap"):
        return GAPProjection(), in_ch
    if mode == "VAE" or mode == "VAE_unsup":
        return IdentityProjection(), 16 * 8 * 8
    raise ValueError(mode)


def build_head(head_type, in_dim, num_classes):
    return LinearHead(in_dim, num_classes) if head_type == "linear" \
        else MLPHead(in_dim, num_classes)


class Pipeline(nn.Module):
    """端到端推理: 主干提前返回 → 投影 → 分类头 (用于 FLOPs / 延迟测量)"""

    def __init__(self, backbone, mode, projection, head, mean=None, std=None,
                 norm=True, layer=None, num_classes=None):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.layer = layer
        self.proj = projection
        self.head = head
        self.norm = norm
        if mean is not None:
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def forward(self, x0, t=0, noise=None, noise_mode="schedule"):
        if self.mode in ("VAE", "VAE_unsup"):
            f = self.backbone.forward_features(x0)
        else:
            if noise_mode == "clean" or noise is None:
                x_t = x0
            else:
                abar = self.backbone.scheduler.alphas_cumprod[int(t)].to(x0.device)
                x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
            tt = torch.full((x0.size(0),), int(t), device=x0.device, dtype=torch.long)
            f = self.backbone.forward_features(x_t, tt, layer=self.layer)
        if self.norm and hasattr(self, "mean"):
            f = (f - self.mean) / self.std
        return self.head(self.proj(f))

