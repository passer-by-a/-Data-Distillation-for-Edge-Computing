"""Shared Stem / Deep Tail: 直接引用现有 EdgeVAE 的模块，不复制参数"""
import torch.nn as nn


class StemTail:
    """把 EdgeEncoder 拆成 stem + tail（tail 包含 mu/logvar 头）"""

    def __init__(self, vae, fusion_stage="8x8"):
        self.enc = vae.encoder
        if fusion_stage not in ("8x8", "16x16"):
            raise ValueError(f"fusion_stage 必须是 8x8 或 16x16，收到 {fusion_stage}")
        self.fusion_stage = fusion_stage
        self.n_stem = 1 if fusion_stage == "16x16" else 2

    @property
    def channels(self):
        return self.enc.conv1[0].out_channels

    @property
    def spatial(self):
        return 32 // (2 ** self.n_stem)

    def stem(self, x):
        h = self.enc.conv1(x)
        for d in self.enc.down[:self.n_stem]:
            h = d(h)
        return h

    def tail(self, h):
        for d in self.enc.down[self.n_stem:]:
            h = d(h)
        h = self.enc.mid(h)
        mu = self.enc.mu_conv(h)
        logvar = self.enc.logvar_conv(h).clamp(-8, 8)
        return mu, logvar
