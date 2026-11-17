"""轻量 Set Attention：只在组维 K 上做注意力，无位置编码"""
import torch
import torch.nn as nn


class MHABlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask):
        kpm = ~mask
        y, _ = self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)
        x = self.norm1(x + y)
        x = self.norm2(x + self.ff(x))
        return x * mask[..., None]


class SetAttention(nn.Module):
    def __init__(self, in_dim, dim=128, heads=4, layers=1, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(
            [MHABlock(dim, heads, dropout) for _ in range(layers)])

    def forward(self, tokens, mask):
        x = self.proj(tokens) * mask[..., None]
        for b in self.blocks:
            x = b(x, mask)
        return x
