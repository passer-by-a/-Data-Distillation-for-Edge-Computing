"""跨图像条件调制 (FiLM) 与 masked attention pooling"""
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """h_i' = h_i + tanh(gate) * (gamma_i * h_i + beta_i)

    gamma 初始为 1，beta 为 0，gate=0，因此初始化时恒等。
    """

    def __init__(self, ctx_dim, channels, hidden=128):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, maps, ctx, mask):
        b, k = ctx.shape[:2]
        gb = self.mlp(ctx).view(b, k, 2, self.channels, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gb[:, :, 0])
        beta = 0.1 * torch.tanh(gb[:, :, 1])
        out = maps + torch.tanh(self.gate) * (gamma * maps + beta)
        return out * mask[..., None, None, None]


class AttentionPool(nn.Module):
    def __init__(self, ctx_dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, maps, ctx, mask):
        s = self.score(ctx).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=1) * mask.float()
        w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
        g = (w[..., None, None, None] * maps).sum(1)
        return g, w.unsqueeze(-1)
