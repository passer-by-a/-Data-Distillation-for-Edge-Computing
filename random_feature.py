"""固定随机 MLP 特征映射：原版 DM 的均值匹配特征空间

用法与用户给出的原版写法一致：
    feature_net = RandomMLP(input_dim=1024)
    feature_net.eval()
    for p in feature_net.parameters():
        p.requires_grad_(False)
    real_feat = feature_net(real_z).detach()
    syn_feat = feature_net(synthetic_z)
    loss = (real_feat.mean(0) - syn_feat.mean(0)).pow(2).mean()

网络参数随机初始化后完全冻结；梯度只穿过 synthetic_z。
"""
import torch
import torch.nn as nn


class RandomMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1024,
                 seed=0):
        super().__init__()
        # 固定初始化，不影响调用方的全局 RNG
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(output_dim)),
            )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    """残差随机 MLP：f(x) = x + MLP(x)

    输出维度必须等于输入维度；残差支路让梯度可以近似恒等回传，
    缓解深层随机映射在合成特征上的梯度衰减。
    """

    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1024,
                 seed=0):
        super().__init__()
        if int(output_dim) != int(input_dim):
            raise ValueError("ResidualMLP 要求 output_dim == input_dim")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(output_dim)),
            )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return x + self.net(x)


class MultiRandomMLP(nn.Module):
    """多个独立随机映射，forward 返回 [K, N, D] 的堆叠特征"""

    def __init__(self, n_mappings=3, input_dim=1024, hidden_dim=2048,
                 output_dim=1024, seed=0):
        super().__init__()
        self.n_mappings = int(n_mappings)
        self.nets = nn.ModuleList([
            RandomMLP(input_dim, hidden_dim, output_dim, seed=int(seed) + i)
            for i in range(self.n_mappings)])

    def forward(self, x):
        return torch.stack([net(x) for net in self.nets], dim=0)


def random_feature_mean_loss(feature_net, real_z, syn_z):
    """原版 DM 的线性核均值匹配（在固定随机特征空间中）"""
    real_feat = feature_net(real_z).detach()
    syn_feat = feature_net(syn_z)
    if real_feat.dim() == 3:            # MultiRandomMLP: [K,N,D]
        return (real_feat.mean(1) - syn_feat.mean(1)).pow(2).mean()
    return (real_feat.mean(0) - syn_feat.mean(0)).pow(2).mean()
