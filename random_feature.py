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


def random_feature_mean_loss(feature_net, real_z, syn_z):
    """原版 DM 的线性核均值匹配（在固定随机特征空间中）"""
    real_feat = feature_net(real_z).detach()
    syn_feat = feature_net(syn_z)
    return (real_feat.mean(0) - syn_feat.mean(0)).pow(2).mean()
