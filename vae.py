"""
EdgeVAE (边缘潜空间, 16ch × 8×8, 3× 压缩) + 训练入口

结构:
  Encoder: 32→16→8 (n_down=2), 全程 64 通道 → 1×1 投影到 latent_dim 通道
           空间固定 8×8 (8×8 聚合度优于 16×16: 后者展平 kNN 崩), 通道可调 (latent_dim=16)
           latent = [N, 16, 8, 8] = 1024 元素 (3× 压缩, 适配边缘)

分类头: conv(latent→64, 3×3) + GAP + Linear → 塑造潜空间类判别性

结构 (权重 key 与命名精确对应, 勿随意改名, 否则无法加载已有 ckpt):
  Encoder: conv1(3→64, 3×3) → down×2 [ResBlock(64) + 4×4s2 conv + BN] → mid[ResBlock×2] → 1×1 → mu/logvar
  Decoder: in_conv(1×1, latent→64) → mid[ResBlock×2] → up×2 [ConvT 4×4s2 + BN + ResBlock(64)] → out ResBlock(64→32) → 3×3 → 3ch (tanh)
  ResBlock: conv1/bn1/conv2/bn2 + identity/1×1 shortcut

训练: loss = MSE重建 + AE_BETA1·KL + cls_lambda·CE(分类)
  python vae.py                              # 默认 16ch×8×8
  python vae.py --latent_dim 8 --cls_lambda 0.5
  python vae.py --no_cls                     # 纯重建 VAE (无分类监督, 判别性对比)

产物: checkpoints/autoencoder_edge_s8_c16.pt  (config.SAVE_DIR, cifar100 时带子目录)
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config
from config import get_cifar10_loaders


class ResBlock(nn.Module):
    """残差块: Conv → BN → ReLU → Conv → BN, 加跳跃连接"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ============================================================
# 编码器: 32×32 → 16×16 → 8×8 (全程 64 通道)
# ============================================================

class EdgeEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=16, base_channels=64, n_down=2):
        super().__init__()
        c = base_channels

        # [3, 32, 32] → [c, 32, 32]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # 每个 down: ResBlock + 4×4 stride2 下采样 + BN  (32→16→8)
        down = []
        for _ in range(n_down):
            down.append(nn.Sequential(
                ResBlock(c, c),
                nn.Conv2d(c, c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ))
        self.down = nn.ModuleList(down)

        # 8×8 处两个残差块提特征
        self.mid = nn.Sequential(ResBlock(c, c), ResBlock(c, c))

        # 1×1 投影到潜空间 mu / logvar
        self.mu_conv = nn.Conv2d(c, latent_dim, 1)
        self.logvar_conv = nn.Conv2d(c, latent_dim, 1)

    def forward(self, x):
        h = self.conv1(x)
        for d in self.down:
            h = d(h)
        h = self.mid(h)
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h).clamp(-8, 8)
        return mu, logvar


# ============================================================
# 解码器: 8×8 → 16×16 → 32×32
# ============================================================

class EdgeDecoder(nn.Module):
    def __init__(self, latent_dim=16, out_channels=3, base_channels=64, n_down=2):
        super().__init__()
        c = base_channels

        # [latent_dim, 8, 8] → [c, 8, 8]
        self.in_conv = nn.Sequential(
            nn.Conv2d(latent_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # 8×8 处两个残差块
        self.mid = nn.Sequential(ResBlock(c, c), ResBlock(c, c))

        # 每个 up: ConvT 4×4 stride2 + BN + ResBlock (8→16→32)
        up = []
        for _ in range(n_down):
            up.append(nn.Sequential(
                nn.ConvTranspose2d(c, c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                ResBlock(c, c),
            ))
        self.up = nn.ModuleList(up)

        # 32×32: 降通道残差 + 3×3 出图 (tanh)
        self.out_conv = nn.Sequential(
            ResBlock(c, c // 2),
            nn.Conv2d(c // 2, out_channels, 3, padding=1),
        )

    def forward(self, z):
        h = self.in_conv(z)
        h = self.mid(h)
        for u in self.up:
            h = u(h)
        return torch.tanh(self.out_conv(h))


# ============================================================
# 完整 EdgeVAE
# ============================================================

class EdgeVAE(nn.Module):
    """变分自编码器 (小通道 × 8×8 布局)

    forward 返回 (recon, z, logits, mu, logvar),
    distill.py 与各评估脚本共用一套 API。
    """

    def __init__(self, in_channels=3, latent_dim=16, num_classes=10,
                 base_channels=64, n_down=2):
        super().__init__()
        self.encoder = EdgeEncoder(in_channels, latent_dim, base_channels, n_down)
        self.decoder = EdgeDecoder(latent_dim, in_channels, base_channels, n_down)
        self.latent_dim = latent_dim
        self.spatial_size = 32 // (2 ** n_down)  # 固定 8

        # 分类头: conv(latent→64) + GAP + FC (在 8×8 上做 3×3 卷积再池化, 塑造判别性)
        self.classifier = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, num_classes),
        )

    def encode_mu_logvar(self, x):
        """图片 → (mu, logvar)"""
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """重参数化采样: z = mu + eps * exp(0.5*logvar)"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        """图片 → 采样潜变量 (VAE 训练用)"""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def encode_mu(self, x):
        """图片 → 确定性潜变量 mu (蒸馏/评估用, 无随机性)"""
        return self.encoder(x)[0]

    def decode(self, z):
        """潜变量 → 图片"""
        return self.decoder(z)

    def classify(self, z):
        """潜变量 → 类别预测"""
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classify(z)
        return recon, z, logits, mu, logvar


# ============================================================
# 工厂函数
# ============================================================

def load_edge_vae(device=None, ckpt_name=None):
    """加载冻结的 EdgeVAE (默认 16ch×8×8: autoencoder_edge_s8_c16.pt)

    ckpt 在 config.SAVE_DIR 下 (DATASET=cifar100 时自动带 cifar100 子目录)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = EdgeVAE(latent_dim=16, num_classes=config.NUM_CLASSES).to(device)
    if ckpt_name is None:
        ckpt_name = f"autoencoder_edge_s{ae.spatial_size}_c{ae.latent_dim}.pt"
    ckpt_path = os.path.join(config.SAVE_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"EdgeVAE 未找到: {ckpt_path} "
                                f"(请先训练: python vae.py --latent_dim 16)")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae


# ============================================================
# 训练
# ============================================================

def train(latent_dim=16, epochs=None, no_cls=False, cls_lambda=None):
    if epochs is None:
        epochs = config.AE_EPOCHS
    if cls_lambda is None:
        cls_lambda = 0.0 if no_cls else config.AE_CLS_LAMBDA

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}, 分类监督 lambda: {cls_lambda}")

    train_loader, _, _ = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, augment=True)

    ae = EdgeVAE(latent_dim=latent_dim, num_classes=config.NUM_CLASSES).to(device)
    opt = optim.Adam(ae.parameters(), lr=config.AE_LR)
    print(f"参数量: {sum(p.numel() for p in ae.parameters()):,}")
    print(f"潜空间: [{ae.latent_dim}, {ae.spatial_size}, {ae.spatial_size}]"
          f" ({config.IMAGE_SIZE * config.IMAGE_SIZE * config.IMAGE_CHANNELS}"
          f" / {ae.latent_dim * ae.spatial_size * ae.spatial_size:.0f}x 压缩)")

    tag = "_noclspart" if no_cls else ""
    ckpt_name = f"autoencoder_edge_s{ae.spatial_size}_c{ae.latent_dim}{tag}.pt"
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        ae.train()
        total_loss = rec_loss = kl_sum = cls_sum = 0.0
        n_batch = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            bx, by = x.to(device), y.to(device)
            opt.zero_grad()
            recon, z, logits, mu, logvar = ae(bx)

            rec = F.mse_loss(recon, bx)                                     # [-1,1] 重建
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()      # 逐元素平均
            cls = F.cross_entropy(logits, by)
            loss = rec + config.AE_BETA1 * kl + cls_lambda * cls

            loss.backward()
            opt.step()

            total_loss += loss.item()
            rec_loss += rec.item()
            kl_sum += kl.item()
            cls_sum += cls.item()
            n_batch += 1

        avg_loss = total_loss / n_batch
        print(f"  loss={avg_loss:.4f} rec={rec_loss / n_batch:.4f} "
              f"kl={kl_sum / n_batch:.4f} cls={cls_sum / n_batch:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(config.SAVE_DIR, ckpt_name)
            torch.save({"epoch": epoch, "loss": best_loss,
                        "latent_dim": latent_dim,
                        "spatial_size": ae.spatial_size,
                        "model_state_dict": ae.state_dict()}, ckpt_path)
            print(f"  [保存] {ckpt_path} (loss={best_loss:.4f})")

    print(f"训练完成, 最佳: {ckpt_path}")
    return ae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 EdgeVAE (16ch×8×8)")
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="潜空间通道数 (空间固定 8×8; 默认16, 即 16×8×8)")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认 config.AE_EPOCHS)")
    parser.add_argument("--cls_lambda", type=float, default=None,
                        help="分类监督权重 (默认 config.AE_CLS_LAMBDA=0.1; edge 判别性不足时可加大到 0.5~1.0)")
    parser.add_argument("--no_cls", action="store_true",
                        help="去掉分类监督 (纯重建+KL), 验证潜空间判别性是否来自分类头")
    args = parser.parse_args()
    train(latent_dim=args.latent_dim, epochs=args.epochs,
          no_cls=args.no_cls, cls_lambda=args.cls_lambda)
