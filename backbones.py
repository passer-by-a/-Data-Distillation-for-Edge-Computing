"""
下游 backbone 模型库 + 训练/统计内核 (被 exp_*.py 复用)

四种经典网络改造 stem 以接收潜变量 (默认 edge 16ch×8×8), 空间小 (8×8) 故全部 stride=1
保持空间分辨率, 末尾全局平均池化 + FC。各网络保留其标志性范式:
  VGG       普通 3×3 卷积堆叠
  ResNet    残差块 (3×3 + shortcut)
  MobileNet depthwise separable 卷积 (倒残差瓶颈)
  ViT       2×2 patch embed (8×8→16 token) + Transformer 编码器 (小数据用小模型 + dropout)

所有 backbone 均含 Dropout(p=0.3)、参数量压到 ~28-32K, 防止小蒸馏集过拟合。

对外接口:
  BACKBONES                          名称 → 模型类
  train_eval_backbone(...)           在潜变量上训练并评估 (标准化 / cosine 衰减 / 测试缓存)
  count_params / count_flops         参数量 / 单张推理 FLOPs
  save_json / plot_curves            结果落盘 / 双面板 loss 图
  load_syn_center                    按 center_mode 查找蒸馏产物

命令行入口见 exp_ipc.py / exp_method_compare.py。
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from utils import encode_test_latents


# ============================================================
# 四种 backbone (stem 改造接收 [N, 16, 8, 8] 潜变量)
# ============================================================

def _stem(in_ch, out_ch):
    """stem: 16 通道潜变量 → 常规通道数, 保持 8×8 空间"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGGBackbone(nn.Module):
    """VGG 风格: 3×3 卷积堆叠, 通道递增 (宽度按参数量对齐削减到 ~31K, 缓解小蒸馏集过拟合)"""
    def __init__(self, in_ch=16, num_classes=10, width=(18, 22, 26), dropout_p=0.3):
        super().__init__()
        layers = list(_stem(in_ch, width[0]))
        prev = width[0]
        for w in width:
            layers += [nn.Conv2d(prev, w, 3, padding=1, bias=False),
                       nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False),
                       nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            prev = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout_p),
                                  nn.Linear(width[-1], num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ResNetBackbone(nn.Module):
    """ResNet 风格: 残差块堆叠"""
    def __init__(self, in_ch=16, num_classes=10, width=23, blocks=3, dropout_p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _stem(in_ch, width),
            *[ResBlock(width) for _ in range(blocks)],
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout_p),
                                  nn.Linear(width, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class InvResidual(nn.Module):
    """MobileNet 风格: 倒残差 depthwise separable (expand → depthwise → project)"""
    def __init__(self, in_ch, out_ch, expand=6, stride=1):
        super().__init__()
        hid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv1 = nn.Conv2d(in_ch, hid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.conv2 = nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.conv3 = nn.Conv2d(hid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        res = x
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + res if self.use_res else out


class MobileNetBackbone(nn.Module):
    """MobileNet 风格: 倒残差 depthwise separable 堆叠 (宽度按参数量对齐削减到 ~28K)"""
    def __init__(self, in_ch=16, num_classes=10, width=16, dropout_p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _stem(in_ch, width),
            InvResidual(width, width, expand=6),
            InvResidual(width, width * 2, expand=6, stride=1),
            InvResidual(width * 2, width * 2, expand=6),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout_p),
                                  nn.Linear(width * 2, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ViTBlock(nn.Module):
    """ViT 风格 Transformer block: pre-norm MHSA + MLP

    qkv/proj 用 nn.Linear (thop 能统计其 FLOPs); attention 内部打分用
    scaled_dot_product_attention (token 少, 占比小)。
    """
    def __init__(self, dim, heads=4, mlp_ratio=2, dropout_p=0.1):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        B, T, D = x.shape                              # [N, tokens, dim]
        qkv = self.qkv(self.norm1(x)).reshape(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)           # 各 [N, heads, T, D/heads]
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(attn))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """ViT 风格: patch embed (8×8→16 token) + Transformer 编码器 (宽度按参数量对齐到 ~29K)

    小蒸馏集 (ipc=100 → 1000 点) 易过拟合, 故用浅层窄模型 + dropout, 并给 pos/cls token
    标准截断正态初始化 (ViT 常规做法, 避免零初始化下位置信息学不动)。
    输入分辨率与初始化不一致时 (如原图直训 3×32×32) 双三次插值位置编码, 与 ViT 常规做法一致。
    """
    def __init__(self, in_ch=16, num_classes=10, embed_dim=32, depth=3,
                 heads=4, patch=2, spatial=8, mlp_ratio=2, dropout_p=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch, stride=patch)
        self.grid = spatial // patch
        n_tokens = 1 + self.grid ** 2                  # cls token + patch token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout_p)
        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, heads, mlp_ratio, dropout_p) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _pos_embed_for(self, n_tokens):
        """token 数与初始化不同时插值位置编码 (patch token 网格数需为平方数)"""
        if n_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        grid = int(round((n_tokens - 1) ** 0.5))
        pe = self.pos_embed[:, 1:].reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(grid, grid), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)
        return torch.cat([self.pos_embed[:, :1], pe], dim=1)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)         # [N, T, dim]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self._pos_embed_for(x.size(1) + 1)
        x = self.blocks(self.pos_drop(x))
        return self.head(self.norm(x)[:, 0])                       # cls token 分类


class PixelResBlock(nn.Module):
    """像素空间残差块 (stride 可为 2; 升维/降采样时 shortcut 走 1×1 conv)"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt)


class PixelResNet(nn.Module):
    """像素空间 ResNet (输入 3×32×32 → 8×8, 约 0.7M 参数)

    专供"在原图上实现的基线"下游训练用; 潜空间 ours 仍用参数量 ~28K 的 ResNetBackbone。
    CIFAR 风格 stem (3×3 stride1) + 3 个 stage (后两段 stride2 下采样), 容量足以在 32×32
    原图上收敛 —— 不用 ResNetBackbone 是因为它是为 8×8 潜变量设计的, 直接吃原图会欠拟合
    (实测 train loss 长期停在 ~1.7/4.0)。注意: 不注册进 BACKBONES, 以免影响按 BACKBONES
    遍历的其它实验。
    """
    def __init__(self, in_ch=3, num_classes=10, width=32, blocks=(2, 2, 2),
                 dropout_p=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        layers, prev = [], width
        for si, nb in enumerate(blocks):
            out_ch = width * (2 ** si)
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                layers.append(PixelResBlock(prev, out_ch, stride))
                prev = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout_p),
                                  nn.Linear(prev, num_classes))

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


BACKBONES = {
    "vgg": VGGBackbone,
    "resnet": ResNetBackbone,
    "mobilenet": MobileNetBackbone,
    "vit": ViTBackbone,
}


# ============================================================
# 统计 & 训练
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, x):
    """单张输入的推理 FLOPs (调 thop 库, 避免手写 hook 漏算子)

    统计口径: thop 按各算子 MACs 汇总, 这里 ×2 转为 FLOPs。
    相比手写 hook 会额外计入 BN/逐元素算子 (mobilenet 差异约 +7%, 属库口径)。
    需要环境装有 thop: pip install thop
    """
    try:
        from thop import profile
    except ImportError as e:
        raise ImportError("请先安装 thop: pip install thop") from e
    was_training = model.training
    model.eval()  # BN 在 batch=1 推理模式下报错, 统计用 running stats 合理
    macs, _ = profile(model, inputs=(x,), verbose=False)
    if was_training:
        model.train()
    return int(macs) * 2


def train_eval_backbone(backbone_cls, z_train, labels, test_loader, ae, device,
                        seed=0, epochs=100, lr=0.001, return_curve=False,
                        test_latents=None, lr_schedule="cosine",
                        return_acc_curve=False):
    """在合成潜变量 [N, 16, 8, 8] 上训练 backbone, 潜空间直接分类

    标准化用 flatten 后 per-dim z-score, 测试图经 ae.encode_mu 编码。
    test_latents=(z_test [N, C_in*H*W] 未标准化, y_test) 可选缓存;
    lr_schedule="cosine" 时 100 epoch 内 lr 余弦衰减到 ~0 (默认, 消除小数据饱和后
    Adam 固定步长随机游走造成的 test loss 尾部尖峰); "none"=固定 lr (历史行为)。
    return_curve=True 时返回 (acc, train_curve, test_curve)。
    return_acc_curve=True (需同时 return_curve=True) 时额外返回逐 epoch 测试准确率曲线,
    返回 (acc, train_curve, test_curve, acc_curve)。逐 epoch 准确率只用于画曲线诊断,
    不作为"挑最好 epoch"的模型选择依据 (报告口径仍是最后一个 epoch 的 acc)。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = z_train.size(0)
    zf = z_train.flatten(1)
    z_mean = zf.mean(0, keepdim=True)
    z_std = zf.std(0, keepdim=True) + 1e-8
    z_train = ((zf - z_mean) / z_std).reshape(z_train.shape)

    model = backbone_cls(in_ch=z_train.size(1), num_classes=config.NUM_CLASSES).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if lr_schedule == "cosine" else None

    # 测试潜变量 (flat) + 重建为模型输入形状
    z_test = y_test = None
    if test_latents is not None:
        z_test, y_test = test_latents
    elif return_curve:
        z_test, y_test = encode_test_latents(test_loader, ae, device)

    z_train = z_train.to(device)
    labels = labels.to(device)
    model.train()
    loss_curve = []
    test_loss_curve = [] if return_curve else None
    acc_curve = [] if (return_curve and return_acc_curve) else None
    for _ in range(epochs):
        epoch_loss, n_batch = 0.0, 0
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = criterion(model(z_train[idx]), labels[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1
        loss_curve.append(epoch_loss / max(n_batch, 1))
        # 每 epoch 泛化 CE loss (测试潜变量缓存, 前向很便宜)
        if return_curve:
            model.eval()
            with torch.no_grad():
                t_loss, n_b, n_correct, n_total = 0.0, 0, 0, 0
                for i in range(0, z_test.size(0), 128):
                    zt = ((z_test[i:i + 128] - z_mean) / z_std).reshape(
                        -1, z_train.size(1), *z_train.shape[2:])
                    yb = y_test[i:i + 128].to(device)
                    logits = model(zt.to(device))
                    t_loss += criterion(logits, yb).item()
                    if acc_curve is not None:      # 同一遍前向顺手统计准确率, 无额外开销
                        n_correct += (logits.argmax(1) == yb).sum().item()
                        n_total += yb.size(0)
                    n_b += 1
                test_loss_curve.append(t_loss / max(n_b, 1))
                if acc_curve is not None:
                    acc_curve.append(n_correct / max(n_total, 1) * 100)
            model.train()
        if sched is not None:
            sched.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        if z_test is not None:
            for i in range(0, z_test.size(0), 128):
                zt = ((z_test[i:i + 128] - z_mean) / z_std).reshape(
                    -1, z_train.size(1), *z_train.shape[2:])
                pred = model(zt.to(device)).argmax(1)
                correct += (pred.cpu() == y_test[i:i + 128]).sum().item()
                total += min(128, z_test.size(0) - i)
        else:
            for bx, by in test_loader:
                z_test_b = ae.encode_mu(bx.to(device)).flatten(1)
                z_test_b = ((z_test_b - z_mean.to(device)) / z_std.to(device)).reshape(
                    -1, z_train.size(1), *z_train.shape[2:])
                pred = model(z_test_b).argmax(1)
                correct += (pred.cpu() == by).sum().item()
                total += by.size(0)
    acc = correct / total * 100
    if return_curve:
        if acc_curve is not None:
            return acc, loss_curve, test_loss_curve, acc_curve
        return acc, loss_curve, test_loss_curve
    return acc


def save_json(results, meta, name):
    """把 backbone 对比结果落盘为 JSON (acc±std / params / FLOPs / 延迟 / 能耗 / 曲线)"""
    import json
    from datetime import datetime
    meta["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(config.RESULT_DIR, name)
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"数据已保存: {path}")
    return path


def plot_curves(curves, path, title):
    """损失曲线图拆成两张单面板 (无大标题, 论文友好): {stem}_train.png / {stem}_test.png

    curves: name -> (acc, train, test)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stem = path[:-4] if path.endswith(".png") else path
    for split, key in (("train", 1), ("test", 2)):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, (acc, train_c, test_c) in curves.items():
            c = train_c if key == 1 else test_c
            if c is None:
                continue
            ax.plot(c, lw=1.8, marker="o", markevery=5, markersize=6,
                    label=f"{name} ({acc:.2f}%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = f"{stem}_{split}.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"损失曲线已保存: {out}")


def load_syn_center(center, ipc, no_cls=False):
    """按 center_mode 找合成潜变量产物

    syn_latents_ipc{ipc}_{center}[_noclspart]_edge.pt
    no_cls 时读无分类头 VAE 的蒸馏产物 (文件名带 _noclspart)
    """
    vae_tag = "_noclspart" if no_cls else ""
    return config.find_pt(f"syn_latents_ipc{ipc}_{center}{vae_tag}_edge.pt")
