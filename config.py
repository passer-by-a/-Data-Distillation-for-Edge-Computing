"""
全局配置 + 数据加载

流程:
  阶段1: 训练 EdgeVAE (16ch×8×8), 图片 ↔ 潜空间双向映射
  阶段2: 在潜空间中蒸馏, 每类压缩为 IPC 个合成潜变量
  下游: 直接在潜空间做分类 (不解码回像素), 见 backbones.py / exp_*.py

最终产出:
  - 合成潜变量库: [num_classes, IPC, latent_dim, H, W]
"""

import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ==================== 路径 ====================
# 数据集切换: DATASET=cifar100 时所有产物隔离到子目录, cifar10 默认不变
DATASET = os.environ.get("DATASET", "cifar10")   # cifar10 / cifar100
DATA_ROOT = "./data"
SAVE_DIR = "./checkpoints"   # 模型权重
LOG_DIR   = "./logs"          # 日志文件
RESULT_DIR = "./results"      # 图片/可视化
PT_DIR    = "./pt"            # 潜空间产物 (合成潜变量/真实子集 .pt)
if DATASET != "cifar10":
    for _d in ["SAVE_DIR", "LOG_DIR", "RESULT_DIR", "PT_DIR"]:
        globals()[_d] = os.path.join(globals()[_d], DATASET)
for d in [SAVE_DIR, LOG_DIR, RESULT_DIR, PT_DIR]:
    os.makedirs(d, exist_ok=True)


def find_pt(name):
    """查找潜空间产物: 优先 pt/, 找不到 fallback 到 results/ (兼容历史产物)"""
    for d in [PT_DIR, RESULT_DIR]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

# ==================== 设备 (会自动检测, 无需手动改) ====================
DEVICE = "cuda"  # 运行时自动 fallback 到 cpu

# ==================== 数据集 ====================
IMAGE_SIZE       = 32
IMAGE_CHANNELS   = 3
NUM_CLASSES      = 100 if DATASET == "cifar100" else 10
BATCH_SIZE       = 128
NUM_WORKERS      = 4

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# ==================== 阶段1: EdgeVAE (16ch×8×8) ====================
AE_LATENT_DIM = 16      # 潜空间通道数 (空间固定 8×8 → 1024 元素)
AE_LR       = 1e-3
AE_EPOCHS   = 100
AE_BETA1    = 1e-4      # KL散度权重
AE_CLS_LAMBDA = 0.1     # 分类损失权重 (multi-task: 重建 + λ*分类)

# ==================== 阶段2: 潜空间数据集蒸馏 ====================
IPC         = 100        # Images Per Class: 每类合成几张潜变量 (默认 100, 子集 1000; ipc 太小会过拟合/失稳)

# ==================== 评估 ====================
EVAL_BATCH_SIZE = 256


# ============================================================
# 数据加载 (CIFAR-10 / CIFAR-100)
# ============================================================

# ---- 镜像加速: 替换 CIFAR-10 下载地址 ----
# 默认源 www.cs.toronto.edu 在国内可能很慢
# 可设环境变量 CIFAR10_MIRROR 切换镜像，例如:
#   export CIFAR10_MIRROR="https://hf-mirror.com/datasets/cifar10/resolve/main"
# 或者直接把下面 MIRROR_URL 改成你最快的镜像地址
_DEFAULT_URL = torchvision.datasets.CIFAR10.url
_MIRROR_URL = os.environ.get("CIFAR10_MIRROR", "")
if _MIRROR_URL:
    torchvision.datasets.CIFAR10.url = f"{_MIRROR_URL}/cifar-10-python.tar.gz"
    print(f"[镜像] CIFAR-10 下载源已切换: {torchvision.datasets.CIFAR10.url}")


def _dataset_cls():
    """按 DATASET 返回 torchvision 数据集类 (cifar10 / cifar100)"""
    return torchvision.datasets.CIFAR100 if DATASET == "cifar100" \
        else torchvision.datasets.CIFAR10


def get_cifar10_loaders(data_root="./data", batch_size=128, num_workers=4,
                        val_ratio=0.02, augment=True):
    """
    获取 CIFAR-10/100 训练/验证/测试 DataLoader

    返回: train_loader, val_loader, test_loader
    """
    ds_cls = _dataset_cls()

    # 训练数据增强
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # 验证/测试: 无数据增强
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载完整训练集
    full_train = ds_cls(
        root=data_root, train=True, download=True, transform=train_transform
    )

    # 划分训练/验证集
    num_train = len(full_train)
    num_val = int(num_train * val_ratio)
    indices = np.random.RandomState(42).permutation(num_train)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]

    train_dataset = Subset(full_train, train_indices)
    # 验证集使用 eval_transform
    val_dataset = ds_cls(
        root=data_root, train=True, download=False, transform=eval_transform
    )
    val_dataset = Subset(val_dataset, val_indices)

    # 测试集
    test_dataset = ds_cls(
        root=data_root, train=False, download=True, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def denormalize(tensor):
    """将 [-1, 1] 的张量反归一化到 [0, 1]"""
    return (tensor * 0.5 + 0.5).clamp(0, 1)
