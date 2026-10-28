"""
阶段2: 潜空间数据集蒸馏 (核心集选择)

流程:
  1. 用 VAE 编码真实图片 → 潜变量 z (确定性 mu)
  2. 每类做 K-means 聚类, 默认取簇均值中心 (mean) 作合成潜变量
  3. 下游 (exp_ipc.py / exp_method_compare.py): CNN backbone 直接分类 (测试图 VAE 编码后同管道)

中心模式 (--center):
  mean     均值中心: K-means 每簇中心 = 簇内潜变量算术平均 (默认)
           → 直接潜空间分类无需解码, 均值点抹掉簇内方差, 是类分布真正的代表
  medoid   质心: 离簇中心最近的真实潜变量 (若需解码复原图片, 用这个更清晰)
  herding  贪心核心集: 迭代挑选使已选集合均值逼近类均值 (与 medoid 相当)

注意: 早期主线追求解码复原, mean 平均中心解码偏糊 (-8%~-13%) 所以默认 medoid;
现在主线改为潜空间直接分类 (跳过解码), 模糊不再影响精度, 默认切回 mean。
不再解码绘制合成图 (syn_images) —— 下游不需要复原图片。

运行 (默认 = mean 中心 + edge 16ch×8×8 潜空间):
  python distill.py --ipc 50 --subset 1000                      # 默认
  python distill.py --ipc 50 --subset 1000 --center medoid      # 需解码复原时用质心
  python distill.py --ipc 50 --subset 1000 --center herding
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans

import config
from config import get_cifar10_loaders
from vae import load_edge_vae


def medoid(z_flat, ref):
    """质心: 离参考点 (均值/簇中心) 最近的真实潜变量

    与均值中心不同, 质心就是一张真实图片的潜变量, 解码不糊;
    同时它位于簇中心区域, 保留代表性。本质是潜空间里的核心集挑选。
    """
    d = ((z_flat - ref) ** 2).sum(1)
    return z_flat[d.argmin()]


def herding(z_flat, n):
    """Herding 贪心核心集: 每次选使"已选集合均值"最逼近类均值的点 (SRe2L 做法)

    与一次性 K-means medoid 不同, herding 迭代贪心且去重,
    选出的 n 个点均值收敛到类均值, 代表性更强、多样性更好。
    z_flat: [N, D] 潜变量; 返回 [n, D] 选中的真实潜变量
    """
    mu = z_flat.mean(0)
    selected = []
    acc = np.zeros(z_flat.shape[1])
    remain = np.arange(z_flat.shape[0])
    for t in range(n):
        target = (t + 1) * mu - acc          # 使 (acc + z)/(t+1) 最接近 mu
        d = ((z_flat[remain] - target) ** 2).sum(1)
        pos = d.argmin()
        selected.append(remain[pos])
        acc = acc + z_flat[remain[pos]]
        remain = np.delete(remain, pos)      # 去重: 每个点只选一次
    return np.stack([z_flat[i] for i in selected])


def distill_dataset(ipc=1, subset_per_class=None, center_mode="mean", no_cls=False):
    print("\n" + "=" * 60)
    print(f"阶段2: 潜空间蒸馏 (IPC={ipc}, 中心模式: {center_mode}, no_cls={no_cls})")
    print("=" * 60)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载冻结编码器 EdgeVAE (16ch×8×8) ----
    ckpt_name = "autoencoder_edge_s8_c16_noclspart.pt" if no_cls else None
    ae = load_edge_vae(device, ckpt_name=ckpt_name)
    print(f"已加载 VAE (冻结, latent_dim={ae.latent_dim}, spatial={ae.spatial_size})")

    D = ae.latent_dim
    H = W = ae.spatial_size
    C = config.NUM_CLASSES

    # ---- 数据 ----
    train_loader, _, _ = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, augment=False)

    # ---- 编码全部训练集 ----
    print("编码训练集到潜空间...")
    latents_all, labels_all, imgs_all = [], [], []
    with torch.no_grad():
        for x, y in tqdm(train_loader, desc="编码"):
            latents_all.append(ae.encode_mu(x.to(device)).cpu())
            labels_all.append(y)
            imgs_all.append(x)
    latents_all = torch.cat(latents_all, dim=0)  # [N, C_in, H, W]
    labels_all = torch.cat(labels_all, dim=0)
    imgs_all = torch.cat(imgs_all, dim=0)        # [N, 3, 32, 32]

    # ---- 每类取子集 (可选) ----
    if subset_per_class is not None and subset_per_class > 0:
        print(f"每类只取 {subset_per_class} 张真实图片用于蒸馏...")
        g = torch.Generator()
        g.manual_seed(42)
        subset_latents, subset_labels, subset_imgs = [], [], []
        for c in range(C):
            mask = labels_all == c
            z_c = latents_all[mask]
            img_c = imgs_all[mask]
            n_sel = min(subset_per_class, z_c.size(0))  # 不能超过该类实际样本数
            idx = torch.randperm(z_c.size(0), generator=g)[:n_sel]
            subset_latents.append(z_c[idx])
            subset_labels.append(torch.full((n_sel,), c, dtype=torch.long))
            subset_imgs.append(img_c[idx])

        latents_for_init = torch.cat(subset_latents, dim=0)
        labels_for_init = torch.cat(subset_labels, dim=0)
        subset_real_imgs = torch.cat(subset_imgs, dim=0)
        subset_real_labels = torch.cat(subset_labels, dim=0)
        torch.save({"images": subset_real_imgs, "labels": subset_real_labels},
                   os.path.join(config.PT_DIR, f"real_subset_ipc{ipc}.pt"))
        print(f"真实子集已保存 (共 {subset_real_imgs.size(0)} 张)")
    else:
        latents_for_init = latents_all
        labels_for_init = labels_all
        subset_real_imgs = imgs_all
        subset_real_labels = labels_all

    # ---- 核心集选择 / 合成中心: 每类 ipc 个潜变量 ----
    # 统一钳制 ipc: 聚类需要 n_samples >= n_clusters, 取每类最小样本数兜底 (避免 cifar100 每类仅 ~500 时 ipc=500 越界)
    min_n = min(int((labels_for_init == c).sum().item()) for c in range(C))
    if ipc >= min_n:
        ipc = max(min_n - 1, 1)
        print(f"[WARN] ipc 超过每类最小样本 {min_n}, 钳制为 {ipc}")
    print(f"中心模式: {center_mode}...")
    syn_latents = torch.zeros(C, ipc, D, H, W)
    with torch.no_grad():
        for c in range(C):
            mask = labels_for_init == c
            z_c = latents_for_init[mask]
            z_flat = z_c.reshape(z_c.size(0), -1).numpy()
            # 每类样本数不足时钳制 ipc, 避免 KMeans(n_clusters>n_samples) 或 herding 越界
            k_ipc = ipc
            if ipc >= z_flat.shape[0]:
                k_ipc = max(z_flat.shape[0] - 1, 1)
                print(f"[WARN] class {c}: ipc={ipc} >= n_samples={z_flat.shape[0]}, 钳制为 {k_ipc}")
            if center_mode == "herding":
                # 贪心核心集: 直接全局贪心, 无需 K-means (ipc=1 时等价 medoid)
                center = herding(z_flat, k_ipc)
            elif center_mode == "mean":
                # K-means 均值中心: 每簇中心 = 合成点 (直接潜空间分类无需解码, 抹掉簇内方差=类分布代表)
                kmeans = KMeans(n_clusters=k_ipc, random_state=42, n_init="auto")
                kmeans.fit(z_flat)
                center = kmeans.cluster_centers_
            else:
                # medoid: K-means 聚类后每簇取离簇中心最近的真实潜变量
                kmeans = KMeans(n_clusters=k_ipc, random_state=42, n_init="auto")
                kmeans.fit(z_flat)
                centers = []
                for k in range(k_ipc):
                    member = z_flat[kmeans.labels_ == k]
                    centers.append(medoid(member, kmeans.cluster_centers_[k]))
                center = np.stack(centers)
            syn_latents[c] = torch.from_numpy(center).reshape(ipc, D, H, W)

    print(f"合成潜变量: {syn_latents.shape}")

    # ---- 保存 (文件名带中心模式 + vae_tag) ----
    vae_tag = "_noclspart" if no_cls else ""
    save_path = os.path.join(config.PT_DIR,
                             f"syn_latents_ipc{ipc}_{center_mode}{vae_tag}_edge.pt")
    torch.save({
        "syn_latents": syn_latents,
        "ipc": ipc,
        "latent_dim": D,
        "spatial_size": H,
        "method": f"kmeans_{center_mode}",
        "center_mode": center_mode,
        "subset_per_class": subset_per_class,
    }, save_path)
    print(f"合成潜变量已保存: {save_path}")

    return syn_latents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="潜空间蒸馏 (核心集选择)")
    parser.add_argument("--ipc", type=int, default=config.IPC, help="每类合成图片数")
    parser.add_argument("--subset", type=int, default=1000, help="每类只用N张真实图蒸馏 (默认 1000)")
    parser.add_argument("--center", type=str, default="mean",
                        choices=["mean", "medoid", "herding"],
                        help="中心模式: mean=簇均值中心(默认, 直接潜空间分类无需解码, 类分布代表), medoid=质心(离簇中心最近的真实图, 解码复原更清晰), herding=贪心核心集(与 medoid 相当)")
    parser.add_argument("--no_cls", action="store_true",
                        help="用纯重建 VAE (autoencoder_edge_s8_c16_noclspart.pt, 无分类监督), 产物加 _noclspart 后缀")
    args = parser.parse_args()

    distill_dataset(ipc=args.ipc, subset_per_class=args.subset, center_mode=args.center,
                    no_cls=args.no_cls)
