"""
方法对比: Ours(VAE+K-means, 潜空间) vs 三种在**原图**上实现的基线 (DC/EL2N/Forgetting)

口径 (--baseline_repr pixel, 默认):
  - Ours                : 冻结 EdgeVAE 潜空间 (16×8×8) 逐类 K-means 簇均值, 下游 ResNetBackbone
                          (in_ch=16, ~28K) 直接吃潜变量 (复用 pt/syn_latents_ipc{ipc}_mean_edge.pt);
  - Gradient matching   : 在原始 32×32 图像上做 DC 风格梯度匹配精炼, 产物训练像素空间 PixelResNet;
    (DC)
  - EL2N / Forgetting   : 在原始图像池上训练代理模型打分选 top-k 真实原图, 产物训练 PixelResNet。

像素侧下游 / 代理网络统一用更大容量的 PixelResNet (~0.7M 参数, CIFAR 风格 3-stage 下采样到 8×8),
避免 32×32 原图上欠拟合 (ResNetBackbone 直接吃原图 train loss 长期停在 ~1.7/4.0);
ours 的潜空间网络仍为 ResNetBackbone (~28K)。

--baseline_repr latent 时三个基线改为在冻结 VAE 潜空间上实现 (旧口径, 同表征对比)。

公平口径:
  - 同一源池 real_subset (与蒸馏同池, 每类 1000 张真实图);
  - 每类恰好 ipc 个点, 50 epoch, lr=0.001 余弦衰减, 3 seeds, 曲线取 seed 平均;
  - 逐 epoch 测试准确率经 train_eval_backbone(return_acc_curve=True) 记录, 只作诊断,
    报告准确率仍是最后一个 epoch 的值 (不按曲线峰值挑 epoch)。

运行:
  python exp_method_compare.py                              # ipc=100, 3 seeds, 像素基线
  DATASET=cifar100 python exp_method_compare.py --ipc 50
出图: figures/method_compare_{dataset}_ipc{ipc}_{baseline_repr}_*.pdf (+ .png)
落盘: results/method_compare_{dataset}_ipc{ipc}_{baseline_repr}.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from backbones import (BACKBONES, ResNetBackbone, PixelResNet, count_params,
                       train_eval_backbone)
from config import get_cifar10_loaders
from utils import encode_test_latents, flat_test_cache
from vae import load_edge_vae
from exp_refine import gradient_refine, kmeans_centroids
from plot_acc_loss import curve_bounds, square_grid_fixed_x, save

C = config.NUM_CLASSES
LATENT = (16, 8, 8)
ORDER = ["ours", "gradmatch", "eln2", "forgetting"]
LABELS = {"ours": "Ours (VAE+K-means)", "gradmatch": "Gradient matching (DC)",
          "eln2": "EL2N", "forgetting": "Forgetting"}
# (颜色, marker, 线型): 四重区分 (颜色/形状/线型/文字), 灰度打印也可分
STYLE = {"ours":       ("#d62728", "o", "-"),     # 本文方法沿用论文里的红
         "gradmatch":  ("#1f77b4", "s", "--"),
         "eln2":       ("#2ca02c", "^", "-."),
         "forgetting": ("#ff7f0e", "v", ":")}
FIG_DIR = "figures"
RESULT_DIR = "./results"
# 4 方法 × 4 backbone 网格图的列顺序 (列 = 模型, 每格 4 条方法线)
BB_ORDER = ["vgg", "resnet", "mobilenet", "vit"]
BB_NAMES = {"vgg": "VGG", "resnet": "ResNet", "mobilenet": "MobileNet", "vit": "ViT"}
BB_TAGS = ["a", "b", "c", "d"]
ELN2_NOISE_FRAC = 0.2        # eln2: 剔除最高分噪声段的比例 (Data Diet §1.3)


# ============================================================
# 源池 / 代理模型打分 (EL2N, Forgetting)
# ============================================================

def load_pool(ipc, subset=1000):
    """源池: real_subset_ipc{ipc}.pt (每类 subset 张真实图, 与蒸馏同一池)"""
    p = config.find_pt(f"real_subset_ipc{ipc}.pt")
    if p is None:                                  # 找不到同 ipc 就用任一 real_subset
        p = next(os.path.join(config.PT_DIR, f) for f in sorted(os.listdir(config.PT_DIR))
                 if f.startswith("real_subset_ipc") and f.endswith(".pt"))
    rs = torch.load(p, map_location="cpu", weights_only=True)
    imgs, labels = rs["images"], rs["labels"]
    print(f"源池: {os.path.basename(p)} {imgs.size(0)} 张 (每类 {subset})")
    return imgs, labels


def train_probe(pool, labels, epochs, device, n_runs=3, in_ch=3,
                proxy_cls=ResNetBackbone):
    """在源池上训小 ResNet 代理, 按 Deep Learning on a Data Diet 算 forget + EL2N 分

    EL2N(x,y) = 𝔼_θ ‖σ(f_θ(x)) − y‖₂ (softmax 概率 − one-hot 的 L2 范数);
    forgetting = 训练过程中"对→错"翻转次数。两者都对 n_runs 次随机初始化取平均。
    注意: 每个 run 必须"先 manual_seed 再建模型", 否则初始权重不受 seed 控制。
    返回 (forget, eln2), 与 pool 样本序对齐。
    """
    C = config.NUM_CLASSES
    N = pool.size(0)
    xy = pool.to(device)
    yy = labels.to(device)
    onehot = torch.eye(C, device=device)[yy]
    crit = nn.CrossEntropyLoss(reduction="none")

    forget_acc = torch.zeros(N, dtype=torch.float64)
    el2n_acc = torch.zeros(N, dtype=torch.float64)
    for run in range(n_runs):
        torch.manual_seed(run * 7 + 1)                # 必须在建模型之前
        model = proxy_cls(in_ch=in_ch, num_classes=C).to(device)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        correct_prev = torch.zeros(N, dtype=torch.bool, device=device)
        forget = torch.zeros(N, dtype=torch.long, device=device)
        el2n = torch.zeros(N, device=device)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(N)
            for i in range(0, N, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss = crit(model(xy[idx]), yy[idx]).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out = torch.cat([model(xy[i:i + 1024])   # 分块前向, 避免整池进显存
                                 for i in range(0, N, 1024)], 0)
                el2n += (torch.softmax(out, 1) - onehot).norm(dim=1)
                correct_now = out.argmax(1) == yy
                forget += (correct_prev & ~correct_now).long()
                correct_prev = correct_now
        forget_acc += forget.double().cpu() / n_runs
        el2n_acc += (el2n.double().cpu() / epochs) / n_runs
        del model
    return forget_acc, el2n_acc


def select_eln2(sc, k, noise_frac, class_ids):
    """剔除最高分噪声段后仍取 EL2N 最高的 k 张 (Data Diet §1.3)"""
    order = np.argsort(np.asarray(sc))[::-1]
    n = len(order)
    cut = min(int(n * noise_frac), max(0, n - k))      # 保证能选满 k
    return class_ids[order[cut:cut + k]]


def load_latent_product(name):
    """读潜空间产物, 兼容两种布局:
       5D [C, ipc, D, H, W]  (distill.py / exp_refine 的 centroid 产物)
       4D [C*ipc, D, H, W]   (exp_refine 的 refined 产物, 点已展平)
    返回 (已展平的 [C*ipc,D,H,W], 原始张量, 路径)
    """
    p = config.find_pt(name)
    if p is None:
        raise FileNotFoundError(f"缺潜空间产物 {name}")
    raw = torch.load(p, map_location="cpu", weights_only=True)["syn_latents"]
    if raw.dim() == 5:
        flat = raw.reshape(raw.size(0) * raw.size(1), *raw.shape[2:])
    elif raw.dim() == 4:
        flat = raw
    else:
        raise ValueError(f"{name} 的 syn_latents 维度异常: {tuple(raw.shape)}")
    return flat, raw, p


def build_dc(z_centroid_raw, z_pool, labels, ipc, T_dc, device):
    """DC 梯度匹配精炼: 优先复用已有产物, 缺失时就地精炼并把结果缓存到 pt/

    就地精炼与 exp_refine.py 用同一个 gradient_refine 内核、同一份 K-means 质心初值,
    只是跳过它附带的多 backbone 下游评估 (这里不需要)。
    """
    name = f"syn_latents_ipc{ipc}_refined_edge.pt"
    if config.find_pt(name) is not None:
        flat, _, p = load_latent_product(name)
        return flat, f"DC 梯度匹配精炼 (复用缓存 {os.path.basename(p)})"
    print(f"  未找到 {name}, 就地做 DC 梯度精炼 (T_dc={T_dc}, 与 exp_refine 同一内核)...")
    syn_labels = torch.arange(C).repeat_interleave(ipc)
    refined, stat = gradient_refine(z_centroid_raw, syn_labels, z_pool, labels, C,
                                    device, T_dc=T_dc)
    out = os.path.join(config.PT_DIR, name)
    torch.save({"syn_latents": refined, "ipc": ipc, "method": "dc_gradient_refine"}, out)
    print(f"  精炼完成 {stat['time_s']:.1f}s → 已缓存 {out}")
    return refined, f"DC 梯度精炼 (就地 T_dc={T_dc}, {stat['time_s']:.0f}s)"


def build_dc_pixel(pool, labels, ipc, T_dc, device):
    """像素空间 DC: 以像素 K-means 簇均值为初值做梯度匹配精炼 (产物接像素空间分类器)

    与 exp_refine.py 同一个 gradient_refine 内核, 只是输入从潜变量换成原始 32×32 图像。
    产物缓存到 pt/syn_pixels_ipc{ipc}_refined_dc.pt 避免重复精炼。
    """
    name = f"syn_pixels_ipc{ipc}_refined_dc.pt"
    p = config.find_pt(name)
    if p is not None:
        raw = torch.load(p, map_location="cpu", weights_only=True)["syn_pixels"]
        if raw.dim() == 5:                      # [C, ipc, 3, 32, 32] → 展平
            flat = raw.reshape(raw.size(0) * raw.size(1), *raw.shape[2:])
        else:                                   # [C*ipc, 3, 32, 32] 已展平
            flat = raw
        return flat, f"像素 DC 梯度精炼 (复用缓存 {os.path.basename(p)})"
    n_syn = C * ipc
    syn_bs = min(n_syn, 1000)                  # 每步合成子集 (显存受限; CIFAR-10 时=全量)
    t_eff = T_dc                               # T_dc 与 CIFAR-10 保持一致
    print(f"  未找到 {name}, 就地做像素 DC (K-means 质心初值 + 梯度匹配, "
          f"T_dc={t_eff}, syn_bs={syn_bs}/{n_syn})...")
    centers, kstat = kmeans_centroids(pool, labels, ipc, C)
    syn_labels = torch.arange(C).repeat_interleave(ipc)
    refined, rstat = gradient_refine(centers, syn_labels, pool, labels, C, device,
                                     T_dc=t_eff, syn_bs=syn_bs, proxy_cls=PixelResNet)
    if torch.isnan(refined).any():
        raise RuntimeError(f"像素 DC 精炼出现 NaN (T_dc={t_eff}, syn_bs={syn_bs}); "
                           f"请调低 syn_lr; 不缓存该产物")
    out = os.path.join(config.PT_DIR, name)
    torch.save({"syn_pixels": refined, "ipc": ipc, "method": "dc_gradient_refine_pixel",
                "syn_bs": syn_bs, "T_dc": t_eff}, out)
    print(f"  精炼完成 {rstat['time_s']:.0f}s (K-means {kstat['time_s']:.0f}s) → 已缓存 {out}")
    return refined, (f"像素 DC 梯度精炼 (T_dc={t_eff}, syn_bs={syn_bs}, "
                     f"{rstat['time_s']:.0f}s)")


def build_representations_pixel(pool, labels, ipc, probe_epochs, T_dc, device):
    """ours = 冻结 VAE 潜空间 (K-means 质心); 其余三个基线 = 原图像素空间实现, 产物接像素分类器

    返回 {method: (data, y, repr, 来源说明)}; repr ∈ {"latent", "pixel"}。
    """
    reps = {}
    z_ours, _, p_ours = load_latent_product(f"syn_latents_ipc{ipc}_mean_edge.pt")
    reps["ours"] = (z_ours, torch.arange(C).repeat_interleave(ipc), "latent",
                    f"K-means 簇均值 (潜空间 {os.path.basename(p_ours)})")
    z_dc, src_dc = build_dc_pixel(pool, labels, ipc, T_dc, device)
    reps["gradmatch"] = (z_dc, torch.arange(C).repeat_interleave(ipc), "pixel", src_dc)

    print(f"\n像素空间训练代理模型 {probe_epochs} epochs (in_ch={pool.size(1)}) "
          f"给 EL2N/Forgetting 打分...")
    forget_sc, eln2_sc = train_probe(pool, labels, probe_epochs, device,
                                      in_ch=pool.size(1), proxy_cls=PixelResNet)
    for m in ("eln2", "forgetting"):
        glob = []
        for c in range(C):
            mask = (labels == c)
            class_ids = np.nonzero(mask.numpy())[0]
            k = min(ipc, int(mask.sum()) - 1)
            if m == "eln2":
                glob.extend(select_eln2(eln2_sc[mask].numpy(), k, ELN2_NOISE_FRAC,
                                        class_ids).tolist())
            else:
                p = np.argsort(forget_sc[mask].numpy(), kind="stable")[::-1][:k]
                glob.extend(class_ids[p].tolist())
        sel = np.array(sorted(glob))
        reps[m] = (pool[sel], labels[sel].clone(), "pixel",
                   f"像素池上代理模型打分选 top-{ipc}/类 (seed 固定)")
        print(f"  [{LABELS[m]}] 选中 {len(sel)} 张真实原图")
    return reps


def build_representations(z_pool, labels, ipc, probe_epochs, T_dc, device):
    """返回 {method: (z [C*ipc,16,8,8], y, repr, 来源说明)} (全部在潜空间, 旧口径)"""
    reps = {}
    z_ours, z_ours_raw, p_ours = load_latent_product(f"syn_latents_ipc{ipc}_mean_edge.pt")
    reps["ours"] = (z_ours, torch.arange(C).repeat_interleave(ipc), "latent",
                    f"K-means 簇均值 ({os.path.basename(p_ours)})")
    z_dc, src_dc = build_dc(z_ours_raw, z_pool, labels, ipc, T_dc, device)
    reps["gradmatch"] = (z_dc, torch.arange(C).repeat_interleave(ipc), "latent", src_dc)

    print(f"\n训练代理模型 {probe_epochs} epochs (潜空间池 in_ch=16) 给 EL2N/Forgetting 打分...")
    forget_sc, eln2_sc = train_probe(z_pool, labels, probe_epochs, device, in_ch=z_pool.size(1))
    for m in ("eln2", "forgetting"):
        glob = []
        for c in range(C):
            mask = (labels == c)
            class_ids = np.nonzero(mask.numpy())[0]
            k = min(ipc, int(mask.sum()) - 1)
            if m == "eln2":
                glob.extend(select_eln2(eln2_sc[mask].numpy(), k, ELN2_NOISE_FRAC,
                                        class_ids).tolist())
            else:
                p = np.argsort(forget_sc[mask].numpy(), kind="stable")[::-1][:k]
                glob.extend(class_ids[p].tolist())
        sel = np.array(sorted(glob))
        reps[m] = (z_pool[sel], labels[sel].clone(), "latent",
                   f"代理模型打分选 top-{ipc}/类 (seed 固定)")
        print(f"  [{LABELS[m]}] 选中 {len(sel)} 个真实潜向量")
    return reps


def run(ipc=100, seeds=3, epochs=50, probe_epochs=8, T_dc=1000,
        baseline_repr="pixel"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print(f"方法对比 | {config.DATASET} | ipc={ipc} ({ipc * C} 点) | backbone=ResNet | "
          f"baseline_repr={baseline_repr} | seeds={seeds} | epochs={epochs} | {device}")
    print("=" * 96)

    ae = load_edge_vae(device)
    _, _, test_loader = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS)
    test_latents = encode_test_latents(test_loader, ae, device)   # ours(潜空间) 用
    test_pixels = flat_test_cache(test_loader)                    # 像素基线用

    pool, labels = load_pool(ipc, 1000)                 # 与蒸馏同池 (真实原图)
    if baseline_repr == "pixel":
        reps = build_representations_pixel(pool, labels, ipc, probe_epochs, T_dc, device)
    else:
        print("编码源池到潜空间 (冻结 EdgeVAE)...")
        with torch.no_grad():
            z_pool = torch.cat([ae.encode_mu(pool[i:i + 256].to(device)).cpu()
                                for i in range(0, pool.size(0), 256)])
        reps = build_representations(z_pool, labels, ipc, probe_epochs, T_dc, device)

    results = {}
    for m in ORDER:
        data, y, rep, src = reps[m]
        cache = test_latents if rep == "latent" else test_pixels
        net_ae = ae if rep == "latent" else None
        net_cls = PixelResNet if rep == "pixel" else ResNetBackbone
        n_par = count_params(net_cls(in_ch=data.size(1), num_classes=C))
        print(f"\n[{LABELS[m]}] {data.size(0)} 点 | repr={rep} | "
              f"net={net_cls.__name__}({n_par:,} params) | {src}")
        accs, tr_all, te_all, ac_all = [], [], [], []
        for s in range(seeds):
            acc, tr, te, ac = train_eval_backbone(
                net_cls, data, y, test_loader, net_ae, device, seed=s,
                epochs=epochs, return_curve=True, return_acc_curve=True,
                test_latents=cache, lr_schedule="cosine")
            accs.append(acc); tr_all.append(tr); te_all.append(te); ac_all.append(ac)
        mean = lambda cs: np.mean([np.asarray(c, float) for c in cs], axis=0).tolist()
        results[m] = {"acc": float(np.mean(accs)), "std": float(np.std(accs)),
                      "seeds_acc": accs, "source": src, "repr": rep,
                      "net": net_cls.__name__, "params": n_par,
                      "train_curve": mean(tr_all), "test_curve": mean(te_all),
                      "acc_curve": mean(ac_all)}
        print(f"  → {seeds}-seed: {results[m]['acc']:.2f}±{results[m]['std']:.2f}  "
              f"| 最好 epoch acc={max(results[m]['acc_curve']):.2f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {"meta": {"exp": "method comparison (loss + acc vs epoch, ResNet)",
                    "dataset": config.DATASET, "ipc": ipc, "points": ipc * C,
                    "repr": "ours=latent (frozen EdgeVAE 16x8x8); baselines="
                            + ("pixel 3x32x32 (implemented on raw images)"
                               if baseline_repr == "pixel"
                               else "latent (frozen EdgeVAE 16x8x8)"),
                    "baseline_repr": baseline_repr, "backbone": "resnet",
                    "latent_net": "ResNetBackbone (in_ch=16, ~28K params, for ours)",
                    "pixel_net": "PixelResNet (in_ch=3, 3-stage CIFAR-style, ~0.7M params, "
                                 "for pixel baselines)",
                    "seeds": seeds, "epochs": epochs, "lr": 0.001,
                    "lr_schedule": "cosine", "probe_epochs": probe_epochs,
                    "T_dc": T_dc,
                    "dc_pixel_init": "pixel K-means cluster mean",
                    "source_pool": "real_subset (与蒸馏同池)",
                    "methods": ORDER,
                    "note": "curves = mean over seeds; reported acc = last epoch "
                            "(no epoch selection on test set)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
           "results": results}
    jp = os.path.join(RESULT_DIR,
                      f"method_compare_{config.DATASET}_ipc{ipc}_{baseline_repr}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {jp}")
    return out, jp


def run_multi_backbone(ipc=100, seeds=3, epochs=50, probe_epochs=8, T_dc=1000,
                       baseline_repr="pixel"):
    """4 方法 × 4 backbone 网格: 表征只构建一次, 对每个 backbone 逐方法训练

    ours = 冻结 VAE 潜空间 (in_ch=16); 三基线 = 像素空间 (in_ch=3, 同 --baseline_repr
    口径)。同一列是同一个 backbone 家族, 只是输入通道数按表征空间调整。
    落盘 results/method_x_backbone_{dataset}_ipc{ipc}_{baseline_repr}.json
    """
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print(f"4 方法 × 4 backbone | {config.DATASET} | ipc={ipc} ({ipc * C} 点) | "
          f"baseline_repr={baseline_repr} | seeds={seeds} | epochs={epochs} | {device}")
    print("=" * 96)

    ae = load_edge_vae(device)
    _, _, test_loader = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS)
    test_latents = encode_test_latents(test_loader, ae, device)   # ours(潜空间) 用
    test_pixels = flat_test_cache(test_loader)                    # 像素基线用

    pool, labels = load_pool(ipc, 1000)
    if baseline_repr == "pixel":
        reps = build_representations_pixel(pool, labels, ipc, probe_epochs, T_dc, device)
    else:
        print("编码源池到潜空间 (冻结 EdgeVAE)...")
        with torch.no_grad():
            z_pool = torch.cat([ae.encode_mu(pool[i:i + 256].to(device)).cpu()
                                for i in range(0, pool.size(0), 256)])
        reps = build_representations(z_pool, labels, ipc, probe_epochs, T_dc, device)

    results = {}
    for bb in BB_ORDER:
        cls = BACKBONES[bb]
        results[bb] = {}
        for m in ORDER:
            data, y, rep, src = reps[m]
            cache = test_latents if rep == "latent" else test_pixels
            net_ae = ae if rep == "latent" else None
            n_par = count_params(cls(in_ch=data.size(1), num_classes=C))
            print(f"\n[{BB_NAMES[bb]} × {LABELS[m]}] {data.size(0)} 点 | repr={rep} | "
                  f"in_ch={data.size(1)} | {cls.__name__}({n_par:,} params)")
            accs, tr_all, ac_all = [], [], []
            for s in range(seeds):
                acc, tr, _, ac = train_eval_backbone(
                    cls, data, y, test_loader, net_ae, device, seed=s,
                    epochs=epochs, return_curve=True, return_acc_curve=True,
                    test_latents=cache, lr_schedule="cosine")
                accs.append(acc); tr_all.append(tr); ac_all.append(ac)
            mean = lambda cs: np.mean([np.asarray(c, float) for c in cs], axis=0).tolist()
            results[bb][m] = {"acc": float(np.mean(accs)), "std": float(np.std(accs)),
                              "seeds_acc": accs, "source": src, "repr": rep,
                              "net": cls.__name__, "params": n_par,
                              "train_curve": mean(tr_all), "acc_curve": mean(ac_all)}
            print(f"  → {seeds}-seed: {results[bb][m]['acc']:.2f}±"
                  f"{results[bb][m]['std']:.2f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {"meta": {"exp": "4 methods × 4 backbones (acc + train loss vs epoch)",
                    "dataset": config.DATASET, "ipc": ipc, "points": ipc * C,
                    "baseline_repr": baseline_repr,
                    "repr": "ours=latent (frozen EdgeVAE 16x8x8, in_ch=16); baselines="
                            + ("pixel 3x32x32 (raw images, in_ch=3)"
                               if baseline_repr == "pixel"
                               else "latent (frozen EdgeVAE 16x8x8, in_ch=16)"),
                    "backbones": BB_ORDER,
                    "seeds": seeds, "epochs": epochs, "lr": 0.001,
                    "lr_schedule": "cosine", "probe_epochs": probe_epochs,
                    "T_dc": T_dc, "source_pool": "real_subset (与蒸馏同池)",
                    "methods": ORDER,
                    "note": "curves = mean over seeds; reported acc = last epoch",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
           "results": results}
    jp = os.path.join(RESULT_DIR,
                      f"method_x_backbone_{config.DATASET}_ipc{ipc}_{baseline_repr}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {jp}")
    return out, jp


# ============================================================
# 出图: 三个单面板 (train loss / test loss / acc), 统一风格
# ============================================================

def _style_axes(ax, key, res):
    """单面板的统一坐标轴/网格样式 (acc 用比例留白; loss 下界最多到 -0.2 且不出刻度)"""
    ax.set_xlabel("Epoch", fontsize=9.5)
    ax.set_ylabel("Test accuracy (%)" if key == "acc_curve" else "Loss", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    if key == "acc_curve":
        ax.margins(y=0.28)
    else:
        lo = min(min(res[m][key]) for m in ORDER)
        hi = max(max(res[m][key]) for m in ORDER)
        ax.set_ylim(min(lo - 0.2, 0.0), hi + 0.28 * (hi - lo))


def _draw_curves(ax, res, key, with_legend, ncol=2, label_with_acc=True):
    for m in ORDER:
        color, mk, ls = STYLE[m]
        c = res[m][key]
        label = (f"{LABELS[m]} ({res[m]['acc']:.2f}%)" if label_with_acc
                 else LABELS[m])
        ax.plot(range(1, len(c) + 1), c, color=color, ls=ls, lw=1.5,
                marker=mk, markevery=5, markersize=5.5, label=label)
    if with_legend:
        ax.legend(ncol=ncol, frameon=False, fontsize=7.5, loc="upper right",
                  handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
                  labelspacing=0.35)


def plot_all(data, stem=f"method_compare"):
    res = data["results"]
    panels = [("train_curve", f"{stem}_loss_train"),
              ("test_curve", f"{stem}_loss_test"),
              ("acc_curve", f"{stem}_acc")]
    os.makedirs(FIG_DIR, exist_ok=True)
    for key, name in panels:
        fig, ax = plt.subplots(figsize=(5.0, 3.75))
        _draw_curves(ax, res, key, with_legend=True)
        _style_axes(ax, key, res)
        if key == "acc_curve":
            fig.tight_layout()          # 准确率跨度大, 方格无法用好看刻度实现, 保持默认
        else:
            lo = min(min(res[m][key]) for m in ORDER)
            hi = max(max(res[m][key]) for m in ORDER)
            bottom, top_min = curve_bounds(lo, hi, "loss")
            ax.set_xlim(0, len(res[ORDER[0]][key]))   # x 刻度固定 0, 5, 10, ...
            fig.tight_layout()
            g = square_grid_fixed_x(ax, x_step=5.0, bottom=bottom, top_min=top_min)
            if g:
                print(f"  [{name}] x_step=5, y_step={g[0]}, 格宽/格高={g[1]:.3f}")
        save(fig, name)


def plot_composite(data10, data100, stem="method_compare_2x2"):
    """2×2 合并图: 上排 = 两个数据集的 acc, 下排 = 两个数据集的 train loss

    共用一份图例 (4 个方法, 放图外下方); 整列正上方各标 (a) CIFAR-10 / (b) CIFAR-100。
    """
    pairs = [("CIFAR-10", data10), ("CIFAR-100", data100)]
    tags = ["a", "b"]                  # (a) = CIFAR-10, (b) = CIFAR-100
    rows = [("acc_curve", "Accuracy (%)"), ("train_curve", "Loss")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True)
    for r, (key, ylabel) in enumerate(rows):
        for c_, (ds, data) in enumerate(pairs):
            ax = axes[r][c_]
            res = data["results"]
            _draw_curves(ax, res, key, with_legend=False, label_with_acc=False)
            _style_axes(ax, key, res)
            ax.set_ylabel(ylabel if c_ == 0 else "", fontsize=9.5)
            if r == 0:
                ax.set_xlabel("")          # sharex 下 x 轴标签只在下排出现
    # 列标题: (a) CIFAR-10, (b) CIFAR-100, 整列正上方各一个 (只在上排两格上画)
    for c_ in range(len(pairs)):
        axes[0][c_].annotate(f"({tags[c_]}) {pairs[c_][0]}", xy=(0.5, 1.03),
                             xycoords="axes fraction", ha="center", va="bottom",
                             fontsize=10.5, color="0.15")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               handlelength=1.8, handletextpad=0.5, columnspacing=1.8,
               labelspacing=0.4)
    fig.tight_layout()
    for r, (key, _) in enumerate(rows):
        for c_, (ds, data) in enumerate(pairs):
            ax = axes[r][c_]
            if key == "acc_curve":
                continue                # 准确率跨度大, 方格无法用好看刻度实现, 保持默认
            res = data["results"]
            lo = min(min(res[m][key]) for m in ORDER)
            hi = max(max(res[m][key]) for m in ORDER)
            bottom, top_min = curve_bounds(lo, hi, "loss")
            ax.set_xlim(0, len(res[ORDER[0]][key]))   # x 刻度固定 0, 5, 10, ...
            g = square_grid_fixed_x(ax, x_step=5.0, bottom=bottom, top_min=top_min)
            if g:
                print(f"  [{stem} {ds} {key}] x_step=5, y_step={g[0]}, "
                      f"格宽/格高={g[1]:.3f}")
    os.makedirs(FIG_DIR, exist_ok=True)
    save(fig, stem)


def plot_method_x_backbone(data, stem="method_x_backbone"):
    """2×4 网格: 上排 acc, 下排 train loss; 列 = 4 个模型, 每格 4 条方法线

    共用一份图例 (4 个方法) 放图外下方; 列标题 (a)~(d) 标在整列正上方。
    """
    res_all = data["results"]
    rows = [("acc_curve", "Accuracy (%)"), ("train_curve", "Loss")]
    fig, axes = plt.subplots(2, len(BB_ORDER), figsize=(13.0, 6.0), sharex=True)
    for r, (key, ylabel) in enumerate(rows):
        for c_, bb in enumerate(BB_ORDER):
            ax = axes[r][c_]
            res = res_all[bb]
            _draw_curves(ax, res, key, with_legend=False, label_with_acc=False)
            _style_axes(ax, key, res)
            # 放大字体 (覆盖 _style_axes 的默认字号)
            ax.set_ylabel(ylabel if c_ == 0 else "", fontsize=13)
            ax.tick_params(labelsize=12)
            if r == 0:
                ax.set_xlabel("")          # sharex 下 x 轴标签只在下排出现
            else:
                ax.set_xlabel("Epoch", fontsize=13)
    for c_, bb in enumerate(BB_ORDER):
        axes[0][c_].annotate(f"({BB_TAGS[c_]}) {BB_NAMES[bb]}", xy=(0.5, 1.03),
                             xycoords="axes fraction", ha="center", va="bottom",
                             fontsize=14, color="0.15")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, fontsize=13.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.065),
               handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
               labelspacing=0.4)
    fig.tight_layout()
    for r, (key, _) in enumerate(rows):
        if key != "train_curve":
            continue                # 准确率跨度大, 方格无法用好看刻度实现, 保持默认
        for c_, bb in enumerate(BB_ORDER):
            ax = axes[r][c_]
            res = res_all[bb]
            lo = min(min(res[m][key]) for m in ORDER)
            hi = max(max(res[m][key]) for m in ORDER)
            bottom, top_min = curve_bounds(lo, hi, "loss")
            ax.set_xlim(0, len(res[ORDER[0]][key]))   # x 刻度固定 0, 10, ...
            g = square_grid_fixed_x(ax, x_step=10.0, bottom=bottom, top_min=top_min)
            if g:
                print(f"  [{stem} {bb} {key}] x_step=10, y_step={g[0]}, "
                      f"格宽/格高={g[1]:.3f}")
    os.makedirs(FIG_DIR, exist_ok=True)
    save(fig, stem)


def main(args):
    if args.multi_plot_only:
        with open(args.multi_plot_only) as f:
            data = json.load(f)
        m = data["meta"]
        plot_method_x_backbone(
            data, stem=f"method_x_backbone_{m['dataset']}_ipc{m['ipc']}_"
                       f"{m.get('baseline_repr', 'pixel')}")
        return
    if args.multi_backbone:
        data, _ = run_multi_backbone(
            ipc=args.ipc, seeds=args.seeds, epochs=args.epochs,
            probe_epochs=args.probe_epochs, T_dc=args.T_dc,
            baseline_repr=args.baseline_repr)
        m = data["meta"]
        plot_method_x_backbone(
            data, stem=f"method_x_backbone_{m['dataset']}_ipc{m['ipc']}_"
                       f"{m['baseline_repr']}")
        return
    if args.composite:
        with open(args.composite[0]) as f:
            d10 = json.load(f)
        with open(args.composite[1]) as f:
            d100 = json.load(f)
        plot_composite(d10, d100)
        return
    if args.plot_only:
        with open(args.plot_only) as f:
            data = json.load(f)
        m = data["meta"]
        tag = m.get("baseline_repr", "latent")
        plot_all(data, stem=f"method_compare_{m['dataset']}_ipc{m['ipc']}_{tag}")
        return
    data, _ = run(ipc=args.ipc, seeds=args.seeds, epochs=args.epochs,
                  probe_epochs=args.probe_epochs, T_dc=args.T_dc,
                  baseline_repr=args.baseline_repr)
    m = data["meta"]
    plot_all(data, stem=f"method_compare_{m['dataset']}_ipc{m['ipc']}_{m['baseline_repr']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="方法对比: Ours vs 梯度匹配 vs EL2N vs Forgetting (ResNet)")
    p.add_argument("--ipc", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--probe_epochs", type=int, default=8)
    p.add_argument("--T_dc", type=int, default=1000, help="DC 梯度精炼迭代数 (缺缓存时就地精炼)")
    p.add_argument("--baseline_repr", choices=["pixel", "latent"], default="pixel",
                   help="基线实现空间: pixel=在原图上实现并接像素分类器(默认); latent=在冻结 VAE 潜空间")
    p.add_argument("--plot_only", type=str, default=None, help="跳过训练, 直接读已有 json 出图")
    p.add_argument("--multi_backbone", action="store_true",
                   help="跑 4 方法 × 4 backbone (VGG/ResNet/MobileNet/ViT) 网格并出 2x4 图")
    p.add_argument("--multi_plot_only", type=str, default=None,
                   help="跳过训练, 读 method_x_backbone json 出 2x4 网格图")
    p.add_argument("--composite", type=str, nargs=2, default=None,
                   help="把两个数据集的 json 拼成 2x2: 上排 acc, 下排 test loss")
    a = p.parse_args()
    main(a)
