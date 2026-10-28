"""
实验: 一次性 K-means 质心蒸馏 vs DC 风格梯度精炼  @ IPC=100 (CIFAR-10)

产出论文两表:
  tab:refinement_accuracy   — 4 backbone 的 Centroid / Gradient Refined / Gain (pt)
  tab:distillation_overhead — K-means 与梯度精炼的 迭代数 / 时间 / FLOPs

口径:
  - Centroid 与 Gradient Refined 两列共用同一套下游协议 (同 backbone / epochs / lr / seeds)。
  - overhead 只计"蒸馏过程本身": 排除全池 VAE 编码与下游分类器训练。
  - K-means FLOPs 用实测 Lloyd 迭代数 (sklearn n_iter_) 解析估算;
    梯度精炼 FLOPs 含合成潜变量优化时的前向 + 反向。
  - 梯度精炼为 DC (Dataset Condensation, Zhao et al. 2021) 风格:
    以 K-means 质心为初值, 在冻结 EdgeVAE 的潜空间上用代理网络的梯度匹配精炼合成潜变量。

运行 (默认 = 论文主配置):
  python exp_refine.py --ipc 100 --subset 1000 --epochs 50 --seeds 3 --T_dc 1000
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans

import config
from backbones import (BACKBONES, ResNetBackbone, count_flops,
                       train_eval_backbone)
from config import get_cifar10_loaders
from analysis import fmt_flops
from utils import encode_test_latents
from vae import load_edge_vae


# ============================================================
# 1. 真实子集 → 潜变量 (全池编码; 不计入 overhead)
# ============================================================

def encode_real_subset(ae, device, ipc):
    """读 pt/real_subset_ipc{ipc}.pt 并编码为确定性潜变量 (mu)"""
    path = config.find_pt(f"real_subset_ipc{ipc}.pt")
    if path is None:
        raise FileNotFoundError(
            f"未找到真实子集 pt/real_subset_ipc{ipc}.pt, 请先运行: "
            f"python distill.py --ipc {ipc} --subset 1000")
    d = torch.load(path, map_location="cpu", weights_only=True)
    imgs, labels = d["images"], d["labels"]
    latents = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), 256):
            latents.append(ae.encode_mu(imgs[i:i + 256].to(device)).cpu())
    return torch.cat(latents, 0), labels, path


# ============================================================
# 2. 一次性 K-means 质心 (Centroid 基线) + 开销统计
# ============================================================

def kmeans_centroids(z_real, labels, ipc, C, n_init="auto"):
    """逐类 KMeans(n_clusters=ipc) 取簇均值中心

    返回 centers [C, ipc, C_in, H, W] 与开销统计:
      mean_iters   : 各类 Lloyd 迭代数 (n_iter_) 的均值 = Ī
      time_s       : K-means 阶段墙钟时间
      flops        : 解析估算 (见下)
    解析口径: 单次 Lloyd 迭代 ≈ n·k 个距离计算 (每个 2D FLOPs: D 乘 + D 加)
              + n·D (簇均值更新), D = C_in·H·W。
    """
    shape = z_real.shape[1:]                       # [C_in, H, W]
    D = int(np.prod(shape))
    centers = torch.zeros(C, ipc, *shape)
    iters, flops = [], 0
    t0 = time.perf_counter()
    for c in range(C):
        zf = z_real[labels == c].reshape(-1, D).numpy()
        n = zf.shape[0]
        km = KMeans(n_clusters=ipc, random_state=42, n_init=n_init)
        km.fit(zf)
        iters.append(int(km.n_iter_))
        centers[c] = torch.from_numpy(km.cluster_centers_).reshape(ipc, *shape)
        flops += int(km.n_iter_) * (2 * n * ipc * D + n * D)
    elapsed = time.perf_counter() - t0
    return centers, {"mean_iters": float(np.mean(iters)),
                     "iters_per_class": iters,
                     "time_s": elapsed, "flops": float(flops),
                     "n_init": n_init, "D": D}


# ============================================================
# 3. DC 风格梯度精炼 (Gradient Refined) + 开销统计
# ============================================================

def gradient_refine(syn_init, syn_labels, z_real, real_labels, C, device,
                    T_dc=1000, real_bs=256, syn_lr=1e-2, proxy_lr=0.01,
                    expert_steps=1, seed=0, log_every=200, syn_bs=None,
                    proxy_cls=ResNetBackbone):
    """DC 风格梯度匹配精炼合成潜变量

    每迭代:
      1) 采样 real 批, 算 ∇θ L_real (不建图)
      2) 合成点前向 (syn_bs 时只取随机子集, 省显存), 算 ∇θ L_syn (create_graph=True, 二阶)
      3) 匹配损失 MSE(∇θ L_syn, ∇θ L_real) → 反传到合成潜变量 S
      4) expert step: 用合成数据 SGD 更新代理网络 θ

    syn_bs: None = 每步用全部合成点 (默认); 整数 = 每步随机取 syn_bs 个点做二阶匹配,
            用于合成点数多 (如 5000) 且代理网络较大时避免显存溢出 (无偏的 SGD 估计)。

    返回 refined [C*ipc, C_in, H, W] 与开销统计 (T_dc / time_s / flops)。
    """
    torch.manual_seed(seed)
    S = syn_init.reshape(syn_init.size(0) * syn_init.size(1), *syn_init.shape[2:])
    S = S.clone().to(device).requires_grad_(True)
    syn_labels = syn_labels.to(device)
    z_real = z_real.to(device)
    real_labels = real_labels.to(device)
    N_real, N_syn = z_real.size(0), S.size(0)

    proxy = proxy_cls(in_ch=S.size(1), num_classes=C).to(device)
    # eval 模式: 冻结 BN 的批统计量 (用初始 running stats) 且关闭 dropout,
    # 避免 BN 批耦合把整批合成点一起推, 从而获得更干净的逐点匹配梯度。
    proxy.eval()
    params = [p for p in proxy.parameters() if p.requires_grad]
    opt_syn = optim.Adam([S], lr=syn_lr)
    opt_proxy = optim.SGD(params, lr=proxy_lr, momentum=0.9)

    t0 = time.perf_counter()
    for it in range(T_dc):
        # (1) 真实数据梯度 (不建图, 反传到 S 无意义)
        idx = torch.randperm(N_real, device=device)[:real_bs]
        loss_real = F.cross_entropy(proxy(z_real[idx]), real_labels[idx])
        g_real = torch.cat([g.reshape(-1) for g in
                            torch.autograd.grad(loss_real, params)]).detach()
        # (2) 合成数据梯度 (需二阶, 才能继续反传到 S); syn_bs 时只取随机子集
        if syn_bs is None or syn_bs >= N_syn:
            sub = S
            sub_lab = syn_labels
        else:
            sub_i = torch.randperm(N_syn, device=device)[:syn_bs]
            sub, sub_lab = S[sub_i], syn_labels[sub_i]
        loss_syn = F.cross_entropy(proxy(sub), sub_lab)
        g_syn = torch.cat([g.reshape(-1) for g in
                           torch.autograd.grad(loss_syn, params, create_graph=True)])
        # (3) 梯度匹配 → 更新合成潜变量
        match = F.mse_loss(g_syn, g_real)
        s_grad = torch.autograd.grad(match, S)[0]
        opt_syn.zero_grad()
        S.grad = s_grad
        opt_syn.step()
        # (4) expert step: 合成数据更新代理网络
        for _ in range(expert_steps):
            opt_proxy.zero_grad()
            F.cross_entropy(proxy(S.detach()), syn_labels).backward()
            opt_proxy.step()
        if log_every and (it + 1) % log_every == 0:
            print(f"    [refine] {it + 1}/{T_dc} match={match.item():.4e}")
    elapsed = time.perf_counter() - t0

    # 解析 FLOPs / 迭代: 3×fwd×(real_bs + n_syn) [前向+反向] + 2×fwd×n_syn [匹配反传到 S]
    n_syn = N_syn if (syn_bs is None or syn_bs >= N_syn) else syn_bs
    fwd = count_flops(proxy, torch.randn(1, S.size(1), S.size(2), S.size(3),
                                         device=device))
    per_iter = 3 * fwd * (real_bs + n_syn) + 2 * fwd * n_syn
    return S.detach().cpu(), {"T_dc": T_dc, "time_s": elapsed,
                              "flops": float(per_iter * T_dc),
                              "fwd_flops_per_sample": int(fwd),
                              "real_bs": real_bs, "syn_bs": syn_bs, "syn_lr": syn_lr,
                              "proxy_lr": proxy_lr, "expert_steps": expert_steps,
                              "proxy": proxy_cls.__name__, "proxy_mode": "eval"}


# ============================================================
# 4. 下游评估 (两列共用同一协议)
# ============================================================

def evaluate_all(centers, refined, ipc, C, test_loader, ae, device,
                 seeds, epochs, lr, lr_schedule, test_latents):
    H, W = centers.shape[-2], centers.shape[-1]
    out = {}
    for tag, syn in (("centroid", centers), ("refined", refined)):
        z_syn = syn.reshape(C * ipc, -1, H, W)
        labels = torch.arange(C).repeat_interleave(ipc)
        out[tag] = {}
        for name, cls in BACKBONES.items():
            accs, curve, tcurve = [], None, None
            for s in range(seeds):
                if s == 0:
                    a, curve, tcurve = train_eval_backbone(
                        cls, z_syn, labels, test_loader, ae, device, seed=s,
                        epochs=epochs, lr=lr, return_curve=True,
                        test_latents=test_latents, lr_schedule=lr_schedule)
                else:
                    a = train_eval_backbone(
                        cls, z_syn, labels, test_loader, ae, device, seed=s,
                        epochs=epochs, lr=lr, test_latents=test_latents,
                        lr_schedule=lr_schedule)
                accs.append(a)
                print(f"  [{tag:<8}] {name:<10} seed{s}: {a:.2f}%")
            out[tag][name] = {"acc": float(np.mean(accs)),
                              "std": float(np.std(accs)),
                              "seeds_acc": [float(x) for x in accs],
                              "train_curve": curve, "test_curve": tcurve}
    return out


# ============================================================
# 5. 输出两张表 (控制台 + LaTeX)
# ============================================================

def print_tables(res, ipc):
    disp = {"vgg": "VGG", "resnet": "ResNet", "mobilenet": "MobileNet",
            "vit": "ViT"}
    print("\n" + "=" * 78)
    print(f"表 1  tab:refinement_accuracy  (IPC={ipc}, CIFAR-10)")
    print("=" * 78)
    print(f"{'Backbone':<12}{'Centroid (%)':>16}{'Gradient Refined (%)':>24}{'Gain (pt)':>12}")
    print("-" * 78)
    for name in BACKBONES:
        c, r = res["centroid"][name], res["refined"][name]
        gain = r["acc"] - c["acc"]
        print(f"{disp[name]:<12}{c['acc']:>10.2f}±{c['std']:<4.2f}"
              f"{r['acc']:>18.2f}±{r['std']:<4.2f}{gain:>+12.2f}")
    print("-" * 78)
    print("  (Gain = Gradient Refined − Centroid, 同一 seed 配对; 表中为均值差)")
    print("=" * 78)

    print("\nLaTeX (tab:refinement_accuracy):")
    print("-" * 78)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Accuracy comparison between one-shot K-means centroid "
          r"distillation and")
    print(r"gradient-based refinement at IPC$=" + str(ipc) +
          r"$ on CIFAR-10.}")
    print(r"\label{tab:refinement_accuracy}")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Backbone & Centroid (\%) & Gradient Refined (\%) & Gain (pt) \\")
    print(r"\hline")
    for name in BACKBONES:
        c, r = res["centroid"][name], res["refined"][name]
        gain = r["acc"] - c["acc"]
        print(f"{disp[name]} & ${c['acc']:.2f}\\pm{c['std']:.2f}$ & "
              f"${r['acc']:.2f}\\pm{r['std']:.2f}$ & ${gain:+.2f}$ \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("-" * 78)


def fmt_flops_tex(x):
    """LaTeX 科学计数: 1.87e+10 → $1.87\\times10^{10}$"""
    exp = int(np.floor(np.log10(x)))
    mant = x / (10.0 ** exp)
    return f"${mant:.2f}\\times10^{{{exp}}}$"


def print_overhead(km_stat, dc_stat, ipc):
    print("\n" + "=" * 78)
    print(f"表 2  tab:distillation_overhead  (IPC={ipc})")
    print("=" * 78)
    km_it = "I=" + f"{km_stat['mean_iters']:.1f}"
    dc_it = "T=" + str(dc_stat["T_dc"])
    print(f"{'Method':<22}{'Iterations':>16}{'Time (s)':>12}{'Distill. FLOPs':>20}")
    print("-" * 78)
    print(f"{'K-means centroid':<22}{km_it:>16}"
          f"{km_stat['time_s']:>12.2f}{fmt_flops(km_stat['flops']):>20}")
    print(f"{'Gradient refinement':<22}{dc_it:>16}"
          f"{dc_stat['time_s']:>12.2f}{fmt_flops(dc_stat['flops']):>20}")
    print("-" * 78)
    print(f"  K-means 逐类迭代数: {km_stat['iters_per_class']}")
    print(f"  梯度精炼代理网络: {dc_stat['proxy']}, real_bs={dc_stat['real_bs']}, "
          f"syn_lr={dc_stat['syn_lr']}, proxy_lr={dc_stat['proxy_lr']}, "
          f"expert_steps={dc_stat['expert_steps']}")
    print("=" * 78)

    print("\nLaTeX (tab:distillation_overhead):")
    print("-" * 78)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Distillation overhead at IPC$=" + str(ipc) + r"$. The reported "
          r"cost excludes the common")
    print(r"full-pool encoding stage and downstream classifier training. K-means "
          r"FLOPs")
    print(r"are analytically estimated using the observed number of Lloyd "
          r"iterations, while")
    print(r"gradient-refinement FLOPs include the forward and backward operations")
    print(r"performed during synthetic-latent optimization.}")
    print(r"\label{tab:distillation_overhead}")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Method & Iterations & Time (s) & Distill. FLOPs \\")
    print(r"\hline")
    print(f"K-means centroid & $\\bar{{I}}={km_stat['mean_iters']:.1f}$ & "
          f"${km_stat['time_s']:.2f}$ & {fmt_flops_tex(km_stat['flops'])} \\\\")
    print(f"Gradient refinement & $T_{{\\mathrm{{DC}}}}={dc_stat['T_dc']}$ & "
          f"${dc_stat['time_s']:.2f}$ & {fmt_flops_tex(dc_stat['flops'])} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("-" * 78)


# ============================================================
# 主流程
# ============================================================

def main(ipc=100, subset=1000, seeds=3, epochs=50, T_dc=1000, real_bs=256,
         syn_lr=1e-2, proxy_lr=0.01, expert_steps=1, n_init="auto",
         device_str=None):
    device = torch.device(
        device_str or (config.DEVICE if torch.cuda.is_available() else "cpu"))
    C = config.NUM_CLASSES
    lr, lr_schedule = 0.001, "cosine"
    print(f"设备: {device} | IPC={ipc} subset={subset} epochs={epochs} "
          f"seeds={seeds} T_dc={T_dc}")

    ae = load_edge_vae(device)
    _, _, test_loader = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS)
    test_latents = encode_test_latents(test_loader, ae, device)

    print("\n[1/4] 编码真实子集 (全池编码, 不计入 overhead)...")
    z_real, labels_real, subset_path = encode_real_subset(ae, device, ipc)
    print(f"      真实潜变量: {tuple(z_real.shape)} | {os.path.basename(subset_path)}")

    print("\n[2/4] 一次性 K-means 质心 (Centroid)...")
    centers, km_stat = kmeans_centroids(z_real, labels_real, ipc, C, n_init=n_init)
    print(f"      I={km_stat['mean_iters']:.1f} | time={km_stat['time_s']:.2f}s | "
          f"FLOPs={fmt_flops(km_stat['flops'])}")

    print(f"\n[3/4] DC 风格梯度精炼 (Gradient Refined, T_dc={T_dc})...")
    syn_labels = torch.arange(C).repeat_interleave(ipc)
    refined, dc_stat = gradient_refine(
        centers, syn_labels, z_real, labels_real, C, device, T_dc=T_dc,
        real_bs=real_bs, syn_lr=syn_lr, proxy_lr=proxy_lr,
        expert_steps=expert_steps)
    print(f"      T_dc={dc_stat['T_dc']} | time={dc_stat['time_s']:.2f}s | "
          f"FLOPs={fmt_flops(dc_stat['flops'])}")

    # 保存合成潜变量
    torch.save({"syn_latents": centers, "ipc": ipc, "center_mode": "kmeans",
                "method": "kmeans_centroid"},
               os.path.join(config.PT_DIR, f"syn_latents_ipc{ipc}_centroid_edge.pt"))
    torch.save({"syn_latents": refined, "ipc": ipc, "method": "dc_gradient_refine"},
               os.path.join(config.PT_DIR, f"syn_latents_ipc{ipc}_refined_edge.pt"))

    print(f"\n[4/4] 下游评估 (4 backbone × {seeds} seeds × {epochs} epochs)...")
    res = evaluate_all(centers, refined, ipc, C, test_loader, ae, device,
                       seeds, epochs, lr, lr_schedule, test_latents)

    print_tables(res, ipc)
    print_overhead(km_stat, dc_stat, ipc)

    save_path = os.path.join(config.RESULT_DIR,
                             f"refine_vs_centroid_ipc{ipc}_d{ae.latent_dim}_edge.json")
    with open(save_path, "w") as f:
        json.dump({
            "meta": {
                "exp": "one-shot K-means centroid vs DC gradient refinement",
                "ipc": ipc, "subset": subset, "seeds": seeds, "epochs": epochs,
                "lr": lr, "lr_schedule": lr_schedule,
                "latent_dim": ae.latent_dim, "spatial": ae.spatial_size,
                "points": C * ipc, "device": str(device),
                "n_init": km_stat["n_init"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "accuracy": {tag: {k: {"acc": v["acc"], "std": v["std"],
                                   "seeds_acc": v["seeds_acc"]}
                               for k, v in res[tag].items()}
                         for tag in res},
            "overhead": {"kmeans_centroid": km_stat,
                         "gradient_refinement": dc_stat},
            "curves": {tag: {k: {"train": v["train_curve"],
                                 "test": v["test_curve"]}
                             for k, v in res[tag].items()}
                       for tag in res},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n数据已保存: {save_path}")
    return res, km_stat, dc_stat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一次性 K-means 质心 vs DC 梯度精炼 (IPC=100)")
    parser.add_argument("--ipc", type=int, default=config.IPC)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--T_dc", type=int, default=1000, help="梯度精炼迭代数")
    parser.add_argument("--real_bs", type=int, default=256, help="每迭代采样真实点数")
    parser.add_argument("--syn_lr", type=float, default=1e-2)
    parser.add_argument("--proxy_lr", type=float, default=0.01)
    parser.add_argument("--expert_steps", type=int, default=1)
    parser.add_argument("--n_init", type=str, default="auto",
                        help="sklearn KMeans n_init ('auto' 或整数)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    n_init = args.n_init if args.n_init == "auto" else int(args.n_init)
    main(ipc=args.ipc, subset=args.subset, seeds=args.seeds, epochs=args.epochs,
         T_dc=args.T_dc, real_bs=args.real_bs, syn_lr=args.syn_lr,
         proxy_lr=args.proxy_lr, expert_steps=args.expert_steps,
         n_init=n_init, device_str=args.device)
