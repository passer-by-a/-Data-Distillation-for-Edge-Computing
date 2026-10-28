"""
4 种蒸馏方法的【全流程能耗】对比 (含 VAE 预训练与推理, ResNet 下游, CIFAR-10, IPC=100)

口径 (与 exp_method_compare.py 的训练配置逐项对齐):
  每种方法从"产出合成数据"到"下游训练 + 推理"的总能耗 =
      VAE 预训练(一次性, 仅 Ours) + 编码器部署(一次性, 仅 Ours)
      + 产出阶段 + 传输 + 50 epoch 下游训练 + 测试集一次推理。
    Ours (VAE+K-means) : VAE 预训练 (N_vae×ep×3×F_vae) + 全池编码 (N×F_enc) + 潜空间逐类 K-means
                         下游在潜空间用 ResNetBackbone (in_ch=16, ~28K)
    Gradient matching  : 像素 K-means 初值 + DC 梯度精炼 (T_dc) [产出]
    EL2N / Forgetting  : 像素池上训练代理模型打分 (train_probe, n_runs×probe_epochs) [产出]
                         两法共用同一次代理训练, 故产出阶段能耗相同
    三基线下游: 与 Ours 同一模型族 ResNetBackbone (in_ch=3, ~28K), 仅输入通道不同;
               DC 精炼 / 代理打分网络仍为像素空间 PixelResNet (in_ch=3, ~0.7M, 与 exp_method_compare 一致)

能耗模型 (与 analysis.py / METRICS.md §6 一致):
    E = 总 FLOPs × ENERGY_PER_FLOP (1e-11 J/FLOP, 边缘加速器 ~10 pJ 量级)
    训练每样本 FLOPs ≈ 单样本前向 FLOPs × TRAIN_FWD_MULT(=3, 前向+反向)
    所有 FLOPs 均由 thop 实测 (前向) 或按显式解析式估算 (K-means / DC 精炼 / 代理训练)。

产出:
  figures/energy_by_method_{dataset}_ipc{ipc}.pdf (+ .png)      # 分阶段能耗柱状图 (4 段)
  figures/flops_by_method_{dataset}_ipc{ipc}.pdf (+ .png)       # 分阶段计算量条形图
  figures/transfer_latency_by_method_{dataset}_ipc{ipc}.pdf (+ .png)  # 传输时延柱状图
  results/energy_by_method_{dataset}_ipc{ipc}.json

运行:
  python exp_energy_methods.py                  # ipc=100, epochs=50
  python exp_energy_methods.py --ipc 50 --epochs 50
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from analysis import (ENERGY_PER_FLOP, ENERGY_PER_BIT, TRAIN_FWD_MULT,
                      BANDWIDTH_MBPS, bytes_of_image, fmt_bytes, LATENT_SPATIAL,
                      measure_encoder_flops, encode_real_subset)
from backbones import ResNetBackbone, PixelResNet, count_flops
from vae import load_edge_vae, EdgeVAE

C = config.NUM_CLASSES
FIG_DIR = "figures"
# VAE 预训练口径: 训练集 = CIFAR train(50000) - 2% 验证 = 49000
# (见 config.get_cifar10_loaders 默认 val_ratio=0.02); epochs = config.AE_EPOCHS
VAE_TRAIN_N = 49000
# 测试集规模 (CIFAR-10/100 均为 10000 张), 用于"推理"阶段能耗
N_TEST = 10000
ORDER = ["ours", "gradmatch", "eln2", "forgetting"]
LABELS = {"ours": "Ours (VAE+K-means)", "gradmatch": "Gradient matching (DC)",
          "eln2": "EL2N", "forgetting": "Forgetting"}
SHORT_LABELS = {"ours": "Ours", "gradmatch": "GradMatch (DC)",
                "eln2": "EL2N", "forgetting": "Forgetting"}
# 统一调色板 (柱状图)
PALETTE = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
# 阶段配色 (柱状图把 6 个阶段分开显示)
STAGE_COLORS = {"vae": PALETTE[0], "deploy": PALETTE[4], "production": PALETTE[1],
                "transfer": PALETTE[3], "downstream": PALETTE[2],
                "inference": PALETTE[5]}
STAGE_LABELS = {"vae": "VAE pretraining (Ours only)",
                "deploy": "Encoder deployment (Ours only)",
                "production": "Data production",
                "transfer": "Data transmission",
                "downstream": "Downstream training",
                "inference": "Inference"}
# 阶段纹理 (黑白打印区分; 与 STAGE_COLORS 一一对应)
HATCH = {"vae": "//", "deploy": "\\\\", "production": "xx",
         "transfer": "..", "downstream": "oo", "inference": "||"}
# 方法纹理 (传输时延图等按方法分组的柱)
METHOD_HATCH = {"raw": "\\\\", "ours": "//", "gradmatch": "xx",
                "eln2": "..", "forgetting": "oo"}


# ============================================================
# FLOPs 统计辅助
# ============================================================

def load_pool(ipc):
    """源池: pt/real_subset_ipc{ipc}.pt (每类 1000 张真实原图, 与蒸馏同池)"""
    p = config.find_pt(f"real_subset_ipc{ipc}.pt")
    if p is None:
        raise FileNotFoundError(
            f"未找到 pt/real_subset_ipc{ipc}.pt; 请先运行 "
            f"python distill.py --ipc {ipc} --subset 1000")
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["images"], d["labels"], p


def classwise_kmeans_iters(z_flat, labels_np, k, n_init="auto"):
    """逐类 KMeans(k) 取真实收敛迭代数 n_iter_ (不改动任何产物)"""
    iters = []
    for c in range(C):
        zf = z_flat[labels_np == c]
        km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        km.fit(zf)
        iters.append(int(km.n_iter_))
    return iters


def kmeans_distill_flops(iters, n_per_class, k, D):
    """论文口径 K-means 蒸馏 FLOPs: 3·Σ_c (Nc·Mc·D·Ic) (=1.5× 旧解析式 2nkD+nD)

    Ours(潜空间 D=1024) 与 DC 像素初值(D=3072) 用同一公式。
    """
    return float(3.0 * sum(n_per_class * k * D * ic for ic in iters))


# ============================================================
# 主流程
# ============================================================

def run(ipc=100, epochs=50, probe_epochs=8, n_runs=3, T_dc=1000,
        real_bs=256, device_str=None, vae_epochs=None, vae_train_n=VAE_TRAIN_N):
    if vae_epochs is None:
        vae_epochs = config.AE_EPOCHS
    device = torch.device(
        device_str or (config.DEVICE if torch.cuda.is_available() else "cpu"))
    N_pool_target = None
    print("=" * 84)
    print(f"4 方法全流程训练能耗 (含 VAE 预训练) | {config.DATASET} | "
          f"ipc={ipc} ({ipc * C} 点) | backbone=ResNet | epochs={epochs} | {device}")
    print("=" * 84)

    ae = load_edge_vae(device)
    pool, labels, pool_path = load_pool(ipc)
    N = pool.size(0)
    M = C * ipc
    labels_np = labels.numpy()
    n_per_class = int(N // C)
    print(f"源池: {os.path.basename(pool_path)} | N={N} (每类 {n_per_class}) | "
          f"M={M} (每类 {ipc})")

    # ---- 1. 单样本前向 FLOPs (thop 实测) ----
    print("\n[1/8] 实测单样本前向 FLOPs (thop)...")
    F_enc, enc_note = measure_encoder_flops(
        ae, device, config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    F_vae = count_flops(
        EdgeVAE(latent_dim=ae.latent_dim, num_classes=C).to(device),
        torch.zeros(1, config.IMAGE_CHANNELS, config.IMAGE_SIZE,
                    config.IMAGE_SIZE, device=device))
    F_lat = count_flops(ResNetBackbone(in_ch=ae.latent_dim, num_classes=C).to(device),
                        torch.zeros(1, ae.latent_dim, ae.spatial_size,
                                    ae.spatial_size, device=device))
    # 基线下游: 与 Ours 同一模型族 ResNetBackbone, 只是输入通道=3 (像素)
    F_lat3 = count_flops(ResNetBackbone(in_ch=3, num_classes=C).to(device),
                         torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                                     device=device))
    # DC 精炼 / 代理打分网络: 仍用像素空间 PixelResNet (与 exp_method_compare 一致)
    F_pix = count_flops(PixelResNet(in_ch=3, num_classes=C).to(device),
                        torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                                    device=device))
    print(f"      F_enc (image->mu)   : {F_enc:,} FLOPs/sample  [{enc_note[:24]}...]")
    print(f"      F_vae (EdgeVAE enc+dec+cls)  : {F_vae:,} FLOPs/sample")
    print(f"      F_lat (ResNetBackbone 16ch, ours 下游) : {F_lat:,} FLOPs/sample")
    print(f"      F_lat3(ResNetBackbone 3ch, 基线下游)   : {F_lat3:,} FLOPs/sample")
    print(f"      F_pix (PixelResNet 3ch, DC/代理网络)   : {F_pix:,} FLOPs/sample")

    # ---- 2. VAE 预训练 (一次性固定成本, 仅 Ours 需要) ----
    print(f"\n[2/8] VAE 预训练 (EdgeVAE, {vae_train_n} 图 × {vae_epochs} ep × "
          f"{TRAIN_FWD_MULT:.0f}× 前向+反向) ...")
    Phi_vae = float(vae_train_n * vae_epochs * TRAIN_FWD_MULT * F_vae)
    print(f"      Phi_vae_pretrain    : {Phi_vae:.4e} FLOPs "
          f"({vae_train_n}×{vae_epochs}×{TRAIN_FWD_MULT:.0f}×{F_vae:,})")

    # ---- 3. Ours 产出: 全池编码 + 潜空间 K-means ----
    print("\n[3/8] Ours 产出阶段: 全池编码 + 潜空间逐类 K-means ...")
    z_real, labels_lat = encode_real_subset(ae, device, ipc)   # [N,16,8,8]
    DZ = int(np.prod(z_real.shape[1:]))                        # 1024
    z_flat = z_real.reshape(N, -1).numpy()
    it_lat = classwise_kmeans_iters(z_flat, labels_lat.numpy(), ipc)
    Phi_enc = float(N * F_enc)
    Phi_km_lat = kmeans_distill_flops(it_lat, n_per_class, ipc, DZ)
    print(f"      编码 Phi_enc        : {Phi_enc:.4e} FLOPs")
    print(f"      潜空间 K-means iters: {it_lat} (avg {np.mean(it_lat):.1f})")
    print(f"      K-means Phi_km_lat  : {Phi_km_lat:.4e} FLOPs")

    # ---- 4. DC (pixel) 产出: 像素 K-means 初值 + 梯度精炼 ----
    print(f"\n[4/8] DC 产出阶段: 像素 K-means 初值 + 梯度精炼 (T_dc={T_dc}) ...")
    pix_flat = pool.reshape(N, -1).numpy()                     # [N,3072]
    it_pix = classwise_kmeans_iters(pix_flat, labels_np, ipc)
    Phi_km_pix = kmeans_distill_flops(it_pix, n_per_class, ipc, pix_flat.shape[1])
    n_syn = min(M, 1000)                                       # exp_method_compare: syn_bs
    per_iter = TRAIN_FWD_MULT * F_pix * (real_bs + n_syn) + 2 * F_pix * n_syn
    Phi_dc = float(per_iter * T_dc)
    print(f"      像素 K-means iters  : {it_pix} (avg {np.mean(it_pix):.1f})")
    print(f"      Phi_km_pix          : {Phi_km_pix:.4e} FLOPs")
    print(f"      精炼/迭代           : {per_iter:.4e} FLOPs "
          f"(real_bs={real_bs}, syn_bs={n_syn})")
    print(f"      Phi_dc_refine       : {Phi_dc:.4e} FLOPs")

    # ---- 5. EL2N / Forgetting 产出: 代理模型训练 ----
    print(f"\n[5/8] EL2N/Forgetting 产出: 代理模型训练 "
          f"(n_runs={n_runs} × {probe_epochs} ep) ...")
    per_sample_ep = (TRAIN_FWD_MULT + 1) * F_pix               # 训练(fwd+bwd) + 全池 eval 前向
    Phi_probe = float(n_runs * probe_epochs * N * per_sample_ep)
    print(f"      单 epoch/样本       : {per_sample_ep:,} FLOPs "
          f"({TRAIN_FWD_MULT:.0f}x 训练 + 1x eval)")
    print(f"      Phi_probe           : {Phi_probe:.4e} FLOPs")

    # ---- 6. 下游训练 (M 点 × epochs × 3×F) ----
    print(f"\n[6/8] 下游训练阶段 (M={M} × {epochs} ep) ...")
    down_lat = float(M * epochs * TRAIN_FWD_MULT * F_lat)
    down_pix = float(M * epochs * TRAIN_FWD_MULT * F_lat3)
    print(f"      潜空间 (ResNetBackbone 16ch, ours)   : {down_lat:.4e} FLOPs")
    print(f"      像素   (ResNetBackbone 3ch, 三基线)  : {down_pix:.4e} FLOPs")

    # ---- 组装各方法 FLOPs 与能耗 ----
    stages = {
        "ours":       {"vae_pretrain": Phi_vae, "production": Phi_enc + Phi_km_lat,
                       "downstream": down_lat},
        "gradmatch":  {"vae_pretrain": 0.0, "production": Phi_km_pix + Phi_dc,
                       "downstream": down_pix},
        "eln2":       {"vae_pretrain": 0.0, "production": Phi_probe,
                       "downstream": down_pix},
        "forgetting": {"vae_pretrain": 0.0, "production": Phi_probe,
                       "downstream": down_pix},
    }
    for m in ORDER:
        s = stages[m]
        s["total_flops"] = s["vae_pretrain"] + s["production"] + s["downstream"]
        for k in ("vae_pretrain", "production", "downstream", "total_flops"):
            s[k + "_energy_j"] = s[k] * ENERGY_PER_FLOP

    # ---- 传输时延: 把合成集上传到边缘设备 = 数据字节数 ÷ 带宽 (@ BANDWIDTH_MBPS) ----
    # 统一部署量化口径 (1 B/标量, 与 analysis.py efficiency 一致): Ours 潜变量 int8,
    # 三个基线产出/选取像素原图 uint8; 点数均为 M。
    # Raw 参照 = 压缩前的原图 = 蒸馏/选取所用源池 (每类 n_per_class 张, 共 N 张) 像素 uint8。
    raw_bytes = float(bytes_of_image(N))
    bytes_by_method = {
        "ours": float(M * ae.latent_dim * LATENT_SPATIAL * LATENT_SPATIAL),  # int8 潜变量
        "gradmatch": float(bytes_of_image(M)),
        "eln2": float(bytes_of_image(M)),
        "forgetting": float(bytes_of_image(M)),
    }
    for m in ORDER:
        b = bytes_by_method[m]
        stages[m]["transfer_bytes"] = b
        stages[m]["transfer_latency_s"] = b * 8 / (BANDWIDTH_MBPS * 1e6)
        # 传输能耗: 每个 bit 的无线传输能耗 × 位数 (与 analysis.py efficiency 同口径)
        stages[m]["transfer_energy_j"] = b * 8 * ENERGY_PER_BIT

    # ---- 编码器部署 (一次性): 把 Ours 的 VAE encoder 权重下发到边缘设备 ----
    # 只计"权重下发"本身; 边缘端逐图编码的算力已包含在 ours.production
    # (全池 N 张编码), 此处不再重复计算, 否则会翻倍。
    enc_params = int(sum(p.numel() for p in ae.encoder.parameters()))
    deploy_bytes = float(enc_params * 4)                     # fp32 权重
    deploy_latency = deploy_bytes * 8 / (BANDWIDTH_MBPS * 1e6)
    deploy_energy = deploy_bytes * 8 * ENERGY_PER_BIT
    print(f"\n[7/8] 编码器部署: encoder {enc_params:,} 参数 fp32 = "
          f"{fmt_bytes(deploy_bytes)} → {deploy_latency:.2f} s / "
          f"{fmt_energy(deploy_energy)} (一次性, 仅 Ours)")

    # ---- 推理 (测试集一次前向, 端到端) ----
    # 推理只计下游分类器前向: Ours 用潜空间分类器 F_lat, 三基线用像素分类器 F_lat3;
    # encoder 前向已计入 production (边缘端编码), 推理阶段不重复计。
    F_inf = {"ours": F_lat, "gradmatch": F_lat3,
             "eln2": F_lat3, "forgetting": F_lat3}
    print(f"[8/8] 推理: {N_TEST} 张测试图 × 单样本分类器前向 "
          f"(Ours=latent cls={F_inf['ours']:,}; 基线=pixel cls={F_lat3:,})")

    for m in ORDER:
        s = stages[m]
        s["deploy_bytes"] = deploy_bytes if m == "ours" else 0.0
        s["deploy_latency_s"] = deploy_latency if m == "ours" else 0.0
        s["deploy_energy_j"] = deploy_energy if m == "ours" else 0.0
        s["inference_flops"] = float(N_TEST * F_inf[m])
        s["inference_energy_j"] = s["inference_flops"] * ENERGY_PER_FLOP
        s["total_energy_j"] = (s["total_flops_energy_j"]
                               + s["transfer_energy_j"]
                               + s["deploy_energy_j"]
                               + s["inference_energy_j"])

    # ---- 打印汇总表 ----
    print("\n" + "=" * 126)
    print(f"全流程能耗 (E = FLOPs × {ENERGY_PER_FLOP:.0e} J/FLOP; "
          f"传输 E = bit × {ENERGY_PER_BIT:.0e} J/bit); "
          f"VAE 预训练/编码器部署为一次性成本, 仅 Ours 需要")
    print("=" * 126)
    cols = [("vae_pretrain_energy_j", "VAE pretrain"),
            ("deploy_energy_j", "Encoder deploy"),
            ("production_energy_j", "Production"),
            ("transfer_energy_j", "Transmission"),
            ("downstream_energy_j", "Downstream"),
            ("inference_energy_j", "Inference"),
            ("total_energy_j", "Total")]
    print(f"{'Method':<22}" + "".join(f"{h:>15}" for _, h in cols))
    print("-" * 126)
    for m in ORDER:
        s = stages[m]
        print(f"{LABELS[m]:<22}" + "".join(f"{fmt_energy(s[k]):>15}" for k, _ in cols))
    print("-" * 126)
    base = min(stages[m]["total_energy_j"] for m in ORDER)
    print(f"  (基准 = 最低能耗方法 = {fmt_energy(base)})")
    for m in ORDER:
        print(f"  {LABELS[m]:<24} = {stages[m]['total_energy_j'] / base:>8.1f}× min")
    print("=" * 126)

    # ---- 落盘 JSON ----
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    out = {
        "meta": {
            "exp": "full-pipeline energy by method (deploy + production + transfer + "
                   "downstream + inference, ResNet downstream)",
            "dataset": config.DATASET, "ipc": ipc, "points": M,
            "pool_size": N, "pool_per_class": n_per_class, "epochs": epochs,
            "probe_epochs": probe_epochs, "probe_runs": n_runs, "T_dc": T_dc,
            "dc_real_bs": real_bs, "dc_syn_bs": n_syn,
            "energy_per_flop_j": ENERGY_PER_FLOP, "train_fwd_mult": TRAIN_FWD_MULT,
            "energy_per_bit_j": ENERGY_PER_BIT,
            "bandwidth_mbps": BANDWIDTH_MBPS,
            "n_test": N_TEST,
            "transfer_latency_note": "合成集上传时延 = 字节数×8 ÷ 带宽; 统一部署量化口径 "
                                     "1 B/标量: Ours=潜变量int8, 基线=像素uint8; "
                                     "见 analysis.bytes_of_image",
            "vae_pretrain": {"train_size": vae_train_n, "epochs": vae_epochs,
                             "flops": Phi_vae,
                             "note": "EdgeVAE 全流程(enc+dec+cls)前向×3(fwd+bwd)×N×ep; "
                                     "一次性固定成本, 仅 Ours 计入"},
            "encoder_deploy": {"params": enc_params, "bytes": deploy_bytes,
                               "latency_s": deploy_latency, "energy_j": deploy_energy,
                               "note": "encoder 权重 fp32 一次性下发到边缘; "
                                       "边缘端逐图编码的算力已计入 production, 此处不重复计"},
            "inference": {"n_test": N_TEST,
                          "flops_per_sample": {"ours": F_inf["ours"],
                                               "baselines": F_lat3},
                          "note": "测试集一次分类器前向 (不含 encoder): "
                                  "Ours = 潜空间分类器, 基线 = 像素分类器; "
                                  "encoder 前向已计入 production 阶段, 不重复计"},
            "flops_per_sample": {"encoder_mu": F_enc, "edge_vae_full": F_vae,
                                 "resnet_latent": F_lat, "resnet_latent_3ch": F_lat3,
                                 "pixel_resnet": F_pix},
            "latent_kmeans_iters": it_lat, "pixel_kmeans_iters": it_pix,
            "net": "ours 下游=ResNetBackbone(16ch,~28K); 三基线下游=ResNetBackbone(3ch,~28K); "
                   "DC 精炼/代理打分网络=PixelResNet(3ch,~0.7M)",
            "note": "E = FLOPs × ENERGY_PER_FLOP; production = encode/distill/select; "
                    "downstream = 50-epoch classifier training on M points; "
                    "transmission = 合成集字节数×8 × ENERGY_PER_BIT; "
                    "deploy = encoder 权重下发(一次性); inference = 测试集分类器一次前向; "
                    "Ours 额外计入 VAE 预训练(一次性)",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "methods": {m: {k: stages[m][k] for k in (
            "vae_pretrain", "production", "downstream", "total_flops",
            "inference_flops",
            "vae_pretrain_energy_j", "production_energy_j",
            "downstream_energy_j", "total_flops_energy_j",
            "transfer_bytes", "transfer_latency_s",
            "transfer_energy_j", "deploy_bytes", "deploy_latency_s",
            "deploy_energy_j", "inference_energy_j", "total_energy_j")}
            for m in ORDER},
        "raw_reference": {"note": "压缩前的原图 = 蒸馏/选取所用源池 (每类 n_per_class 张, 共 N 张), 像素 uint8",
                          "n_images": N, "n_per_class": n_per_class, "transfer_bytes": raw_bytes,
                          "transfer_latency_s": raw_bytes * 8 / (BANDWIDTH_MBPS * 1e6)},
    }
    jp = os.path.join(config.RESULT_DIR,
                      f"energy_by_method_{config.DATASET}_ipc{ipc}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"数据已保存: {jp}")

    # ---- 摊薄 N 次复用: 每任务能耗 = (VAE 预训练 + 编码器部署)/N + 产出 + 传输 + 下游 + 推理 ----
    ours = stages["ours"]
    ours_once = once_energy(ours)
    ours_recur = recurring_energy(ours)
    print("\n" + "=" * 100)
    print("摊薄 N 次复用后, 每任务全流程能耗 "
          "(VAE 预训练 + 编码器部署按 1/N 摊销; 基线无一次性成本)")
    print("=" * 100)
    print(f"  {'N (tasks)':>10}" +
          "".join(f"{LABELS[m].split(' ')[0]:>16}" for m in ORDER))
    for nn in (1, 2, 5, 10, 20, 50, 100, 1000):
        cells = "".join(f"{fmt_energy(once_energy(stages[m]) / nn
                                      + recurring_energy(stages[m])):>16}"
                        for m in ORDER)
        print(f"  {nn:>10}" + cells)
    for m in ORDER:
        if m == "ours":
            continue
        d = recurring_energy(stages[m]) - ours_recur
        if d > 0:
            print(f"  交叉点: Ours 摊薄后低于 {LABELS[m]} 需 N ≥ {ours_once / d:.1f} 个任务")
    print("=" * 100)

    # ---- 传输时延汇总 ----
    print("\n" + "=" * 92)
    print(f"传输时延 (统一 1 B/标量; 合成集上传到边缘: 数据字节数×8 ÷ {BANDWIDTH_MBPS:.0f} Mbps)")
    print("=" * 92)
    print(f"  {f'Raw ({n_per_class}/class)':<24}{fmt_bytes(raw_bytes):>12}"
          f"  (pixel uint8)          {raw_bytes * 8 / (BANDWIDTH_MBPS * 1e6):>8.2f} s")
    for m in ORDER:
        s = stages[m]
        form = (f"latent int8 ({ae.latent_dim}x8x8)" if m == "ours"
                else "pixel uint8 (3x32x32)")
        print(f"  {LABELS[m]:<24}{fmt_bytes(s['transfer_bytes']):>12}"
              f"  ({form})  {s['transfer_latency_s']:>8.2f} s")
    print("=" * 92)

    plot_bars(stages, ipc, epochs)
    plot_flops(stages, ipc, epochs)
    plot_transfer_latency(stages, ipc, raw_bytes=raw_bytes)
    return out


# ============================================================
# 出图: 分组柱状图 (VAE预训练 / 产出 / 下游训练 分段显示), 对数 y 轴
# ============================================================

def fmt_energy(j):
    if j >= 1e3:
        return f"{j / 1e3:.2f} kJ"
    if j >= 1.0:
        return f"{j:.2f} J"
    return f"{j * 1e3:.2f} mJ"


def save_fig(fig, stem, msg="图已保存"):
    """统一保存 pdf+png 并关闭画布 (避免各绘图函数重复同一段代码)"""
    pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
    png = os.path.join(FIG_DIR, f"{stem}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{msg}: {pdf} / {png}")


def once_energy(s):
    """一次性成本 (可跨任务摊销): VAE 预训练 + 编码器部署 (基线均为 0)"""
    return s["vae_pretrain_energy_j"] + s["deploy_energy_j"]


def recurring_energy(s):
    """每任务固定成本: 产出 + 传输 + 下游训练 + 推理"""
    return (s["production_energy_j"] + s["transfer_energy_j"]
            + s["downstream_energy_j"] + s["inference_energy_j"])


def plot_bars(stages, ipc, epochs):
    """分组柱状图: 每个方法把 [VAE预训练 / 产出 / 下游 / 传输] 四段分开显示 (对数 y 轴)

    vae 仅 Ours 非零 (基线留空)。部署/推理不计入本图 (见 JSON 与摊薄图)。
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    x = np.arange(len(ORDER))
    w = 0.20
    ylo = 0.3                       # 对数轴下界 (用于柱底锚点)
    comps = [("vae", "vae_pretrain_energy_j"),
             ("production", "production_energy_j"),
             ("downstream", "downstream_energy_j"),
             ("transfer", "transfer_energy_j")]
    off = {"vae": -1.5 * w, "production": -0.5 * w,
           "downstream": 0.5 * w, "transfer": 1.5 * w}

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)

    for m, xi in zip(ORDER, x):
        s = stages[m]
        for key, ek in comps:
            v = s[ek]
            if v <= ylo:            # 基线无 VAE 预训练 (v=0), 跳过留空
                continue
            bar = ax.bar(xi + off[key], v - ylo, w, bottom=ylo,
                         color=STAGE_COLORS[key], edgecolor="white",
                         linewidth=0.4, hatch=HATCH[key])[0]
            ax.annotate(fmt_energy(v), (bar.get_x() + bar.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x, ["Ours", "GradMatch", "EL2N", "Forgetting"], fontsize=16)
    ax.set_ylabel("Energy consumption (J)", fontsize=18)
    ax.tick_params(axis="y", labelsize=17)
    ymax = max(stages[m]["total_energy_j"] for m in ORDER)
    ax.set_ylim(ylo, ymax * 5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[k], hatch=HATCH[k])
               for k, _ in comps]
    ax.legend(handles, ["VAE pretraining", "Data production",
                        "Downstream training", "Data transmission"],
              fontsize=15.5, loc="upper right", framealpha=0.9,
              handlelength=2.0, handleheight=1.5, labelspacing=0.5,
              borderpad=0.6, borderaxespad=0.6)
    fig.tight_layout()

    save_fig(fig, f"energy_by_method_{config.DATASET}_ipc{ipc}")


# ============================================================
# 出图: 计算量 (FLOPs) 分组条形图, 对数 x 轴
# ============================================================

def fmt_flops(f):
    if f >= 1e15:
        return f"{f / 1e15:.2f} P"
    if f >= 1e12:
        return f"{f / 1e12:.2f} T"
    if f >= 1e9:
        return f"{f / 1e9:.2f} G"
    return f"{f / 1e6:.2f} M"


def plot_flops(stages, ipc, epochs):
    """计算量分组条形图: [VAE预训练 / 产出 / 下游训练 / 推理] 四段 (对数 x 轴)

    传输为通信开销 (bit), 不产生 FLOPs, 故不计入本图。
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    comps = [("vae", "vae_pretrain"), ("production", "production"),
             ("downstream", "downstream"), ("inference", "inference_flops")]
    h = 0.20
    xlo = 1e9                        # 对数轴下界 (1 GFLOPs, 覆盖 Ours 分类器推理)
    y = np.arange(len(ORDER))
    off = {k: (i - 1.5) * h for i, (k, _) in enumerate(comps)}

    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    ax.set_xscale("log")
    ax.grid(True, axis="x", which="major", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)

    for m, yi in zip(ORDER, y):
        s = stages[m]
        for key, fk in comps:
            v = s[fk]
            if v <= xlo:
                continue
            ax.barh(yi + off[key], v - xlo, h, left=xlo,
                    color=STAGE_COLORS[key], edgecolor="white",
                    linewidth=0.4, hatch=HATCH[key])
            ax.annotate(fmt_flops(v), (v, yi + off[key]),
                        textcoords="offset points", xytext=(4, 0),
                        ha="left", va="center", fontsize=13.5)

    # 简短标签 + 45° 斜放
    ax.set_yticks(y, [SHORT_LABELS[m] for m in ORDER],
                  fontsize=18, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", pad=8)
    ax.invert_yaxis()                # Ours 在最上方
    ax.set_xlabel("Computation (FLOPs)", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    vmax = max(max(stages[m][k] for m in ORDER) for _, k in comps)
    ax.set_xlim(xlo, vmax * 8)       # 右侧留约 0.9 个数量级放数值标注
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[k], hatch=HATCH[k])
               for k, _ in comps]
    # 图例: 2 排 (2×2), 字号 18 pt
    ax.legend(handles, [STAGE_LABELS[k] for k, _ in comps],
              fontsize=18, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, frameon=False, handlelength=1.3, handleheight=1.3,
              handletextpad=0.6, columnspacing=1.8, labelspacing=0.4)
    fig.tight_layout()

    save_fig(fig, f"flops_by_method_{config.DATASET}_ipc{ipc}")


# ============================================================
# 出图: 传输时延柱状图 (合成集上传到边缘), 线性 y 轴
# ============================================================

def plot_transfer_latency(stages, ipc, raw_bytes=None):
    """传输时延: 各方法合成集 (+ 压缩前原图参照) 上传到边缘设备所需时间

    统一部署量化口径 (1 B/标量): Ours 潜变量 int8, 基线像素 uint8,
    Raw = 压缩前原图 = 源池 (每类 1000 张, 共 10000 张像素 uint8)。
    时延 = 数据字节数 × 8 ÷ 带宽, 与 analysis.py efficiency 同口径; y 轴对数。
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    names = (["Raw"] if raw_bytes else []) + ["Ours", "GradMatch", "EL2N", "Forgetting"]
    delays = (([raw_bytes * 8 / (BANDWIDTH_MBPS * 1e6)] if raw_bytes else [])
              + [stages[m]["transfer_latency_s"] for m in ORDER])
    sizes = (([raw_bytes] if raw_bytes else [])
             + [stages[m]["transfer_bytes"] for m in ORDER])
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    # 方法级纹理: Raw 用斜线, 其余 4 方法各配一种 (黑白打印可区分)
    key_of = {"Raw": "raw", "Ours": "ours", "GradMatch": "gradmatch",
              "EL2N": "eln2", "Forgetting": "forgetting"}
    bars = ax.bar(x, delays, 0.62, color=colors,
                  hatch=[METHOD_HATCH[key_of[n]] for n in names])
    for bar, d, sz in zip(bars, delays, sizes):
        ax.annotate(f"{d:.2f} s\n{sz / 1024 / 1024:.2f} MB",
                    (bar.get_x() + bar.get_width() / 2, d),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=12)

    ax.set_yscale("log")
    ax.set_ylim(min(delays) * 0.5, max(delays) * 2.8)
    ax.set_xticks(x, names, fontsize=15)
    ax.legend(list(bars), names,
              fontsize=15, loc="upper right", framealpha=0.9,
              handlelength=2.0, labelspacing=0.5, borderpad=0.6,
              borderaxespad=0.6)
    ax.set_ylabel("Transmission latency (s)", fontsize=18)
    ax.tick_params(axis="y", labelsize=17)
    ax.grid(True, axis="y", which="both", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    save_fig(fig, f"transfer_latency_by_method_{config.DATASET}_ipc{ipc}",
             "传输时延图已保存")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="4 方法全流程训练能耗对比 (Ours/DC/EL2N/Forgetting, ResNet)")
    p.add_argument("--ipc", type=int, default=config.IPC)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--probe_epochs", type=int, default=8)
    p.add_argument("--probe_runs", type=int, default=3)
    p.add_argument("--T_dc", type=int, default=1000)
    p.add_argument("--real_bs", type=int, default=256)
    p.add_argument("--vae_epochs", type=int, default=None,
                   help="VAE 预训练轮数 (默认 config.AE_EPOCHS)")
    p.add_argument("--vae_train_n", type=int, default=VAE_TRAIN_N,
                   help="VAE 预训练样本数 (默认 49000 = CIFAR train 50000 - 2%% 验证)")
    p.add_argument("--device", type=str, default=None)
    a = p.parse_args()
    run(ipc=a.ipc, epochs=a.epochs, probe_epochs=a.probe_epochs,
        n_runs=a.probe_runs, T_dc=a.T_dc, real_bs=a.real_bs, device_str=a.device,
        vae_epochs=a.vae_epochs, vae_train_n=a.vae_train_n)
