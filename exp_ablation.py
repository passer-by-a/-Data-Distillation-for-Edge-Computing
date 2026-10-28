"""
带分类头 vs 无分类头 EdgeVAE 对比 (cifar10): vit 与 resnet 在同一张图上

目的: 直观比较"分类监督 VAE"与"纯重建 VAE"对下游潜空间分类的影响。
同一套蒸馏参数 (center=mean, ipc, subset), 只是 VAE 不同 (autoencoder_edge_s8_c16.pt
带分类头 vs autoencoder_edge_s8_c16_noclspart.pt 无分类头)。两种 VAE 各自编码测试集
并各自蒸馏, 下游在各自蒸馏潜变量上训练 vit 与 resnet, 4 条曲线画到同一张双面板图。

运行 (需先有两套蒸馏产物):
  python distill.py --center mean --no_cls      # 无分类头蒸馏 (带分类头产物可复用)
  python exp_ablation.py

出图: {RESULT_DIR}/backbone_cls_vs_noclspart_vit_resnet_d{latent_dim}_edge.png
"""

import os
import argparse
import numpy as np
import torch

import config
from backbones import ResNetBackbone, ViTBackbone, \
    train_eval_backbone, load_syn_center
from config import get_cifar10_loaders
from utils import encode_test_latents
from vae import load_edge_vae

BACKBONES = {"resnet": ResNetBackbone, "vit": ViTBackbone}


def plot_combined(curves, path, title):
    """拆成两张单面板 (无大标题, 论文友好): {stem}_train.png / {stem}_test.png
    curves: name -> (acc, train, test); noclspart 用虚线"""
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
            ls = "--" if "noclspart" in name else "-"
            ax.plot(c, lw=1.8, ls=ls, marker="o", markevery=5, markersize=6,
                    label=f"{name} ({acc:.2f}%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = f"{stem}_{split}.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"组合对比图已保存: {out}")


def main(ipc=100, subset=1000, center="mean", seeds=3, epochs=50):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | center={center} | ipc={ipc} | subset={subset} | seeds={seeds}")

    _, _, test_loader = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS)
    C = config.NUM_CLASSES

    curves = {}
    results = {}
    for no_cls in (False, True):
        tag = "noclspart" if no_cls else "cls"
        ckpt = "autoencoder_edge_s8_c16_noclspart.pt" if no_cls else None
        ae = load_edge_vae(device, ckpt_name=ckpt)
        test_latents = encode_test_latents(test_loader, ae, device)

        syn_path = load_syn_center(center, ipc, no_cls=no_cls)
        if syn_path is None:
            raise FileNotFoundError(
                f"缺 {( '无分类头' if no_cls else '带分类头' )} 蒸馏产物, 请先跑 "
                f"python distill.py --ipc {ipc} --subset {subset} --center {center}"
                f"{' --no_cls' if no_cls else ''}")
        syn = torch.load(syn_path, map_location="cpu", weights_only=True)["syn_latents"]
        z_syn = syn.reshape(C * ipc, -1, syn.size(-2), syn.size(-1))
        labels = torch.arange(C).repeat_interleave(ipc)

        print(f"\n[{tag}] 蒸馏产物: {os.path.basename(syn_path)} | 训练点 {z_syn.size(0)}")
        for name, cls in BACKBONES.items():
            accs, tr_c, te_c = [], None, None
            for s in range(seeds):
                if s == 0:
                    a, tr_c, te_c = train_eval_backbone(
                        cls, z_syn, labels, test_loader, ae, device, seed=s,
                        epochs=epochs, return_curve=True, test_latents=test_latents,
                        lr_schedule="cosine")
                else:
                    a = train_eval_backbone(cls, z_syn, labels, test_loader, ae, device,
                                            seed=s, epochs=epochs, test_latents=test_latents,
                                            lr_schedule="cosine")
                accs.append(a)
            acc = float(np.mean(accs))
            curves[f"{tag}_{name}"] = (acc, tr_c, te_c)
            results.setdefault(tag, {})[name] = {
                "acc": acc, "std": float(np.std(accs)), "seeds_acc": accs,
                "train_curve": tr_c, "test_curve": te_c,
                "syn_path": os.path.basename(syn_path)}
            print(f"  {name}: {acc:.2f}%  ±{np.std(accs):.2f}  (seeds={accs})")

    latent_dim = 16
    title = (f"CLS vs No-CLS EdgeVAE (Latent-Space, ipc={ipc}, {center}, "
             f"dim={latent_dim}, arch=edge)")
    path = os.path.join(config.RESULT_DIR,
                        f"backbone_cls_vs_noclspart_{'_'.join(BACKBONES)}"
                        f"_d{latent_dim}_edge.png")
    plot_combined(curves, path, title)

    # ---- 落盘结构化数据 (消融: 分类监督的贡献) ----
    import json
    from datetime import datetime
    dump = {"meta": {"exp": "VAE cls-head ablation", "ipc": ipc, "subset": subset,
                     "center": center, "seeds": seeds, "epochs": epochs,
                     "arch": "edge", "latent_dim": latent_dim,
                     "vaes": {"cls": "autoencoder_edge_s8_c16.pt",
                              "noclspart": "autoencoder_edge_s8_c16_noclspart.pt"},
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            "results": results}
    jpath = os.path.join(config.RESULT_DIR, f"ablation_cls_vs_noclspart_ipc{ipc}.json")
    with open(jpath, "w") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)
    print(f"消融数据已保存: {jpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="带CLS vs 不含分类头 VAE 下游 vit/resnet 对比")
    parser.add_argument("--ipc", type=int, default=config.IPC)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--center", type=str, default="mean",
                        choices=["mean", "medoid", "herding"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(ipc=args.ipc, subset=args.subset, center=args.center,
         seeds=args.seeds, epochs=args.epochs)