"""标签预算实验（子命令）

  python label_budget.py pretrain --shots 5,20,100,500,4900  # 预训练 CE 只用 N/类
  python label_budget.py transfer                            # 预训练少标签 + 下游全标签
  python label_budget.py zeroshot --shots 1,5,20,50          # 零参数分类器 NCC / 1-NN

合并自 vae_label_budget.py / vae_pretrain_shot.py / zeroshot_probe.py。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from common import DEV, ROOT, resolve_ckpt, seed_all
from data import extract, load_split, make_noise
from models import DDPMBackbone, VAEBackbone
from train import evaluate, fit
from vae import EdgeVAE


# ============================================================
# 公共工具
# ============================================================

def subset_idx(y, shot, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for c in range(int(y.max()) + 1):
        ci = (y == c).nonzero().flatten()
        out.append(ci[torch.randperm(len(ci), generator=g)[:shot]])
    return torch.cat(out)


@torch.no_grad()
def encode(ae, x, bs=256):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(ae.encode_mu(x[i:i + bs].to(DEV)).float().cpu().half())
    return torch.cat(out)


def save_json(name, out):
    p = os.path.join(ROOT, "results", name)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果: {p}")


# ============================================================
# 子命令 1: 预训练 CE 标签预算
# ============================================================

def train_vae(x_all, y_all, lab_idx, num_classes, epochs, bs=128, lr=1e-3,
              beta=config.AE_BETA1, lam=config.AE_CLS_LAMBDA, seed=0,
              log_every=10):
    seed_all(seed)
    ae = EdgeVAE(latent_dim=16, num_classes=num_classes).to(DEV)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    n = x_all.size(0)
    lab_mask = torch.zeros(n, dtype=torch.bool)
    lab_mask[lab_idx] = True
    t0 = time.time()
    for ep in range(1, epochs + 1):
        ae.train()
        perm = torch.randperm(n)
        tot = rec_s = kl_s = cls_s = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            bx = x_all[idx].to(DEV)
            bs_mask = lab_mask[idx]
            opt.zero_grad(set_to_none=True)
            recon, z, logits, mu, logvar = ae(bx)
            rec = F.mse_loss(recon, bx)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = rec + beta * kl
            cls = torch.zeros(())
            if bs_mask.any():
                cls = F.cross_entropy(logits[bs_mask.to(DEV)], y_all[idx][bs_mask].to(DEV))
                loss = loss + lam * cls
            loss.backward()
            opt.step()
            tot += loss.item(); rec_s += rec.item(); kl_s += kl.item()
            cls_s += float(cls); nb += 1
        if ep % log_every == 0 or ep == 1:
            print(f"    ep {ep:>3}/{epochs} loss {tot/nb:.4f} rec {rec_s/nb:.4f} "
                  f"kl {kl_s/nb:.4f} ce {cls_s/nb:.3f}", flush=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae, time.time() - t0


def cmd_pretrain(a):
    num_classes = 10 if a.dataset == "cifar10" else 100
    d = load_split(a.dataset)
    x_tr = d["train"][0].float().div(127.5).sub(1)
    y_tr = d["train"][1]
    x_te, y_te = d["test"][0].float().div(127.5).sub(1), d["test"][1]
    x_va, y_va = d["val"][0].float().div(127.5).sub(1), d["val"][1]
    print(f"{a.dataset}: train {x_tr.size(0)} / test {x_te.size(0)} | "
          f"λ_CE={a.lam} | epochs={a.epochs}", flush=True)

    out = {}
    for shot in [int(s) for s in a.shots.split(",")]:
        print(f"\n===== {shot}/类 ({shot*num_classes} 个标签) =====", flush=True)
        li = subset_idx(y_tr, shot)
        ae, t_train = train_vae(x_tr, y_tr, li, num_classes, a.epochs, lam=a.lam)
        z_tr, z_te = encode(ae, x_tr), encode(ae, x_te)
        z_va = encode(ae, x_va)
        print(f"    VAE 训练 {t_train:.0f}s | 潜空间 {tuple(z_tr.shape[1:])}", flush=True)

        # 分类头也只用同一批标签
        accs = []
        for s in range(a.probe_seeds):
            net, info = fit(z_tr[li], y_tr[li], z_va, y_va,
                            "VAE", 16, "linear", s, epochs=a.probe_epochs,
                            verbose=False)
            r = evaluate(net, z_te, y_te, info["mean"], info["std"], True,
                         num_classes=num_classes)
            accs.append(r["acc"])
        acc = float(np.mean(accs)); std = float(np.std(accs))
        out[shot] = {"acc": acc, "std": std, "train_sec": t_train}
        print(f"  → {shot}-shot VAE: {acc:.2f}±{std:.2f}%", flush=True)

        ck = os.path.join(ROOT, "checkpoints", f"autoencoder_s8_c16_lab{shot}.pt")
        torch.save({"model_state_dict": ae.state_dict(), "shot": shot,
                    "lam": a.lam, "epochs": a.epochs}, ck)

    save_json(f"{a.tag}.json", out)
    for k, v in out.items():
        print(f"  {k:>5d}/类  {v['acc']:.2f}±{v['std']:.2f}%")


# ============================================================
# 子命令 2: 预训练少标签 + 下游全标签
# ============================================================

def cmd_transfer(a):
    d = load_split("cifar10")
    x_tr = d["train"][0].float().div(127.5).sub(1)
    y_tr = d["train"][1]
    x_va = d["val"][0].float().div(127.5).sub(1)
    y_va = d["val"][1]
    x_te = d["test"][0].float().div(127.5).sub(1)
    y_te = d["test"][1]

    out = {}
    for shot in [int(s) for s in a.shots.split(",")]:
        ck = os.path.join(ROOT, "checkpoints", f"autoencoder_s8_c16_lab{shot}.pt")
        if not os.path.exists(ck):
            print(f"跳过 {shot}: 缺少 {ck}", flush=True)
            continue
        ae = EdgeVAE(latent_dim=16, num_classes=10).to(DEV)
        ae.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True)["model_state_dict"])
        ae.eval()
        for p in ae.parameters():
            p.requires_grad = False

        z_tr, z_va, z_te = encode(ae, x_tr), encode(ae, x_va), encode(ae, x_te)
        accs = []
        for s in range(a.seeds):
            net, info = fit(z_tr, y_tr, z_va, y_va, "VAE", 16, "linear", s,
                            epochs=a.epochs, verbose=False)
            r = evaluate(net, z_te, y_te, info["mean"], info["std"], True, num_classes=10)
            accs.append(r["acc"])
        acc, std = float(np.mean(accs)), float(np.std(accs))
        out[shot] = {"acc": acc, "std": std}
        print(f"  预训练 {shot:>4}/类标签 → 下游全标签: {acc:.2f}±{std:.2f}%", flush=True)

    save_json(f"{a.tag}.json", out)


# ============================================================
# 子命令 3: 零参数分类器
# ============================================================

def zscore(z_tr, z_te):
    m = z_tr.mean(0, keepdim=True)
    s = z_tr.std(0, keepdim=True).clamp_min(1e-6)
    return (z_tr - m) / s, (z_te - m) / s


@torch.no_grad()
def ncc(z_tr, y_tr, z_te, y_te, n_cls):
    cent = torch.stack([z_tr[y_tr == c].mean(0) for c in range(n_cls)])
    d = torch.cdist(z_te, cent)
    return (d.argmin(1) == y_te).float().mean().item() * 100


@torch.no_grad()
def knn1(z_tr, y_tr, z_te, y_te):
    d = torch.cdist(z_te, z_tr)
    return (y_tr[d.argmin(1)] == y_te).float().mean().item() * 100


def cmd_zeroshot(a):
    n_cls = 10 if a.dataset == "cifar10" else 100
    d = load_split(a.dataset)
    noise = {s: make_noise(len(d[s][1])) for s in d}

    feats = {}
    v = VAEBackbone(n_cls, DEV, resolve_ckpt(a.dataset, "vae_unsup"))
    for s in ("train", "test"):
        f, y = extract(v, d[s][0], d[s][1], mode="VAE")
        feats[("VAE_unsup", s)] = (f.flatten(1), y)
    bb = DDPMBackbone(resolve_ckpt(a.dataset, "ddpm"), DEV)
    for s in ("train", "test"):
        f, y = extract(bb, d[s][0], d[s][1], mode="A", t=a.t, noise=noise[s])
        feats[("DDPM_A", s)] = (f.flatten(1), y)
    # DDPM 原始特征方差极大, 先做逐通道标准化再展平更稳
    print("特征已提取", {k[0]: tuple(vv[0].shape) for k, vv in feats.items()}, flush=True)

    out = {}
    for name in ("DDPM_A", "VAE_unsup"):
        z_tr_all, y_tr_all = feats[(name, "train")]
        z_te, y_te = feats[(name, "test")]
        for shot in [int(s) for s in a.shots.split(",")]:
            li = subset_idx(y_tr_all, shot)
            zt, ze = zscore(z_tr_all[li], z_te)
            n_acc = ncc(zt, y_tr_all[li], ze, y_te, n_cls)
            k_acc = knn1(zt, y_tr_all[li], ze, y_te) if shot <= 50 else float("nan")
            out[f"{name}_{shot}"] = {"ncc": n_acc, "knn1": k_acc}
            print(f"  {name:<11} {shot:>3}/类  NCC {n_acc:6.2f}%  1-NN {k_acc:6.2f}%", flush=True)

    save_json(f"{a.tag}.json", out)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="标签预算实验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain", help="预训练 CE 只用 N/类标签")
    p.add_argument("--shots", default="5,20,100,500")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--probe_seeds", type=int, default=5)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--lam", type=float, default=config.AE_CLS_LAMBDA)
    p.add_argument("--tag", default="vae_shot")
    p.set_defaults(fn=cmd_pretrain)

    p = sub.add_parser("transfer", help="预训练少标签 + 下游全标签")
    p.add_argument("--shots", default="5,20,100,500,4900")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tag", default="vae_pretrain_shot_fullprobe")
    p.set_defaults(fn=cmd_transfer)

    p = sub.add_parser("zeroshot", help="零参数分类器 NCC / 1-NN")
    p.add_argument("--shots", default="1,5,20,50")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--t", type=int, default=20)
    p.add_argument("--tag", default="zeroshot_probe")
    p.set_defaults(fn=cmd_zeroshot)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
