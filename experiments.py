"""DDPM vs VAE 主实验入口（子命令）

  python experiments.py core            # VAE / DDPM-A/B1/B2 冻结特征对比
  python experiments.py supervised ...  # DDPM-FT / DDPM-Joint
  python experiments.py decoder-probe   # 编码器 vs 解码器取层验证

合并自 run_core.py / ddpm_supervised.py / exp_decoder_probe.py。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

import config
from common import (DEV, ROOT, append_csv, count_flops, latency_ms,
                    resolve_ckpt, seed_all)
from data import extract, load_split, make_noise
from models import (LAYERS, DDPMBackbone, Pipeline, VAEBackbone,
                    build_head, build_projection)
from train import evaluate, fit, n_params

IN_CH = 256      # 现有多级 UNet 在 8×8 与 16×16 处均为 256 通道
OUT_DIR = os.path.join(ROOT, "checkpoints")


# ============================================================
# 子命令 1: 核心对比
# ============================================================

def cmd_core(a):
    modes = [m for m in a.modes.split(",") if m]
    norm = not a.no_norm
    seed_all(0)
    num_classes = 10 if a.dataset == "cifar10" else 100

    print(f"数据集 {a.dataset} | 类别 {num_classes} | modes {modes} | t={a.t} "
          f"| head={a.head} | noise={a.noise_mode} | norm={norm}", flush=True)
    data = load_split(a.dataset, val_ratio=a.val_ratio)
    if a.shot:
        # 所有方法共用同一批 shot 张图 (seed 固定), 保证标签预算完全一致
        x_tr, y_tr = data["train"]
        g = torch.Generator().manual_seed(0)
        sel = []
        for c in range(num_classes):
            ci = (y_tr == c).nonzero().flatten()
            sel.append(ci[torch.randperm(len(ci), generator=g)[:a.shot]])
        sel = torch.cat(sel)
        data["train"] = (x_tr[sel], y_tr[sel])
        print(f"{a.shot}-shot 子集: 每类 {a.shot} 张 → 共 {len(sel)} 张训练样本", flush=True)
    n_tr, n_va, n_te = (len(data[s][1]) for s in ("train", "val", "test"))
    print(f"train {n_tr} / val {n_va} / test {n_te}", flush=True)
    noise = {s: make_noise(len(data[s][1])) for s in data}

    bb = {}
    if any(m not in ("VAE", "VAE_unsup") for m in modes):
        ddpm_path = a.ddpm_ckpt or resolve_ckpt(a.dataset, "ddpm")
        bb["ddpm"] = DDPMBackbone(ddpm_path, DEV)
        print(f"DDPM 主干 {bb['ddpm'].n_params:,} 参数 | 预训练 {bb['ddpm'].pretrain_step} 步 "
              f"| loss {bb['ddpm'].pretrain_loss:.4f} | 权重 {os.path.basename(ddpm_path)}", flush=True)
    if "VAE" in modes:
        bb["vae"] = VAEBackbone(num_classes, DEV, resolve_ckpt(a.dataset, "vae"))
        print(f"VAE 主干 {bb['vae'].n_params:,} 参数 (重建+KL+分类监督)", flush=True)
    if "VAE_unsup" in modes:
        bb["vae_unsup"] = VAEBackbone(num_classes, DEV, resolve_ckpt(a.dataset, "vae_unsup"))
        print(f"VAE 主干 {bb['vae_unsup'].n_params:,} 参数 (仅重建+KL)", flush=True)

    rows = []
    groups = {}
    for m in modes:
        groups.setdefault("VAE" if m in ("VAE", "VAE_unsup") else LAYERS[m], []).append(m)

    for layer, ms in groups.items():
        if layer == "VAE":
            raw = {}
            for mode in ms:
                key = "vae" if mode == "VAE" else "vae_unsup"
                raw[mode] = {}
                for s in data:
                    f, y = extract(bb[key], data[s][0], data[s][1], mode="VAE")
                    raw[mode][s] = (f.half(), y)
                    print(f"  提取 {mode}/{s}: {tuple(f.shape)}", flush=True)
            # 统一成 {mode: {split: (feats, labels)}}
            raw_by_mode = raw
        else:
            raw = {}
            for s in data:
                f, y = extract(bb["ddpm"], data[s][0], data[s][1], mode=ms[0], t=a.t,
                               noise=noise[s], noise_mode=a.noise_mode)
                raw[s] = (f.half(), y)
                print(f"  提取 {layer}/{s}: {tuple(f.shape)}", flush=True)

        for mode in ms:
            cur = raw_by_mode[mode] if mode in ("VAE", "VAE_unsup") else raw
            bb_key = {"VAE": "vae", "VAE_unsup": "vae_unsup"}.get(mode, "ddpm")
            in_ch = 16 if mode in ("VAE", "VAE_unsup") else IN_CH
            for seed in range(a.seeds):
                print(f"\n[{mode}] seed {seed}", flush=True)
                net, info = fit(cur["train"][0], cur["train"][1],
                                cur["val"][0], cur["val"][1],
                                mode, in_ch, a.head, seed, epochs=a.epochs, norm=norm)
                te = evaluate(net, cur["test"][0], cur["test"][1],
                              info["mean"], info["std"], norm, num_classes=num_classes)
                proj, in_dim = build_projection(mode, in_ch)
                head = build_head(a.head, in_dim, num_classes)
                mean, std = info["mean"], info["std"]
                pipe = Pipeline(bb[bb_key], mode, proj, head, mean, std, norm,
                                layer=None if mode in ("VAE", "VAE_unsup") else layer).to(DEV).eval()
                x1 = data["test"][0][:1].to(DEV).float().div(127.5).sub(1)
                nb = noise["test"][:1].to(DEV).float()
                pipe(x1, a.t, nb, a.noise_mode)
                flops = count_flops(lambda: pipe(x1, a.t, nb, a.noise_mode))
                lat = latency_ms(lambda: pipe(x1, a.t, nb, a.noise_mode))

                row = {
                    "dataset": a.dataset, "mode": mode, "t": a.t, "seed": seed,
                    "head": a.head, "noise_mode": a.noise_mode,
                    "val_acc": round(info["val_acc"], 2), "test_acc": round(te["acc"], 2),
                    "test_f1": round(te["f1"], 2),
                    "backbone_params": bb[bb_key].n_params,
                    "proj_params": n_params(proj), "head_params": n_params(head),
                    "feat_elems": int(in_dim),
                    "feat_bytes_fp32": int(in_dim) * 4,
                    "flops_m": round(flops, 1), "latency_ms": round(lat, 2),
                    "train_sec": round(info["train_sec"], 1),
                }
                rows.append(row)
                print(f"  val {row['val_acc']:.2f}% | test {row['test_acc']:.2f}% "
                      f"| F1 {row['test_f1']:.2f} | {flops:.1f} MFLOPs | {lat:.2f} ms", flush=True)
        del raw
        if layer == "VAE":
            del raw_by_mode

    tag = a.tag or f"{a.dataset}_t{a.t}_{a.head}_{a.noise_mode}"
    csv_path = os.path.join(a.out_dir, f"core_{tag}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    append_csv(csv_path, rows)

    print("\n汇总 (mean ± std over seeds):")
    summary = {}
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        acc = np.array([r["test_acc"] for r in rs])
        f1 = np.array([r["test_f1"] for r in rs])
        summary[mode] = {"acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
                         "f1_mean": float(f1.mean()), "f1_std": float(f1.std())}
        print(f"  {mode:<6s} acc {acc.mean():.2f}±{acc.std():.2f} | "
              f"F1 {f1.mean():.2f}±{f1.std():.2f} | "
              f"proj {rs[0]['proj_params']:,} + head {rs[0]['head_params']:,} | "
              f"{rs[0]['flops_m']} MFLOPs | {rs[0]['latency_ms']} ms")

    with open(os.path.join(a.out_dir, f"core_{tag}.json"), "w") as f:
        args = {k: v for k, v in vars(a).items() if k != "fn"}
        json.dump({"args": args, "summary": summary, "rows": rows}, f, indent=2)
    print(f"\n结果: {csv_path}")


# ============================================================
# 子命令 2: DDPM-FT / DDPM-Joint
# ============================================================

ENC_MODULES = ("conv_in", "time_proj", "time_embedding",
               "down_blocks.0", "down_blocks.1", "down_blocks.2")


class ClsHead(nn.Module):
    def __init__(self, in_ch=256, mid=16, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(8, mid),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(mid * 8 * 8, num_classes),
        )

    def forward(self, f):
        return self.net(f)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


class DenoiserWithHead(nn.Module):
    """UNet + 分类头; encode_mid 只跑编码器到 8×8 (下采样前)"""

    def __init__(self, unet, head):
        super().__init__()
        self.unet = unet
        self.head = head

    def encode_mid(self, x):
        t = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        return self.unet_forward_features(x, t)

    def unet_forward_features(self, x_t, t):
        u = self.unet
        temb = u.time_embedding(u.time_proj(t), None).to(x_t.dtype)
        h = u.conv_in(x_t)
        h, res0 = u.down_blocks[0](h, temb)
        h, res1 = u.down_blocks[1](h, temb)
        h, res2 = u.down_blocks[2](h, temb)
        return res2[-1] if res2[-1].shape[-1] == res2[-2].shape[-1] else res2[-2]

    def forward(self, x_t, t):
        return self.unet(x_t, t).sample


def add_noise(x0, abar, noise):
    return abar.sqrt() * x0 + (1 - abar).sqrt() * noise


def train_supervised(mode, dataset, steps, init=None, lr=None, lam=0.1, batch=128,
                     t_cls_max=200, ema_decay=0.9999, warmup=1000, bench=0, seed=0,
                     ckpt_name=None, amp=True, augment=True):
    seed_all(seed)
    if mode == "ft":
        lr = lr or 1e-4
        init = init or os.path.join(ROOT, "checkpoints", f"ddpm_{dataset}.pt")
        ckpt_name = ckpt_name or f"ddpm_{dataset}_ft.pt"
    else:
        lr = lr or 2e-4
        ckpt_name = ckpt_name or f"ddpm_{dataset}_joint.pt"

    split = load_split(dataset)
    x_tr = split["train"][0].float().div(127.5).sub(1)
    y_tr = split["train"][1]
    n_cls = 10 if dataset == "cifar10" else 100

    if init:
        ck = torch.load(init, map_location="cpu", weights_only=True)
        unet = UNet2DModel(**ck["arch"])
        unet.load_state_dict(ck["ema_state_dict"])
        sched = DDPMScheduler(**ck["scheduler"])
        arch_cfg, sched_cfg = ck["arch"], ck["scheduler"]
        print(f"初始化自 {os.path.basename(init)} (step {ck.get('step')})", flush=True)
    else:
        from diffusion import ARCH, SCHED
        unet = UNet2DModel(**ARCH)
        sched = DDPMScheduler(**SCHED)
        arch_cfg, sched_cfg = dict(ARCH), dict(SCHED)
        print("从零初始化", flush=True)

    unet = unet.to(DEV)
    head = ClsHead(256, 16, n_cls).to(DEV)
    model = DenoiserWithHead(unet, head).to(DEV)

    # 冻结: FT 模式下只更新编码器路径; Joint 模式全参可训
    for n, p in unet.named_parameters():
        p.requires_grad = mode == "joint" or n.startswith(ENC_MODULES)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模式 {mode} | 可训参数 {n_train:,} / 总 {sum(p.numel() for p in model.parameters()):,} "
          f"| λ={lam} | lr={lr} | t_cls~U[0,{t_cls_max}) | steps={steps}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, ema_decay)
    abar = sched.alphas_cumprod.to(DEV)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    flip = augment
    step, run_d, run_c, run_a, t0 = 0, 0.0, 0.0, 0.0, time.time()
    pbar = tqdm(total=steps, desc=f"ddpm-{mode}")
    done = False
    while not done:
        perm = torch.randperm(x_tr.size(0))
        for i in range(0, x_tr.size(0), batch):
            if step >= steps:
                done = True
                break
            idx = perm[i:i + batch]
            x0 = x_tr[idx].to(DEV, non_blocking=True)
            y = y_tr[idx].to(DEV, non_blocking=True)
            if flip:
                m = torch.rand(x0.size(0), device=DEV) < 0.5
                x0 = torch.where(m[:, None, None, None], x0.flip(-1), x0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            # 分类分支: 独立低噪声时间步, 只跑编码器
            t_cls = torch.randint(0, t_cls_max, (x0.size(0),), device=DEV)
            eps_c = torch.randn_like(x0)
            a_c = abar[t_cls].view(-1, 1, 1, 1)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                feat = model.unet_forward_features(add_noise(x0, a_c, eps_c), t_cls)
                logits = model.head(feat).float()
                loss_c = F.cross_entropy(logits, y)
                acc = (logits.argmax(1) == y).float().mean()

            if mode == "joint":
                t = torch.randint(0, 1000, (x0.size(0),), device=DEV)
                eps = torch.randn_like(x0)
                a = abar[t].view(-1, 1, 1, 1)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    pred = model(add_noise(x0, a, eps), t).float()
                    loss_d = F.mse_loss(pred, eps)
                loss = loss_d + lam * loss_c
            else:
                loss_d = torch.zeros(())
                loss = loss_c

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            step += 1
            run_d += float(loss_d)
            run_c += float(loss_c)
            run_a += float(acc)
            if step % 50 == 0:
                pbar.set_postfix(mse=f"{run_d/50:.4f}", ce=f"{run_c/50:.3f}",
                                 acc=f"{run_a/50*100:.1f}%")
                run_d = run_c = run_a = 0.0
            pbar.update(1)
            if bench and step >= bench:
                pbar.close()
                dt = time.time() - t0
                print(f"\n测速 {bench} 步 / {dt:.1f}s = {bench/dt:.2f} step/s "
                      f"→ {steps/(bench/dt)/3600:.2f} 小时跑完 {steps} 步", flush=True)
                return
    pbar.close()

    state = {k: v for k, v in ema.shadow.items()}
    unet_keys = {k[len("unet."):]: v for k, v in state.items() if k.startswith("unet.")}
    head_keys = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    ckpt = {"arch": arch_cfg, "scheduler": sched_cfg,
            "model_state_dict": unet_keys, "ema_state_dict": unet_keys,
            "head_state_dict": head_keys,
            "step": step, "loss": float(loss_c), "mode": mode, "lam": lam}
    if mode == "ft":
        ckpt["pretrained_from"] = os.path.basename(init)
    path = os.path.join(OUT_DIR, ckpt_name)
    torch.save(ckpt, path)
    print(f"权重已保存: {path} ({time.time()-t0:.0f}s)", flush=True)


def cmd_supervised(a):
    steps = a.steps or (8000 if a.mode == "ft" else 100_000)
    train_supervised(a.mode, a.dataset, steps, init=a.init, lr=a.lr, lam=a.lam,
                     batch=a.batch, t_cls_max=a.t_cls_max, ema_decay=a.ema,
                     bench=a.bench, ckpt_name=a.ckpt,
                     amp=not a.no_amp, augment=not a.no_augment)


# ============================================================
# 子命令 3: 编码器 vs 解码器取层
# ============================================================

LAYERS_PROBE = {
    "down2_8x8": "down_blocks.2.resnets.1",
    "mid_4x4":   "mid_block",
    "up1_8x8.0": "up_blocks.1.resnets.0",
    "up1_8x8.1": "up_blocks.1.resnets.1",
    "up1_8x8.2": "up_blocks.1.resnets.2",
}


def cmd_decoder_probe(a):
    from diffusion import (balanced_subset, eval_features, extract_features,
                           load_cifar_arrays, load_frozen_unet)

    dev = torch.device("cuda")
    x_tr, y_tr, x_te, y_te = load_cifar_arrays(config.DATA_ROOT)
    idx = balanced_subset(y_tr, 100)
    print(f"扫描数据: train {len(idx)} 点 | test {x_te.size(0)} 点", flush=True)
    unet, sched = load_frozen_unet(dev, a.ckpt)

    names = [s for s in a.layers.split(",") if s]
    out = {}
    for t in [int(v) for v in a.t.split(",")]:
        for name in names:
            mod = LAYERS_PROBE[name]
            f_tr = extract_features(unet, sched, x_tr[idx], mod, t, dev, noise_reps=a.reps)
            f_te = extract_features(unet, sched, x_te, mod, t, dev, noise_reps=a.reps)
            accs = eval_features("resnet", f_tr, y_tr[idx], f_te, y_te, dev,
                                 seeds=a.seeds, epochs=a.epochs)
            m, s = float(np.mean(accs)), float(np.std(accs))
            out[f"{name}@t{t}"] = {"acc": m, "std": s, "shape": list(f_tr.shape[1:])}
            print(f"  {name:11s} t={t:<4d} {tuple(f_tr.shape[1:])}  "
                  f"acc = {m:.2f}±{s:.2f}%", flush=True)
            del f_tr, f_te

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    p = os.path.join(config.RESULT_DIR, "decoder_vs_encoder_8x8.json")
    with open(p, "w") as f:
        json.dump({"ckpt": a.ckpt, "protocol": {"points": len(idx), "seeds": a.seeds,
                                                "epochs": a.epochs, "reps": a.reps},
                   "results": out}, f, indent=1)
    print(f"\n已保存: {p}")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="DDPM vs VAE 主实验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("core", help="VAE / DDPM-A/B1/B2 冻结特征对比")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--t", type=int, default=0)
    p.add_argument("--modes", default="A,B1,B2,VAE")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--noise_mode", default="schedule", choices=["schedule", "clean"])
    p.add_argument("--no_norm", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--ddpm_ckpt", default=None,
                   help="覆盖 DDPM 权重路径 (评估微调/联合监督后的主干)")
    p.add_argument("--shot", type=int, default=0,
                   help="每类只用 N 个训练样本 (0=全量); 5-shot 标签预算实验")
    p.add_argument("--val_ratio", type=float, default=0.02,
                   help="从官方训练集划出的验证集比例 (与主项目一致)")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results"))
    p.set_defaults(fn=cmd_core)

    p = sub.add_parser("supervised", help="DDPM-FT / DDPM-Joint")
    p.add_argument("--mode", required=True, choices=["ft", "joint"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--t_cls_max", type=int, default=200)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--bench", type=int, default=0)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.set_defaults(fn=cmd_supervised)

    p = sub.add_parser("decoder-probe", help="编码器 vs 解码器取层验证")
    p.add_argument("--t", default="10")
    p.add_argument("--layers", default=",".join(LAYERS_PROBE))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--ckpt", default="checkpoints/ddpm_cifar10.pt")
    p.set_defaults(fn=cmd_decoder_probe)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
