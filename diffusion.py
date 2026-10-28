"""
扩散对照线: 自训 DDPM + 冻结特征下游对照

两个子命令:
  train    从零训练 DDPM (CIFAR-10 32×32) —— "截取 UNet 做下游"的对照实验前半段
  compare  冻结 UNet 提中间层特征 → 训练下游 backbone, 作为上界/对照参照

为什么要有自训版 (对比 compare 里的官方模型):
  google/ddpm-cifar10-32 是官方预训练 (~800k 步, 不可部署) 的上界参照;
  自训 DDPM —— 数据切分、训练环境完全可控, 构成"无标签生成式预训练" vs
  "我们带分类监督的潜空间蒸馏" 的公平对照。

训练配置对齐官方 CIFAR-10 DDPM (保证中间层索引与特征提取代码兼容):
  架构: UNet2DModel, block_out_channels=(128,256,256,256), layers_per_block=2,
        后 3 个 down/up block 带 attention
  调度: DDPMScheduler(1000 步, linear β: 1e-4 → 0.02)
  目标: ε-prediction MSE (完全无标签)
  优化: Adam(2e-4) + EMA(0.9999) + 梯度裁剪 1.0 + AMP
  数据: config.get_cifar10_loaders 的训练集 (±2% 验证剔除), 与主实验同一份, 只用图像不用标签

运行:
  python diffusion.py train --steps 100000
  python diffusion.py train --steps 100000 --resume checkpoints/ddpm_cifar10.pt   # 断点续训
  python diffusion.py compare --sweep --ckpt checkpoints/ddpm_cifar10.pt
  python diffusion.py compare --full  --ckpt checkpoints/ddpm_cifar10.pt
  HF_ENDPOINT=https://hf-mirror.com python diffusion.py compare --sweep           # 官方模型上界

产物:
  checkpoints/ddpm_cifar10.pt              (最新, 含 EMA 权重)
  checkpoints/ddpm_cifar10_step{N}.pt      (周期快照)
  results/ddpm_samples_step{N}.png         (DDIM 采样抽检)
  results/diffusion_sweep{_tag}.json
  results/diffusion_upperbound{_tag}.json
  results/diffusion_upperbound_compare{_tag}.png
  (compare 加 --ckpt 时 _tag=_selfddpm, 不覆盖官方模型结果)
"""

import argparse
import functools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

import config
from backbones import BACKBONES, count_params, save_json, train_eval_backbone
from config import denormalize, get_cifar10_loaders

# 与 google/ddpm-cifar10-32 一致的架构 (层名 down_blocks.*.resnets.* / mid_block 可直接 hook)
ARCH = dict(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 256, 256),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)
SCHED = dict(num_train_timesteps=1000, beta_schedule="linear",
             beta_start=0.0001, beta_end=0.02)
CKPT_NAME = "ddpm_cifar10.pt"

MODEL_ID = "google/ddpm-cifar10-32"
C = config.NUM_CLASSES

# 候选中间层 (都是 256 通道; 键名 = 空间分辨率标注)
LAYER_CHOICES = {
    "down1_16x16": "down_blocks.1.resnets.1",
    "down2_8x8":   "down_blocks.2.resnets.1",   # 空间分辨率与我们的潜空间一致
    "down3_4x4":   "down_blocks.3.resnets.1",
    "mid_4x4":     "mid_block",                 # bottleneck
}
T_CHOICES = [0, 10, 50, 100, 200, 500]


# ============================================================
# 一、DDPM 训练 (子命令 train)
# ============================================================

class EMA:
    """权重指数滑动平均 (DDPM 必需, 直接决定采样与特征质量)"""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def build_unet():
    return UNet2DModel(**ARCH)


def build_scheduler(device=None):
    sched = DDPMScheduler(**SCHED)
    if device is not None:
        sched.alphas_cumprod = sched.alphas_cumprod.to(device)
    return sched


def save_ckpt(path, unet, ema, step, loss):
    torch.save({"step": step, "loss": loss, "arch": ARCH, "scheduler": SCHED,
                "model_state_dict": unet.state_dict(),
                "ema_state_dict": ema.state_dict()}, path)


def save_samples(unet, ema, path, device, n=16, steps=25, seed=0):
    """用 EMA 权重 + DDIM 25 步采样抽检, 确认模型真的学到了分布"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffusers import DDIMScheduler

    raw = {k: v.detach().clone() for k, v in unet.state_dict().items()}
    unet.load_state_dict(ema.state_dict())

    ddim = DDIMScheduler(**SCHED)
    ddim.set_timesteps(steps)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 3, ARCH["sample_size"], ARCH["sample_size"],
                    generator=g, device=device)
    with torch.no_grad():
        for t in ddim.timesteps:
            eps = unet(x, t.to(device)).sample
            x = ddim.step(eps, t, x).prev_sample

    unet.load_state_dict(raw)                       # 还原训练权重

    imgs = denormalize(x).cpu().permute(0, 2, 3, 1).numpy()
    gs = int(n ** 0.5)
    fig, axes = plt.subplots(gs, gs, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i]); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close()
    print(f"  采样抽检已保存: {path}")


def train(steps=100000, batch_size=None, lr=2e-4, ema_decay=0.9999,
          save_every=10000, sample_every=20000, amp=True, resume=None, seed=0):
    torch.backends.cudnn.benchmark = True      # 输入尺寸固定, 开启卷积自动调优
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or config.BATCH_SIZE
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    train_loader, _, _ = get_cifar10_loaders(
        data_root=config.DATA_ROOT, batch_size=batch_size,
        num_workers=config.NUM_WORKERS, augment=True)

    unet = build_unet().to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    sched = build_scheduler(device)
    opt = torch.optim.Adam(unet.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(unet, ema_decay)

    print(f"设备: {device} | DDPM (ε-prediction, 无标签) | {n_params:,} 参数")
    print(f"目标步数: {steps} | batch={batch_size} | lr={lr} | EMA={ema_decay} | AMP={amp}")

    start = 0
    ckpt_path = os.path.join(config.SAVE_DIR, CKPT_NAME)
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        unet.load_state_dict(ckpt["model_state_dict"])
        ema.shadow = {k: v.to(device).clone() for k, v in ckpt["ema_state_dict"].items()}
        start = ckpt["step"]
        print(f"断点续训: {resume} (step={start})")

    step, avg_loss, n_log = start, 0.0, 0
    t0 = time.time()
    pbar = tqdm(total=steps, initial=start, desc="DDPM")
    stop = False
    while not stop:
        for x, _ in train_loader:            # 标签不使用 (完全无监督)
            if step >= steps:
                stop = True
                break
            x = x.to(device, non_blocking=True)
            noise = torch.randn_like(x)
            t = torch.randint(0, SCHED["num_train_timesteps"], (x.size(0),), device=device)
            x_t = sched.add_noise(x, noise, t)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = F.mse_loss(unet(x_t, t).sample.float(), noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(unet)

            step += 1
            avg_loss += loss.item()
            n_log += 1
            pbar.update(1)

            if step % 100 == 0:
                sps = (step - start) / (time.time() - t0)
                pbar.set_postfix(loss=f"{avg_loss / n_log:.4f}", steps_s=f"{sps:.1f}")
                avg_loss, n_log = 0.0, 0

            if step % save_every == 0:
                save_ckpt(ckpt_path, unet, ema, step, loss.item())
                snap = os.path.join(config.SAVE_DIR,
                                    CKPT_NAME.replace(".pt", f"_step{step}.pt"))
                save_ckpt(snap, unet, ema, step, loss.item())
                print(f"\n  [保存] step={step} → {ckpt_path} / {snap}")

            if sample_every and step % sample_every == 0:
                save_samples(unet, ema,
                             os.path.join(config.RESULT_DIR, f"ddpm_samples_step{step}.png"),
                             device)

    pbar.close()
    save_ckpt(ckpt_path, unet, ema, step, loss.item())
    print(f"训练完成: {steps} 步 | 最终 ckpt: {ckpt_path}")


# ============================================================
# 二、冻结扩散特征对照 (子命令 compare)
# ============================================================

# ============================================================
# 数据: 直接读原始数组 (等价 ToTensor+Normalize, 但比遍历 DataLoader 快得多)
# ============================================================

def load_cifar_arrays(data_root):
    """返回 (x_train [49000,3,32,32], y_train, x_test [10000,3,32,32], y_test), 均在 [-1,1]

    训练集划分复刻 config.get_cifar10_loaders: RandomState(42) 打乱后前 2% 作验证集并剔除,
    保证与主实验用同一份训练数据, 不泄漏。
    """
    ds_tr = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    ds_te = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)

    def to_tensor(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255).sub_(0.5).div_(0.5)
        return x, torch.tensor(ds.targets, dtype=torch.long)

    x_tr, y_tr = to_tensor(ds_tr)
    x_te, y_te = to_tensor(ds_te)
    n, n_val = x_tr.size(0), int(x_tr.size(0) * 0.02)
    keep = np.random.RandomState(42).permutation(n)[n_val:]
    return x_tr[keep], y_tr[keep], x_te, y_te


def balanced_subset(y, n_per_class, seed=0):
    """每类取 n_per_class 张的下标 (与蒸馏 IPC 预算口径一致)"""
    g = np.random.RandomState(seed)
    parts = [g.choice(np.where(y.numpy() == c)[0], n_per_class, replace=False) for c in range(C)]
    return np.sort(np.concatenate(parts))


# ============================================================
# 冻结 UNet + 特征提取
# ============================================================

def load_frozen_unet(device, ckpt=None):
    """加载冻结 DDPM UNet

    ckpt=None → HF 官方 google/ddpm-cifar10-32
    ckpt=路径 → 我们自己 train 子命令训出的权重 (用 EMA 权重, 特征质量更好),
                调度器用存档里的配置, 保证 ᾱ 曲线与训练时一致。
    """
    if ckpt is None:
        unet = UNet2DModel.from_pretrained(MODEL_ID).to(device).eval()
        sched = DDPMScheduler.from_pretrained(MODEL_ID)
        desc = MODEL_ID
    else:
        c = torch.load(ckpt, map_location="cpu", weights_only=True)
        unet = UNet2DModel(**c["arch"]).to(device).eval()
        unet.load_state_dict(c["ema_state_dict"])
        sched = DDPMScheduler(**c["scheduler"])
        desc = f"{ckpt} (self-trained, step={c.get('step')}, EMA 权重)"
    unet.requires_grad_(False)
    n = sum(p.numel() for p in unet.parameters())
    print(f"已加载冻结扩散 UNet: {desc} | {n:,} 参数 | "
          f"训练步数 {sched.config.num_train_timesteps}")
    return unet, sched


@torch.no_grad()
def extract_features(unet, sched, images, layer, t, device, batch=256, noise_reps=4):
    """冻结 UNet 提特征 → [N, 256, H, W] float32 (cpu)

    按 DDPM 前向加噪: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε (t=0 时 ᾱ≈1, 基本等于原图),
    再对 noise_reps 次噪声实现取平均 —— 单次噪声实现会让大 t 特征被该噪声主导。
    """
    mod = dict(unet.named_modules())[layer]
    abar_all = sched.alphas_cumprod.cpu()
    acc = None
    for rep in range(noise_reps):
        cap = {}

        def hook(m, inp, out, _cap=cap):
            _cap["f"] = out[0] if isinstance(out, tuple) else out

        h = mod.register_forward_hook(hook)
        chunks = []
        g = torch.Generator().manual_seed(1234 + rep)      # 固定噪声, 保证可复现
        for i in range(0, images.size(0), batch):
            xb = images[i:i + batch]
            eps = torch.randn(xb.shape, generator=g)
            tb = torch.full((xb.size(0),), t, dtype=torch.long)
            abar = abar_all[tb].view(-1, 1, 1, 1)
            unet((abar.sqrt() * xb + (1 - abar).sqrt() * eps).to(device), tb.to(device))
            chunks.append(cap["f"].float().cpu())
        h.remove()
        cur = torch.cat(chunks, 0)
        acc = cur if acc is None else acc + cur
    return acc / noise_reps


def make_backbone(name, spatial):
    """ViT 需要显式 spatial (patch embed 网格), 其余 backbone 不用"""
    cls = BACKBONES[name]
    return functools.partial(cls, spatial=spatial) if name == "vit" else cls


def eval_features(backbone_name, f_train, y_train, f_test, y_test, device,
                  seeds=6, epochs=50):
    """在预提特征上训练 backbone (口径与主实验一致: 50 epoch / lr 1e-3 / cosine)"""
    cls = make_backbone(backbone_name, f_train.shape[-1])
    test_latents = (f_test.flatten(1), y_test)
    return [train_eval_backbone(cls, f_train, y_train, None, None, device,
                                seed=s, epochs=epochs, lr=0.001,
                                test_latents=test_latents)
            for s in range(seeds)]


# ============================================================
# 阶段1: 扫 中间层 × timestep
# ============================================================

def run_sweep(device, n_per_class=100, probe_epochs=30, probe_seeds=2, noise_reps=4,
              ckpt=None, out_tag=""):
    x_tr, y_tr, x_te, y_te = load_cifar_arrays(config.DATA_ROOT)
    idx = balanced_subset(y_tr, n_per_class)
    print(f"扫描数据: train {len(idx)} 点 (每类 {n_per_class}) | test {x_te.size(0)} 点")
    unet, sched = load_frozen_unet(device, ckpt)

    records = []
    for lname, layer in LAYER_CHOICES.items():
        for t in T_CHOICES:
            f_tr = extract_features(unet, sched, x_tr[idx], layer, t, device,
                                    noise_reps=noise_reps)
            f_te = extract_features(unet, sched, x_te, layer, t, device,
                                    noise_reps=noise_reps)
            accs = eval_features("resnet", f_tr, y_tr[idx], f_te, y_te, device,
                                 seeds=probe_seeds, epochs=probe_epochs)
            acc, std = float(np.mean(accs)), float(np.std(accs))
            records.append({"layer": lname, "module": layer, "t": t,
                            "spatial": int(f_tr.shape[-1]), "acc": acc, "std": std})
            print(f"  {lname:12s} t={t:<4d} {f_tr.shape[-1]}×{f_tr.shape[-1]}  "
                  f"probe acc = {acc:.2f}±{std:.2f}%")

    best = max(records, key=lambda r: r["acc"])
    print(f"\n最优配置: {best['layer']} (t={best['t']}) → {best['acc']:.2f}%")
    save_json({"records": records}, {
        "exp": "diffusion feature sweep (layer × timestep)",
        "model": ckpt or MODEL_ID, "probe_backbone": "resnet", "probe_epochs": probe_epochs,
        "probe_seeds": probe_seeds, "noise_reps": noise_reps, "points": len(idx),
        "t_choices": T_CHOICES, "layers": LAYER_CHOICES, "best": best},
        f"diffusion_sweep{out_tag}.json")
    return best


# ============================================================
# 阶段2: 完整对比 (最优配置 × 4 backbone × 2 数据预算)
# ============================================================

def load_baseline(fname):
    """读主实验蒸馏结果作为对比基线 (兼容 method_x_backbone 新格式)"""
    p = os.path.join(config.RESULT_DIR, fname)
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return {k: (v["acc"], v["std"], v["params"]) for k, v in d["results"].items()}
    # 旧 backbone_mean_* 产物已移除, 回退到 4 方法 × 4 backbone 网格结果里的 Ours 列
    alt = os.path.join(config.RESULT_DIR, "method_x_backbone_cifar10_ipc100_pixel.json")
    if os.path.exists(alt):
        with open(alt) as f:
            d = json.load(f)
        return {k: (v["ours"]["acc"], v["ours"]["std"], v["ours"].get("params"))
                for k, v in d["results"].items()}
    print(f"[跳过] 未找到基线 {fname}")
    return None


def run_full(device, layer, t, seeds=6, epochs=50, noise_reps=4, ckpt=None,
             out_tag="", no_full_budget=False):
    x_tr, y_tr, x_te, y_te = load_cifar_arrays(config.DATA_ROOT)
    unet, sched = load_frozen_unet(device, ckpt)

    print(f"\n提取测试集特征 ({x_te.size(0)} 张)...")
    f_te = extract_features(unet, sched, x_te, layer, t, device, noise_reps=noise_reps)

    budgets = [("ipc100", balanced_subset(y_tr, 100)), ("full", np.arange(y_tr.numel()))]
    if no_full_budget:      # 49k 全量预算极慢且不参与对比图, 可跳过
        budgets = budgets[:1]
    out = {}
    for tag, idx in budgets:
        print(f"\n提取训练集特征 [{tag}] ({len(idx)} 张)...")
        f_tr = extract_features(unet, sched, x_tr[idx], layer, t, device,
                                noise_reps=noise_reps)
        out[tag] = {"points": int(len(idx)), "results": {}}
        for name in BACKBONES:
            accs = eval_features(name, f_tr, y_tr[idx], f_te, y_te, device,
                                 seeds=seeds, epochs=epochs)
            params = count_params(make_backbone(name, f_tr.shape[-1])(
                in_ch=f_tr.size(1), num_classes=C))
            out[tag]["results"][name] = {
                "acc": float(np.mean(accs)), "std": float(np.std(accs)),
                "seeds_acc": [float(a) for a in accs], "params": params}
            print(f"  {tag:7s} {name:<10s} {np.mean(accs):>6.2f}±{np.std(accs):.2f}%  "
                  f"({params:,} params)")

    save_json(out, {
        "exp": "frozen pretrained diffusion UNet features — NON-DEPLOYABLE upper bound",
        "model": ckpt or MODEL_ID, "model_params": sum(p.numel() for p in unet.parameters()),
        "layer": layer, "t": t, "seeds": seeds, "epochs": epochs,
        "lr": 0.001, "lr_schedule": "cosine", "noise_reps": noise_reps,
        "note": "backbone 以 in_ch=256 直接接收扩散特征, 参数量远大于主实验的 16ch 潜变量版本; "
                "本结果不可部署, 仅作上界参照"},
        f"diffusion_upperbound{out_tag}.json")
    return out


# ============================================================
# 论文用对比图
# ============================================================

def plot_compare(upper, path, diff_label="Diffusion upper bound (1k real)"):
    """分组柱状图: 每个 backbone 一组 (edge蒸馏 / 扩散上界@1k)

    柱顶标 'mean±std', 无误差棒 (论文格式约定)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_edge = load_baseline("backbone_mean_ipc100_d16_edge.json")
    if base_edge is None:
        print("无基线, 跳过对比图")
        return

    # 论文图表: 4 个 backbone, 去掉全量数据集上界 (与主实验 IPC=100 口径对齐)
    names = ["vgg", "resnet", "mobilenet", "vit"]
    series = {}
    series["Distilled (ours)"] = \
        [(base_edge[n][0], base_edge[n][1]) for n in names]
    series[diff_label] = \
        [(upper["ipc100"]["results"][n]["acc"], upper["ipc100"]["results"][n]["std"]) for n in names]

    n_s = len(series)
    x = np.arange(len(names))
    w = 0.8 / n_s
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for si, (sname, vals) in enumerate(series.items()):
        off = (si - (n_s - 1) / 2) * w
        means = [v[0] for v in vals]
        ax.bar(x + off, means, w, label=sname)
        for xi, (m, s) in zip(x + off, vals):
            ax.text(xi, m + 0.5, f"{m:.2f}±{s:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"对比图已保存: {path}")


def run_compare(args, parser):
    """子命令 compare 的主体"""
    if not (args.sweep or args.full):
        parser.error("至少指定 --sweep 或 --full")

    out_tag = args.tag if args.tag is not None else ("_selfddpm" if args.ckpt else "")
    is_self = bool(args.ckpt)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(未设置)')} | "
          f"模型: {args.ckpt or MODEL_ID}")

    if args.sweep:
        run_sweep(device, noise_reps=args.noise_reps, ckpt=args.ckpt, out_tag=out_tag)

    if args.full:
        layer, t = args.layer, args.t
        if layer is None or t is None:
            p = os.path.join(config.RESULT_DIR, f"diffusion_sweep{out_tag}.json")
            if not os.path.exists(p):
                parser.error(f"未找到 diffusion_sweep{out_tag}.json, "
                             f"请先跑 --sweep 或显式指定 --layer/--t")
            best = json.load(open(p))["meta"]["best"]
            layer = layer or best["layer"]
            t = best["t"] if t is None else t
            print(f"采用扫描最优配置: {layer} (t={t})")
        upper = run_full(device, LAYER_CHOICES[layer], t, seeds=args.seeds,
                         epochs=args.epochs, noise_reps=args.noise_reps,
                         ckpt=args.ckpt, out_tag=out_tag,
                         no_full_budget=args.no_full_budget)
        diff_label = "DDPM" if is_self else "Diffusion upper bound (1k real)"
        plot_compare(upper, os.path.join(config.RESULT_DIR,
                                         f"diffusion_upperbound_compare{out_tag}.png"),
                     diff_label=diff_label)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扩散对照线: 自训 DDPM + 冻结特征下游对照")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="从零训练 DDPM (CIFAR-10), 供特征提取")
    p_tr.add_argument("--steps", type=int, default=100000)
    p_tr.add_argument("--batch_size", type=int, default=None, help="默认 config.BATCH_SIZE")
    p_tr.add_argument("--lr", type=float, default=2e-4)
    p_tr.add_argument("--ema_decay", type=float, default=0.9999)
    p_tr.add_argument("--save_every", type=int, default=10000)
    p_tr.add_argument("--sample_every", type=int, default=20000, help="0 表示不抽检")
    p_tr.add_argument("--no_amp", action="store_true", help="关闭混合精度")
    p_tr.add_argument("--resume", type=str, default=None, help="从 ckpt 断点续训")
    p_tr.add_argument("--seed", type=int, default=0)

    p_cmp = sub.add_parser("compare", help="冻结扩散 UNet 特征上界/对照")
    p_cmp.add_argument("--sweep", action="store_true", help="阶段1: 扫 中间层 × timestep")
    p_cmp.add_argument("--full", action="store_true", help="阶段2: 完整对比")
    p_cmp.add_argument("--layer", type=str, default=None, choices=list(LAYER_CHOICES),
                       help="阶段2 指定层 (默认读 diffusion_sweep.json 的最优)")
    p_cmp.add_argument("--t", type=int, default=None, help="阶段2 指定 timestep")
    p_cmp.add_argument("--seeds", type=int, default=6)
    p_cmp.add_argument("--epochs", type=int, default=50)
    p_cmp.add_argument("--noise_reps", type=int, default=4)
    p_cmp.add_argument("--ckpt", type=str, default=None,
                       help="用自己训练的 DDPM 权重 (train 子命令产物); 默认 HF 官方模型")
    p_cmp.add_argument("--tag", type=str, default=None,
                       help="产物文件名后缀 (默认: 用 --ckpt 时为 _selfddpm, 否则无后缀)")
    p_cmp.add_argument("--no_full_budget", action="store_true",
                       help="跳过 49k 全量预算 (太慢, 且不参与对比图), 只跑 1k 预算")

    args = parser.parse_args()
    if args.cmd == "train":
        train(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
              ema_decay=args.ema_decay, save_every=args.save_every,
              sample_every=args.sample_every, amp=not args.no_amp,
              resume=args.resume, seed=args.seed)
    else:
        run_compare(args, p_cmp)
