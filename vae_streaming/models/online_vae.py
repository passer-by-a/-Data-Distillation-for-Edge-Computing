"""Online VAE: 只在 Initial Base Set 初始化，之后按轮更新

损失与第一阶段一致:
    L = MSE(recon, x) + beta * KL + lambda_c * CE
在线阶段可选:
    + lambda_s * || E_t(x_replay) - sg(E_{t-1}(x_replay)) ||^2

编码器始终更新；上一轮快照只作稳定性参考，不接收梯度。
"""
import copy
import time

import torch
import torch.nn.functional as F

from vae import EdgeVAE


def to_float(images_u8, device):
    """uint8 [0,255] → float [-1,1]"""
    return images_u8.to(device).float().div(127.5).sub(1.0)


def build_vae(num_classes, latent_dim=16, base_channels=64, seed=None,
              device="cuda"):
    if seed is not None:
        torch.manual_seed(seed)
    vae = EdgeVAE(latent_dim=latent_dim, num_classes=num_classes,
                  base_channels=base_channels).to(device)
    return vae


# ======================================================================
# 冻结快照
# ======================================================================
class FrozenEncoder:
    """上一轮 Encoder 的深拷贝快照（eval 模式，无梯度）"""

    def __init__(self, encoder, version):
        self.version = int(version)
        self.module = copy.deepcopy(encoder).eval()
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_mu(self, x):
        return self.module(x)[0]

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.module.state_dict().items()}


def snapshot_encoder(vae, version):
    return FrozenEncoder(vae.encoder, version)


# ======================================================================
# 特征归一化协议（因果、冻结）
# ======================================================================
class FeatureProtocol:
    """逐通道 z-score，统计量只在 Initial Base Set 上拟合一次后冻结。

    冻结归一化是边缘与云端共享的固定线性变换，编码器更新不会破坏
    两侧一致性；这也避免使用未来数据重估归一化统计量。
    """

    def __init__(self, mean, std, fitted_on="initial_base"):
        self.mean = mean.float()          # [1,C,1,1]
        self.std = std.float().clamp_min(1e-6)
        self.fitted_on = fitted_on

    @classmethod
    def fit(cls, raw):
        """raw: [N,C,H,W] float"""
        c = raw.size(1)
        s = torch.zeros(c, dtype=torch.float64)
        ss = torch.zeros(c, dtype=torch.float64)
        n = 0
        for i in range(0, raw.size(0), 512):
            f = raw[i:i + 512].float().double()
            s += f.sum(dim=(0, 2, 3))
            ss += f.pow(2).sum(dim=(0, 2, 3))
            n += f.size(0) * f.size(2) * f.size(3)
        m = s / n
        v = (ss / n - m.pow(2)).clamp_min(1e-12)
        return cls(m.float().view(1, -1, 1, 1),
                   v.sqrt().float().view(1, -1, 1, 1))

    def normalize(self, raw):
        f = (raw.float().cpu() - self.mean) / self.std
        return f.flatten(1)

    def meta(self):
        return {"fitted_on": self.fitted_on, "shape": list(self.mean.shape),
                "mean": self.mean.flatten().tolist(),
                "std": self.std.flatten().tolist()}


@torch.no_grad()
def encode_mu(encoder, images_u8, batch=256, device="cuda"):
    """原始图像 → 未归一化 mu [N,C,H,W] (CPU float32)

    encoder 可以是 EdgeEncoder（forward）或 FrozenEncoder（encode_mu）。
    """
    if hasattr(encoder, "eval"):
        encoder.eval()
    out = []
    for i in range(0, images_u8.size(0), batch):
        x = to_float(images_u8[i:i + batch], device)
        mu = encoder.encode_mu(x) if hasattr(encoder, "encode_mu") \
            else encoder(x)[0]
        out.append(mu.float().cpu())
    return torch.cat(out, 0) if out else torch.empty(0)


@torch.no_grad()
def encode_features(encoder, images_u8, protocol, batch=256, device="cuda"):
    """原始图像 → 冻结协议下的 1024 维特征 [N,1024] (CPU float32)"""
    return protocol.normalize(encode_mu(encoder, images_u8, batch, device))


# ======================================================================
# 数据增强（与第一阶段一致的 flip + crop，仅训练时）
# ======================================================================
def _augment(x):
    from torchvision.transforms import v2
    if x.size(0) == 0:
        return x
    x = v2.RandomHorizontalFlip(p=0.5)(x)
    x = v2.RandomCrop(32, padding=4, padding_mode="reflect")(x)
    return x


# ======================================================================
# 损失
# ======================================================================
def vae_losses(vae, x, y, beta, cls_lambda):
    recon, z, logits, mu, logvar = vae(x)
    rec = F.mse_loss(recon, x)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    ce = F.cross_entropy(logits, y)
    loss = rec + beta * kl + cls_lambda * ce
    return loss, {"rec": rec.detach(), "kl": kl.detach(), "ce": ce.detach()}


# ======================================================================
# 初始训练（只使用 Initial Base Set）
# ======================================================================
def train_initial_vae(vae, images, labels, cfg, device="cuda", verbose=True):
    vcfg = cfg["vae"]
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=float(vcfg["lr"]))
    bs = int(vcfg["batch_size"])
    n = len(labels)
    log = []
    t0 = time.time()
    for ep in range(1, int(vcfg["initial_epochs"]) + 1):
        perm = torch.randperm(n)
        agg = {"rec": 0.0, "kl": 0.0, "ce": 0.0, "loss": 0.0}
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            x = to_float(images[idx], device)
            y = labels[idx].to(device)
            if vcfg.get("augment", True):
                x = _augment(x)
            loss, parts = vae_losses(vae, x, y, float(vcfg["beta"]),
                                     float(vcfg["cls_lambda"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg["loss"] += float(loss.detach())
            for k, v in parts.items():
                agg[k] += float(v)
            nb += 1
        row = {k: v / max(1, nb) for k, v in agg.items()}
        row["epoch"] = ep
        log.append(row)
        if verbose and (ep % 10 == 0 or ep == int(vcfg["initial_epochs"])):
            print(f"  [base VAE] ep {ep:3d}/{vcfg['initial_epochs']} "
                  f"loss {row['loss']:.4f} rec {row['rec']:.4f} "
                  f"kl {row['kl']:.4f} ce {row['ce']:.4f}", flush=True)
    return {"log": log, "train_sec": time.time() - t0,
            "steps": int(vcfg["initial_epochs"]) * max(1, (n + bs - 1) // bs),
            "n_images": int(n), "accessible": "initial_base_only"}


# ======================================================================
# 在线一轮更新
# ======================================================================
def online_vae_update(vae, images, labels, replay_images, replay_labels,
                      prev_encoder, cfg, device="cuda", round_idx=0):
    """固定 optimization steps，当前数据 + replay 组成每个 batch。"""
    vcfg = cfg["vae"]
    if not bool(vcfg.get("online_update", True)):
        return {"skipped": True, "steps": 0, "train_sec": 0.0,
                "accessible": f"round<= {round_idx}"}

    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=float(vcfg["lr"]))
    bs = int(vcfg["batch_size"])
    steps = int(vcfg["steps_per_round"])
    replay_ratio = float(vcfg.get("replay_ratio", 0.0)) \
        if bool(vcfg.get("replay_enabled", True)) else 0.0
    n_rep = int(round(bs * replay_ratio)) if replay_images is not None \
        and len(replay_images) > 0 else 0
    n_rep = min(n_rep, bs - 1)
    n_cur = bs - n_rep
    stability = bool(vcfg.get("stability_enabled", False)) \
        and prev_encoder is not None and n_rep > 0
    lam_s = float(vcfg.get("lambda_stab", 0.0)) if stability else 0.0

    g = torch.Generator().manual_seed(int(vcfg.get("seed", 42)) + round_idx)
    agg = {"rec": 0.0, "kl": 0.0, "ce": 0.0, "stab": 0.0, "loss": 0.0}
    t0 = time.time()

    for _ in range(steps):
        ic = torch.randint(0, len(labels), (n_cur,), generator=g)
        xc = to_float(images[ic], device)
        yc = labels[ic].to(device)
        if n_rep > 0:
            ir = torch.randint(0, len(replay_labels), (n_rep,), generator=g)
            xr = to_float(replay_images[ir], device)
            yr = replay_labels[ir].to(device)
            x = torch.cat([xc, xr], 0)
            y = torch.cat([yc, yr], 0)
        else:
            x, y = xc, yc
        if vcfg.get("augment", True):
            x = _augment(x)

        loss, parts = vae_losses(vae, x, y, float(vcfg["beta"]),
                                 float(vcfg["cls_lambda"]))
        stab = torch.zeros((), device=device)
        if stability:
            mu_cur = vae.encode_mu(x[-n_rep:])
            mu_prev = prev_encoder.encode_mu(x[-n_rep:])
            stab = F.mse_loss(mu_cur, mu_prev.detach())
            loss = loss + lam_s * stab
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        agg["loss"] += float(loss.detach())
        agg["stab"] += float(stab.detach()) if stability else 0.0
        for k, v in parts.items():
            agg[k] += float(v)

    row = {k: v / max(1, steps) for k, v in agg.items()}
    row.update({"steps": steps, "train_sec": time.time() - t0,
                "replay_ratio": float(n_rep / max(1, bs)),
                "n_replay_in_batch": int(n_rep),
                "accessible": f"round<= {round_idx}"})
    return row


# ======================================================================
# 诊断
# ======================================================================
@torch.no_grad()
def evaluate_vae(vae, images, labels, cfg, batch=256, device="cuda"):
    vae.eval()
    rec_all, kl_all, correct = 0.0, 0.0, 0
    n = 0
    for i in range(0, len(labels), batch):
        x = to_float(images[i:i + batch], device)
        y = labels[i:i + batch].to(device)
        recon, z, logits, mu, logvar = vae(x)
        rec_all += float(F.mse_loss(recon, x, reduction="sum"))
        kl_all += float((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
                        .sum())
        correct += int((logits.argmax(1) == y).sum())
        n += x.size(0)
    return {"rec": rec_all / max(1, n), "kl": kl_all / max(1, n),
            "cls_acc": 100.0 * correct / max(1, n), "n": int(n)}


def encoder_delta(vae, prev_encoder):
    """当前 Encoder 参数相对上一轮快照的 L2 变化"""
    total, count = 0.0, 0
    for (name, p), (_, q) in zip(vae.encoder.named_parameters(),
                                 prev_encoder.state_dict().items()):
        total += float((p.detach().cpu() - q).pow(2).sum())
        count += p.numel()
    return float((total / max(1, count)) ** 0.5)
