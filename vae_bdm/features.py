"""VAE 潜特征提取、缓存与加载

协议与现有 90% 实验一致:
  图像 [-1,1] → EdgeVAE.encode_mu → [N,16,8,8]
  → 训练集逐通道 z-score → flatten [N,1024]
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

from common import DEV, seed_all
from data import load_split
from models import VAEBackbone
from train import channel_stats

from .config_utils import cache_dir

FILES = ("train_features.npy", "train_labels.npy",
         "val_features.npy", "val_labels.npy",
         "test_features.npy", "test_labels.npy")


def _sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def _encode(mu_net, x, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i + batch].to(DEV).float().div(127.5).sub(1)
            out.append(mu_net(xb).float().cpu())
    return torch.cat(out, 0)


def _normalize(raw, mean, std):
    return ((raw - mean.to(raw.device)) / std.to(raw.device)).flatten(1)


def edge_cost(model, device=None, warmup=10, runs=50):
    """编码器参数量 / 单图 FLOPs / 延迟"""
    device = device or DEV
    enc = model.ae.encoder
    x = torch.randn(1, 3, 32, 32, device=device)
    flops = None
    try:
        from thop import profile
        macs, _ = profile(enc, inputs=(x,), verbose=False)
        flops = macs * 2 / 1e6          # MFLOPs
    except Exception:
        pass
    for _ in range(warmup):
        enc(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        enc(x)
    torch.cuda.synchronize()
    return {"encoder_params": sum(p.numel() for p in enc.parameters()),
            "encoder_flops_m": flops,
            "encoder_latency_ms": (time.time() - t0) / runs * 1000}


def extract_and_cache(cfg, force=False):
    d = cache_dir(cfg)
    meta_path = os.path.join(d, "metadata.json")
    if all(os.path.exists(os.path.join(d, f)) for f in FILES) and not force:
        print(f"特征缓存已存在: {d}")
        return d

    seed_all(cfg["split_seed"])
    t0 = time.time()
    split = load_split(cfg["dataset"], val_ratio=cfg["val_ratio"],
                       seed=cfg["split_seed"])
    bb = VAEBackbone(cfg["num_classes"], DEV, cfg["vae_ckpt"])
    print(f"VAE: {os.path.basename(cfg['vae_ckpt'])} | {bb.n_params:,} 参数 | "
          f"latent=mu", flush=True)

    raw = {}
    for s in ("train", "val", "test"):
        t = time.time()
        raw[s] = _encode(bb.ae.encode_mu, split[s][0], cfg["feature"]["batch"])
        print(f"  {s}: {tuple(raw[s].shape)} | {time.time()-t:.1f}s", flush=True)
    mean, std = channel_stats(raw["train"])
    feats = {s: _normalize(raw[s], mean, std) for s in raw}
    torch.save({"mean": mean.cpu(), "std": std.cpu()},
               os.path.join(d, "norm_stats.pt"))

    np.save(os.path.join(d, "train_features.npy"), feats["train"].numpy())
    np.save(os.path.join(d, "train_labels.npy"), split["train"][1].numpy())
    np.save(os.path.join(d, "val_features.npy"), feats["val"].numpy())
    np.save(os.path.join(d, "val_labels.npy"), split["val"][1].numpy())
    np.save(os.path.join(d, "test_features.npy"), feats["test"].numpy())
    np.save(os.path.join(d, "test_labels.npy"), split["test"][1].numpy())

    cost = edge_cost(bb)
    n = 50000
    perm = np.random.RandomState(cfg["split_seed"]).permutation(n)
    n_val = int(n * cfg["val_ratio"])
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    assert len(set(tr_idx.tolist()) & set(va_idx.tolist())) == 0
    shot = int(cfg.get("vae_label_shot", 0))
    n_train = int(len(split["train"][1]))
    if shot:
        note = (f"VAE 重建+KL 使用全部训练图像；分类头仅用 {shot}/类"
                f"（共 {shot * cfg['num_classes']} 个）标签训练")
        labels_used = shot * cfg["num_classes"]
    else:
        note = "VAE 预训练使用重建+KL+全部训练标签；本特征不是无监督特征"
        labels_used = n_train
    meta = {
        "dataset": cfg["dataset"], "vae_ckpt": os.path.relpath(cfg["vae_ckpt"], os.path.dirname(os.path.dirname(__file__))),
        "latent": "posterior_mu", "feature_dim": 1024,
        "feature_shape": [16, 8, 8],
        "input_norm": "[-1,1]",
        "feature_norm": "per-channel z-score fitted on train only",
        "split_seed": cfg["split_seed"], "val_ratio": cfg["val_ratio"],
        "n_train": n_train,
        "n_val": int(len(split["val"][1])),
        "n_test": int(len(split["test"][1])),
        "train_index_sha": _sha(tr_idx), "val_index_sha": _sha(va_idx),
        "supervised_pretrain": True,
        "vae_label_shot": shot,
        "vae_labels_used": labels_used,
        "supervised_note": note,
        "feature_time_sec": time.time() - t0,
        **cost,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已保存: {d}\nmeta: {meta}")
    return d


def load_features(cfg):
    d = cache_dir(cfg)
    out = {}
    for name in ("train", "val", "test"):
        out[name] = {
            "x": torch.from_numpy(np.load(os.path.join(d, f"{name}_features.npy"))).float(),
            "y": torch.from_numpy(np.load(os.path.join(d, f"{name}_labels.npy"))).long(),
        }
    with open(os.path.join(d, "metadata.json")) as f:
        out["meta"] = json.load(f)
    return out


def extract_vae_features(images, cfg):
    """[B,3,32,32] 原始图像（[-1,1]）→ [B,1024]"""
    bb = VAEBackbone(cfg["num_classes"], DEV, cfg["vae_ckpt"])
    d = cache_dir(cfg)
    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)
    raw = _encode(bb.ae.encode_mu, images, cfg["feature"]["batch"])
    # 复用训练集统计量: 从 npy 训练特征反推不可行，直接重新计算并缓存
    # 首次调用时由 extract_and_cache 保证 metadata 与 npy 一致；此处用
    # 训练集 raw 统计量在 cache 中保存的归一化参数。
    stats_path = os.path.join(d, "norm_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError("缺少 norm_stats.pt，请先运行 extract_features")
    st = torch.load(stats_path, map_location="cpu")
    return _normalize(raw, st["mean"], st["std"])
