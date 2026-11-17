"""编码成本测量：单图 batch=10 vs 组级 K=10"""
import argparse
import json
import time

import torch

from group_vae.config_utils import load_config, result_path
from group_vae.models.group_vae import GroupVAE
from group_vae.training.common import load_norm, load_vae


@torch.no_grad()
def _time(fn, warmup=10, runs=50):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--fusion", default="film")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="cost")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(cfg, dev)
    model = GroupVAE(vae, cfg["num_classes"], cfg["group"]).to(dev)
    mean, std = load_norm(cfg)
    model.set_feature_norm(mean, std)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get("group") or ck.get("model"), strict=False)
    model.eval()
    x1 = torch.randn(10, 3, 32, 32, device=dev)
    xg = torch.randn(1, a.k, 3, 32, 32, device=dev)
    m1 = torch.ones(1, a.k, dtype=torch.bool, device=dev)
    t_single10 = _time(lambda: model.encode_single(x1))
    t_group = _time(lambda: model.encode_group(xg, m1))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        model.encode_group(xg, m1)
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
    else:
        peak = 0.0
    total = sum(p.numel() for p in model.parameters())
    vae_p = sum(p.numel() for p in model.vae.parameters())
    out = {"fusion": a.fusion, "k": a.k,
           "params_total": total, "params_vae": vae_p,
           "params_group": total - vae_p,
           "single_batch10_ms": t_single10, "group_k_ms": t_group,
           "speedup_vs_single10": t_single10 / max(1e-9, t_group),
           "peak_gpu_gb": peak, "device": str(dev)}
    p = result_path(cfg, a.tag + ".json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
