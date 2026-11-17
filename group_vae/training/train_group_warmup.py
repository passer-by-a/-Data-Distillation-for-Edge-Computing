"""Stage 1: Group module warm-up（VAE 冻结）"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from group_vae.config_utils import ensure_dirs, load_config, result_path
from group_vae.data.group_sampler import (ClassBatchSampler, GroupDataset)
from group_vae.models.group_vae import GroupVAE
from group_vae.training.common import (build_train_groups, freeze_vae,
                                       group_params, load_norm, load_vae,
                                       subset_groups)
from group_vae.training.losses import group_dm_loss, group_kl, variance_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--fusion", default=None, help="mean | attention | film")
    ap.add_argument("--group_size", type=int, default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda_dm", type=float, default=None)
    ap.add_argument("--lambda_var", type=float, default=None)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ensure_dirs(cfg)
    fusion = a.fusion or cfg["group"]["fusion"]
    strategy = a.strategy or cfg["group"]["strategy"]
    tag = a.tag or f"warmup_{fusion}_{cfg['dataset']}_r{a.ratio:g}"
    if a.epochs:
        cfg["warmup"]["epochs"] = a.epochs
    if a.lambda_dm is not None:
        cfg["warmup"]["lambda_dm"] = a.lambda_dm
    if a.lambda_var is not None:
        cfg["warmup"]["lambda_var"] = a.lambda_var

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(cfg, dev)
    norm_mean, norm_std = load_norm(cfg)
    model = GroupVAE(vae, cfg["num_classes"], cfg["group"]).to(dev)
    model.set_feature_norm(norm_mean, norm_std)
    freeze_vae(model)
    if a.resume:
        sd = torch.load(a.resume, map_location=dev, weights_only=False)["group"]
        model.load_state_dict(sd, strict=False)
        print(f"resume group: {a.resume}")

    d, groups = build_train_groups(cfg, a.ratio, a.group_size, strategy,
                                   cfg["split_seed"])
    rng = np.random.RandomState(cfg["split_seed"])
    perm = rng.permutation(len(groups["labels"]))
    n_hold = max(64, int(len(perm) * cfg["warmup"].get("holdout_frac", 0.05)))
    hold, tr = perm[:n_hold], perm[n_hold:]
    g_tr, g_ho = subset_groups(groups, tr), subset_groups(groups, hold)
    ds_tr = GroupDataset(d["train"][0], g_tr)
    ds_ho = GroupDataset(d["train"][0], g_ho)
    sampler = ClassBatchSampler(g_tr["labels"],
                                cfg["warmup"]["batch_classes"],
                                cfg["warmup"]["batch_groups_per_class"],
                                seed=cfg["split_seed"])
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, num_workers=0)
    dl_ho = DataLoader(ds_ho, batch_size=32, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(group_params(model),
                            lr=cfg["warmup"]["lr"],
                            weight_decay=cfg["warmup"]["weight_decay"])
    epochs = int(cfg["warmup"]["epochs"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, len(sampler)))
    w = cfg["warmup"]
    best_ce, best_sd = float("inf"), None
    t0 = time.time()
    peak = 0.0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for ep in range(1, epochs + 1):
        model.train()
        model.vae.eval()
        sampler.set_epoch(ep)
        agg = {"loss": 0.0, "dm_mean": 0.0, "dm_var": 0.0, "ce": 0.0,
               "kl": 0.0, "var_ratio": 0.0}
        n = 0
        for images, mask, labels in dl_tr:
            images = images.to(dev)
            mask = mask.to(dev)
            labels = labels.to(dev)
            out = model.forward_group(images, mask, with_single_ref=True)
            dm, st = group_dm_loss(out["group_mu"], out["single_mu"].detach(),
                                   labels, mask, w["lambda_var"])
            ce = torch.nn.functional.cross_entropy(out["group_logits"], labels)
            kl = group_kl(out["group_mu_raw"], out["group_logvar"])
            loss = w["lambda_dm"] * dm + w["lambda_gce"] * ce + w["lambda_gkl"] * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(group_params(model), 1.0)
            opt.step()
            sch.step()
            agg["loss"] += float(loss.detach())
            agg["dm_mean"] += st["dm_mean"]
            agg["dm_var"] += st["dm_var"]
            agg["ce"] += float(ce.detach())
            agg["kl"] += float(kl.detach())
            agg["var_ratio"] += variance_ratio(out["group_mu"].detach(),
                                               out["single_mu"], labels, mask)
            n += 1
            if dev.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(dev) / 1e9)
        model.eval()
        ce_ho = 0.0
        with torch.no_grad():
            for images, mask, labels in dl_ho:
                out = model.forward_group(images.to(dev), mask.to(dev),
                                          with_single_ref=False)
                ce_ho += float(torch.nn.functional.cross_entropy(
                    out["group_logits"], labels.to(dev)))
        ce_ho /= max(1, len(dl_ho))
        if ce_ho < best_ce:
            best_ce = ce_ho
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()
                       if not k.startswith("vae.")}
        if ep % 1 == 0:
            print(f"ep {ep:>2}/{epochs} loss {agg['loss']/n:.4f} "
                  f"dm {agg['dm_mean']/n:.4f}/{agg['dm_var']/n:.4f} "
                  f"ce {agg['ce']/n:.3f} kl {agg['kl']/n:.4f} "
                  f"vr {agg['var_ratio']/n:.3f} | hold CE {ce_ho:.3f}", flush=True)

    meta = {"dataset": cfg["dataset"], "fusion": fusion, "stage": cfg["group"]["fusion_stage"],
            "ratio": a.ratio, "group_size": int(g_tr["group_size"]),
            "n_groups": int(len(groups["labels"])), "strategy": strategy,
            "epochs": epochs, "best_hold_ce": best_ce,
            "train_sec": time.time() - t0, "peak_gpu_gb": peak,
            "vae_ckpt": os.path.relpath(cfg["vae_ckpt"], cfg["_path"].rsplit("/", 3)[0]),
            "protocol_version": cfg.get("protocol_version")}
    ck = {"group": best_sd, "meta": meta, "config": cfg}
    p = result_path(cfg, tag + ".pt")
    torch.save(ck, p)
    with open(result_path(cfg, tag + ".json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
