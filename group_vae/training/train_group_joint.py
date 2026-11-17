"""Stage 2: Joint fine-tuning（单图 VAE + 组级路径）"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from group_vae.config_utils import ROOT, ensure_dirs, load_config, result_path
from group_vae.data.group_sampler import ClassBatchSampler, GroupDataset
from group_vae.models.group_vae import GroupVAE
from group_vae.training.common import (build_train_groups, load_norm, load_vae,
                                       subset_groups)
from group_vae.training.losses import (group_dm_loss, group_kl,
                                       single_vae_loss, variance_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--group_size", type=int, default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--init_group", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ensure_dirs(cfg)
    fusion = a.fusion or cfg["group"]["fusion"]
    strategy = a.strategy or cfg["group"]["strategy"]
    tag = a.tag or f"joint_{fusion}_{cfg['dataset']}_r{a.ratio:g}"
    if a.epochs:
        cfg["joint"]["epochs"] = a.epochs

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(cfg, dev)
    norm_mean, norm_std = load_norm(cfg)
    model = GroupVAE(vae, cfg["num_classes"], cfg["group"]).to(dev)
    model.set_feature_norm(norm_mean, norm_std)
    if a.init_group:
        sd = torch.load(a.init_group, map_location=dev, weights_only=False)["group"]
        model.load_state_dict(sd, strict=False)
        print(f"init group: {a.init_group}")

    d, groups = build_train_groups(cfg, a.ratio, a.group_size, strategy,
                                   cfg["split_seed"])
    ds = GroupDataset(d["train"][0], groups)
    sampler = ClassBatchSampler(groups["labels"],
                                cfg["joint"]["batch_classes"],
                                cfg["joint"]["batch_groups_per_class"],
                                seed=cfg["split_seed"])
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["joint"]["lr"])
    j = cfg["joint"]
    epochs = int(j["epochs"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, epochs * max(1, len(sampler)))
    t0 = time.time()
    peak = 0.0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for ep in range(1, epochs + 1):
        model.train()
        sampler.set_epoch(ep)
        agg = {k: 0.0 for k in ("loss", "rec", "kl", "cls", "dm_mean",
                                "dm_var", "gce", "gkl", "vr")}
        n = 0
        for images, mask, labels in dl:
            images = images.to(dev)
            mask = mask.to(dev)
            labels = labels.to(dev)
            out = model.forward_joint(images, mask)
            l_single, st = single_vae_loss(
                out["recon"], images, out["logits_single"], labels,
                out["mu_single"], out["logvar_single"], out["valid"],
                beta=j["lambda_kl"], cls_lambda=j["lambda_cls"])
            dm, dst = group_dm_loss(out["group_mu"],
                                    out["single_mu"].detach(), labels, mask,
                                    j["lambda_var"])
            gce = torch.nn.functional.cross_entropy(out["group_logits"], labels)
            gkl = group_kl(out["group_mu_raw"], out["group_logvar"])
            loss = (j["lambda_rec"] * l_single + j["lambda_dm"] * dm
                    + j["lambda_gce"] * gce + j["lambda_gkl"] * gkl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            agg["loss"] += float(loss.detach())
            agg["rec"] += st["rec"]; agg["kl"] += st["kl"]; agg["cls"] += st["cls"]
            agg["dm_mean"] += dst["dm_mean"]; agg["dm_var"] += dst["dm_var"]
            agg["gce"] += float(gce.detach()); agg["gkl"] += float(gkl.detach())
            agg["vr"] += variance_ratio(out["group_mu"].detach(),
                                        out["single_mu"], labels, mask)
            n += 1
            if dev.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(dev) / 1e9)
        print(f"ep {ep:>2}/{epochs} loss {agg['loss']/n:.4f} "
              f"rec {agg['rec']/n:.4f} cls {agg['cls']/n:.3f} "
              f"dm {agg['dm_mean']/n:.4f}/{agg['dm_var']/n:.4f} "
              f"gce {agg['gce']/n:.3f} vr {agg['vr']/n:.3f}", flush=True)

    meta = {"dataset": cfg["dataset"], "fusion": fusion,
            "stage": cfg["group"]["fusion_stage"], "ratio": a.ratio,
            "group_size": int(groups["group_size"]),
            "n_groups": int(len(groups["labels"])), "strategy": strategy,
            "epochs": epochs, "train_sec": time.time() - t0,
            "peak_gpu_gb": peak, "init_group": a.init_group,
            "vae_ckpt": os.path.relpath(cfg["vae_ckpt"], ROOT),
            "protocol_version": cfg.get("protocol_version")}
    p = result_path(cfg, tag + ".pt")
    torch.save({"model": model.state_dict(), "meta": meta, "config": cfg}, p)
    with open(result_path(cfg, tag + ".json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
