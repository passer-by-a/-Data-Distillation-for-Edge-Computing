"""把训练好的 Group-VAE 编码成 [M,1024] 代表特征，并导出单图 val/test 特征"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from group_vae.config_utils import ROOT, ensure_dirs, load_config, result_path
from group_vae.data.group_sampler import GroupDataset
from group_vae.models.group_vae import GroupVAE
from group_vae.training.common import (build_train_groups, load_norm, load_vae)


@torch.no_grad()
def encode_groups(model, loader, dev):
    feats, labels = [], []
    for images, mask, y in loader:
        f = model.encode_group(images.to(dev), mask.to(dev))
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def encode_singles(model, images_u8, y, dev, batch=256):
    feats = []
    for i in range(0, len(images_u8), batch):
        x = images_u8[i:i + batch].to(dev).float().div(127.5).sub(1)
        feats.append(model.encode_single(x).cpu())
    return torch.cat(feats), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--group_size", type=int, default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ensure_dirs(cfg)
    fusion = a.fusion or cfg["group"]["fusion"]
    strategy = a.strategy or cfg["group"]["strategy"]
    tag = a.tag or f"group_{fusion}_{cfg['dataset']}_r{a.ratio:g}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(cfg, dev)
    model = GroupVAE(vae, cfg["num_classes"], cfg["group"]).to(dev)
    mean, std = load_norm(cfg)
    model.set_feature_norm(mean, std)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = ck.get("group") or ck.get("model")
    model.load_state_dict(sd, strict=False)
    model.eval()

    d, groups = build_train_groups(cfg, a.ratio, a.group_size, strategy,
                                   cfg["split_seed"])
    ds = GroupDataset(d["train"][0], groups)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    t0 = time.time()
    g_feat, g_y = encode_groups(model, loader, dev)
    tr_feat, tr_y = encode_singles(model, d["train"][0], d["train"][1], dev)
    v_feat, v_y = encode_singles(model, d["val"][0], d["val"][1], dev)
    t_feat, t_y = encode_singles(model, d["test"][0], d["test"][1], dev)

    meta = {"dataset": cfg["dataset"], "fusion": fusion,
            "stage": cfg["group"]["fusion_stage"], "ratio": a.ratio,
            "group_size": int(groups["group_size"]),
            "n_groups": int(len(g_y)), "strategy": strategy,
            "ckpt": a.ckpt, "vae_ckpt": os.path.relpath(cfg["vae_ckpt"], ROOT),
            "protocol_version": cfg.get("protocol_version"),
            "encode_sec": time.time() - t0,
            "norm_mean": mean.flatten().tolist(), "norm_std": std.flatten().tolist()}
    p = result_path(cfg, tag + ".npz")
    np.savez_compressed(p, features=g_feat.numpy(), labels=g_y.numpy(),
                        indices=groups["indices"], mask=groups["mask"],
                        meta=json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(result_path(cfg, tag + "_train.npz"),
                        features=tr_feat.numpy(), labels=tr_y.numpy())
    np.savez_compressed(result_path(cfg, tag + "_val.npz"),
                        features=v_feat.numpy(), labels=v_y.numpy())
    np.savez_compressed(result_path(cfg, tag + "_test.npz"),
                        features=t_feat.numpy(), labels=t_y.numpy())
    with open(result_path(cfg, tag + ".json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"groups {tuple(g_feat.shape)} | val {tuple(v_feat.shape)} | "
          f"test {tuple(t_feat.shape)} | {meta['encode_sec']:.1f}s | {p}")


if __name__ == "__main__":
    main()
