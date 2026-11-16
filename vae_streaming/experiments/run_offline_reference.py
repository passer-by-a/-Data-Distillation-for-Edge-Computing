"""非因果离线参考: 全量 49k 特征上的 Batch DM / K-means / Herding

该脚本显式违反 Streaming 因果约束，只用于估计在线处理造成的性能差距，
不可与在线方法当作同等数据可访问条件比较。

python -m vae_streaming.experiments.run_offline_reference \
    --config vae_streaming/configs/cifar100.yaml \
    --vae-ckpt vae_streaming/checkpoints/cifar100/iid/seed42/full_method/vae_v5.pt \
    --protocol-from vae_streaming/results/cifar100/iid/full_method_seed42.json \
    --method bdm
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vae import EdgeVAE  # noqa: E402
from vae_bdm.baselines import allocate_quotas, build_baseline  # noqa: E402
from vae_bdm.bdm import BatchDM  # noqa: E402
from vae_bdm.cloud import (evaluate_cloud_classifier,  # noqa: E402
                           train_cloud_classifier)
from vae_streaming.config_utils import load_config, result_path  # noqa: E402
from vae_streaming.data.causal_stream import CausalStream  # noqa: E402
from vae_streaming.models.online_vae import (FeatureProtocol,  # noqa: E402
                                             encode_mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vae_streaming/configs/cifar100.yaml")
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--protocol-from", default=None,
                    help="共享归一化协议的 JSON；缺省则在全量训练集上重估（非因果）")
    ap.add_argument("--method", default="bdm",
                    choices=["random", "kmeans", "herding", "bdm"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C = int(cfg["num_classes"])
    dim = int(cfg["feature_dim"])
    M = int(cfg["memory"]["representative_budget"])
    s = cfg["stream"]
    stream = CausalStream(dataset=cfg["dataset"],
                          initial_fraction=float(s["initial_fraction"]),
                          num_rounds=int(s["num_rounds"]),
                          seed=int(s["shuffle_seed"]),
                          scenario="iid",
                          val_ratio=float(cfg["val_ratio"]),
                          split_seed=int(cfg["split_seed"]))
    t0 = time.time()

    vae = EdgeVAE(latent_dim=int(cfg["latent_dim"]), num_classes=C,
                  base_channels=int(cfg["base_channels"])).to(device)
    ck = torch.load(args.vae_ckpt, map_location=device, weights_only=True)
    vae.load_state_dict(ck["model_state_dict"])
    vae.eval()

    tr_raw = encode_mu(vae.encoder, stream.train_images, device=device)
    va_raw = encode_mu(vae.encoder, stream.val_images, device=device)
    te_raw = encode_mu(vae.encoder, stream.test_images, device=device)
    if args.protocol_from:
        obj = json.load(open(args.protocol_from))
        proto = obj["rounds"][0]["feat_norm_protocol"]
        mean = torch.tensor(proto["mean"]).view(1, -1, 1, 1)
        std = torch.tensor(proto["std"]).view(1, -1, 1, 1)
        protocol = FeatureProtocol(mean, std, fitted_on="shared_with_online")
        protocol_source = "shared_initial_base"
    else:
        protocol = FeatureProtocol.fit(tr_raw)
        protocol_source = "full_train_set_non_causal"
    tr = protocol.normalize(tr_raw)
    va = protocol.normalize(va_raw)
    te = protocol.normalize(te_raw)

    quotas = allocate_quotas(stream.train_labels.numpy(), M, C).tolist()

    if args.method == "bdm":
        dcfg = copy.deepcopy(cfg["distillation"])
        dcfg["feature_dim"] = dim
        dcfg.setdefault("updates_per_class", 20)
        dcfg["selection"] = "final"
        dm = BatchDM(tr, stream.train_labels, quotas, C, dcfg,
                     seed=int(cfg["distillation"].get("seed", 42)),
                     init="kmeans", device=device, val=None)
        out = dm.fit(verbose=False)
        reps, rep_y = out["features"], out["labels"]
        cost = {"distill_sec": out["train_sec"] + out["init_sec"]}
    else:
        bcfg = {"kmeans_n_init": 1, "kmeans_max_iter": 100,
                "moment_eps": 1e-6, "moment_max_scale": 3.0}
        reps, rep_y, info = build_baseline(tr, stream.train_labels, 49000 / M,
                                           args.method, C, bcfg, seed=42)
        cost = {"distill_sec": info["distill_sec"]}

    steps = int(cfg["cloud"]["steps_per_round"]) * (int(s["num_rounds"]) + 1)
    accs, f1s, infos = [], [], []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        model, info = train_cloud_classifier(
            reps, rep_y, head=cfg["cloud"]["head"], num_classes=C, seed=seed,
            epochs=cfg["cloud"]["epochs"], lr=cfg["cloud"]["lr"],
            wd=cfg["cloud"]["wd"], batch=cfg["cloud"]["batch"], device=device,
            val_features=va, val_labels=stream.val_labels,
            budget_mode="fixed_steps", steps=steps)
        m = evaluate_cloud_classifier(model, te, stream.test_labels, C)
        accs.append(m["acc"])
        f1s.append(m["f1"])
        infos.append(info)

    accessible = ck.get("accessible_rounds") or ck.get("accessible") \
        or "full_49k_offline"
    res = {
        "summary": {
            "method": args.method, "dataset": cfg["dataset"],
            "tag": args.tag or f"offline_{args.method}",
            "scenario": "offline_non_causal", "seed": None,
            "causal": False,
            "vae_ckpt": os.path.relpath(args.vae_ckpt, ROOT),
            "vae_accessible": accessible,
            "protocol_source": protocol_source,
            "n_train_used": int(len(tr)), "n_reps": int(len(reps)),
            "initial_acc": None, "final_acc": float(np.mean(accs)),
            "final_acc_std": float(np.std(accs)),
            "avg_online_acc": float(np.mean(accs)),
            "final_f1": float(np.mean(f1s)),
            "final_f1_std": float(np.std(f1s)),
            "avg_online_f1": float(np.mean(f1s)),
            "final_test_int8_acc": None,
            "initial_upload_bytes": None,
            "cumulative_upload_bytes": int(len(reps) * dim * 1 +
                                           len(reps) * 2),
            "sync_mode": "one_shot", "quantization": "int8",
            "n_reps_budget": M,
            "cloud_steps": steps, "cloud_head": cfg["cloud"]["head"],
            "cloud_seeds": [int(x) for x in args.seeds.split(",") if x.strip()],
            "edge_total_sec": time.time() - t0,
            "cloud_total_sec": float(sum(i["train_sec"] for i in infos)),
            **cost,
        },
        "quotas": quotas,
        "cloud_infos": infos,
    }
    tag = args.tag or f"offline_{args.method}"
    p = result_path(cfg, f"{cfg['dataset']}/offline/{tag}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
