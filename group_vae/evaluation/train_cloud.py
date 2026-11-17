"""云端训练与评估 Group-VAE 代表特征（v2 external validation）"""
import argparse
import json
import os

import numpy as np
import torch

from group_vae.config_utils import ensure_dirs, load_config, result_path
from vae_bdm.cloud import evaluate_cloud_classifier, train_cloud_classifier
from vae_bdm.quant import Quantizer, payload_table, save_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--tag", required=True, help="encode_groups 的 tag")
    ap.add_argument("--modes", default="fp32,fp16,int8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ensure_dirs(cfg)
    R = cfg["results_dir"]
    g = np.load(os.path.join(R, a.tag + ".npz"), allow_pickle=True)
    v = np.load(os.path.join(R, a.tag + "_val.npz"))
    t = np.load(os.path.join(R, a.tag + "_test.npz"))
    reps = torch.from_numpy(g["features"]).float()
    labels = torch.from_numpy(g["labels"]).long()
    val_x = torch.from_numpy(v["features"]).float()
    val_y = torch.from_numpy(v["labels"]).long()
    test_x = torch.from_numpy(t["features"]).float()
    test_y = torch.from_numpy(t["labels"]).long()
    meta = json.loads(str(g["meta"]))
    cloud = cfg["cloud"]
    seeds = [int(s) for s in cloud.get("seeds", [0, 1, 2])]
    tb = cloud.get("training_budget", {})
    epochs = int(tb.get("epochs", cloud.get("epochs", 200)))
    out = {}
    payload_dir = os.path.join(R, "payload")
    for mode in [m for m in a.modes.split(",") if m]:
        q = Quantizer(mode).fit(reps)
        reps_dq = q.dequantize(q.quantize(reps))
        val_dq = q.dequantize(q.quantize(val_x))
        test_dq = q.dequantize(q.quantize(test_x))
        accs, f1s, infos = [], [], []
        for s in seeds:
            model, info = train_cloud_classifier(
                reps_dq, labels, head=cloud["head"],
                num_classes=cfg["num_classes"], seed=s, epochs=epochs,
                lr=cloud["lr"], wd=cloud["wd"], batch=cloud["batch"],
                device="cuda", val_features=val_dq, val_labels=val_y)
            m = evaluate_cloud_classifier(model, test_dq, test_y,
                                          cfg["num_classes"])
            accs.append(m["acc"]); f1s.append(m["f1"]); infos.append(info)
        sizes = save_payload(reps, labels, q, payload_dir,
                             f"{cfg['dataset']}_{a.tag}_{mode}")
        out[mode] = {"acc_mean": float(np.mean(accs)),
                     "acc_std": float(np.std(accs)),
                     "f1_mean": float(np.mean(f1s)),
                     "f1_std": float(np.std(f1s)),
                     "accs": accs, "f1s": f1s,
                     "n_input_features": int(len(reps)),
                     "n_cloud_train": int(infos[0]["n_train"]),
                     "n_cloud_val": int(infos[0]["n_val"]),
                     "validation_mode": infos[0]["validation_mode"],
                     "cloud_optimizer_steps": infos[0]["optimizer_steps"],
                     "best_cloud_epoch": infos[0]["best_epoch"],
                     "payload": sizes, "quant": q.meta()}
        print(f"{mode:<5} {out[mode]['acc_mean']:6.2f}±{out[mode]['acc_std']:.2f} "
              f"bytes {sizes['total']}", flush=True)
    pt = payload_table(49000, len(reps), cfg["payload"]["label_bytes"],
                       cfg["payload"]["bandwidth_mbps"])
    res = {"tag": a.tag, "meta": meta, "cloud": cloud, "modes": out,
           "payload_table": pt}
    p = result_path(cfg, (a.out or a.tag + "_cloud") + ".json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"结果: {p}")


if __name__ == "__main__":
    main()
