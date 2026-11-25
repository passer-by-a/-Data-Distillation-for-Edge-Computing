"""末轮云端重训配对诊断（Plain DM，CIFAR-10 80 轮，u2）

对每个 seed 固定同一份 r79 代表集与同一份 val/test 特征，比较三组：
  A. 原增量模型：加载断点中的云端分类器，只评估（0 步）
  B. 从头 200 步：新建 MLP + AdamW，余弦计划 T_max=200
  C. 从头 20 步：与 B 相同的初始权重 / 采样 / Dropout 序列，T_max=20

训练特征固定为断点里的 cloud.features（云端实际收到的反量化特征），
不使用边缘 memory.features；验证/测试特征由该 seed 的 r79 VAE 编码一次后共享。
所有训练调用 val_features=None，禁止轮内按验证集选优。

用法：
  python -m vae_streaming.experiments.diag_final_cloud_refit \
      --tag sup80final --seeds 42,43,44
"""
import argparse
import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from vae_bdm.cloud import evaluate_cloud_classifier
from vae_bdm.quant import Quantizer

from ..cloud.incremental_trainer import IncrementalCloudLearner
from ..config_utils import checkpoint_path, load_config, result_path
from ..experiments.engine import _build_stream
from ..models.online_vae import FeatureProtocol, build_vae, encode_mu


def _module_sha1(module):
    h = hashlib.sha1()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _encode(vae, protocol, images, device):
    with torch.no_grad():
        raw = encode_mu(vae.encoder, images, device=device)
    return protocol.normalize(raw).float().cpu()


def _eval(model, feats, labels, num_classes, device):
    out = evaluate_cloud_classifier(model, feats, labels, num_classes)
    model.eval()
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), 512):
            xb = feats[i:i + 512].float().to(device)
            yb = labels[i:i + 512].long().to(device)
            logits = model(xb)
            nll_sum += F.cross_entropy(logits, yb,
                                       reduction="sum").item()
            n += int(len(yb))
    return {"acc": float(out["acc"]), "f1": float(out["f1"]),
            "nll": float(nll_sum / max(n, 1)), "pred": out["pred"]}


def _train_scratch(cfg, feats, labels, steps, round_idx, num_classes,
                   device):
    c = copy.deepcopy(cfg)
    c["cloud"]["steps_per_round"] = int(steps)
    learner = IncrementalCloudLearner(c, feats.shape[1], num_classes,
                                      device=device)
    info = learner.update(feats, labels, None, None,
                          round_idx=round_idx, mode="scratch")
    return learner, info


def _pair_counts(orig_pred, new_pred, labels):
    orig_right = orig_pred == labels
    new_right = new_pred == labels
    corrected = int((~orig_right & new_right).sum())
    worsened = int((orig_right & ~new_right).sum())
    return {"corrected": corrected, "worsened": worsened,
            "net": corrected - worsened,
            "orig_right": int(orig_right.sum()),
            "new_right": int(new_right.sum()),
            "n": int(len(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="vae_streaming/configs/cifar10_supervised_multi80_u2.yaml")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--method", default="dm")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--tag", default="sup80final")
    ap.add_argument("--ref-tag", default="sup80")
    ap.add_argument("--round", type=int, default=79)
    ap.add_argument("--short-steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = _build_stream(cfg, None)
    num_classes = int(cfg["num_classes"])
    steps_full = int(cfg["cloud"]["steps_per_round"])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []

    for seed in seeds:
        ckpt = checkpoint_path(
            cfg, f"{args.dataset}/iid/seed{seed}/{args.method}_{args.tag}"
                 f"/stream_state.pt")
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        last = int(st["last_round"])
        if last != args.round:
            raise RuntimeError(
                f"seed{seed} 断点 last_round={last} != {args.round}: {ckpt}")

        vae = build_vae(num_classes, int(cfg["latent_dim"]),
                        int(cfg["base_channels"]), seed=None, device=device)
        vae.load_state_dict(st["vae"])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        protocol = FeatureProtocol(st["protocol_mean"], st["protocol_std"])

        train_x = torch.as_tensor(st["cloud"]["features"]).float().cpu()
        train_y = torch.as_tensor(st["cloud"]["labels"]).long().cpu()
        val_x = _encode(vae, protocol, stream.val_images, device)
        test_x = _encode(vae, protocol, stream.test_images, device)
        # 主表另有 test_int8_acc：用该轮边缘代表的全局 INT8 量化器处理测试特征
        mem_x = torch.as_tensor(st["memory"]["features"]).float().cpu()
        q = Quantizer("int8").fit(mem_x)
        test_x_int8 = q.dequantize(q.quantize(test_x)).float().cpu()
        val_y = torch.as_tensor(stream.val_labels).long().cpu()
        test_y = torch.as_tensor(stream.test_labels).long().cpu()

        # A. 原增量模型：只加载与评估
        orig = IncrementalCloudLearner(cfg, train_x.shape[1], num_classes,
                                       device=device)
        orig.load_state_dict(st["learner"])
        m_orig = orig.model

        # B/C 的初始权重必须逐位相同（scratch 用 seed + round 构建）
        l200 = IncrementalCloudLearner(cfg, train_x.shape[1], num_classes,
                                       device=device)
        l20 = IncrementalCloudLearner(cfg, train_x.shape[1], num_classes,
                                      device=device)
        init200, _ = l200._build(l200.seed + args.round)
        init20, _ = l20._build(l20.seed + args.round)
        init_same = _module_sha1(init200) == _module_sha1(init20)
        init_sha1 = _module_sha1(init200)
        del init200, init20

        l200, info200 = _train_scratch(cfg, train_x, train_y, steps_full,
                                       args.round, num_classes, device)
        l20, info20 = _train_scratch(cfg, train_x, train_y,
                                     args.short_steps, args.round,
                                     num_classes, device)

        result = {"seed": seed, "last_round": last,
                  "n_train": int(len(train_x)),
                  "init_same": bool(init_same), "init_sha1": init_sha1,
                  "orig_round": int(orig.round),
                  "full_steps": steps_full,
                  "short_steps": int(args.short_steps)}
        preds = {}
        for name, model in (("original", m_orig),
                            ("scratch200", l200.model),
                            ("scratch20", l20.model)):
            val_metrics = _eval(model, val_x, val_y, num_classes, device)
            test_metrics = _eval(model, test_x, test_y, num_classes, device)
            int8_metrics = _eval(model, test_x_int8, test_y, num_classes,
                                 device)
            train_metrics = _eval(model, train_x, train_y, num_classes,
                                  device)
            preds[name] = (val_metrics.pop("pred"), test_metrics.pop("pred"))
            int8_metrics.pop("pred", None)
            train_metrics.pop("pred", None)
            result[name] = {"val": val_metrics, "test": test_metrics,
                            "test_int8": int8_metrics,
                            "train_rep": train_metrics}

        result["paired_val"] = _pair_counts(
            preds["original"][0], preds["scratch200"][0], val_y)
        result["paired_test"] = _pair_counts(
            preds["original"][1], preds["scratch200"][1], test_y)

        # 与主表原 JSON 对照，确认加载的是 Plain DM 而不是 Reservoir
        try:
            ref = json.load(open(result_path(
                cfg, f"{args.dataset}/iid/{args.method}_{args.ref_tag}"
                     f"_seed{seed}.json")))
            result["ref_final_acc"] = float(ref["rounds"][-1]["acc"])
        except Exception:
            result["ref_final_acc"] = None
        result["original_final_acc"] = result["original"]["test"]["acc"]
        result["delta_vs_ref"] = (
            None if result["ref_final_acc"] is None
            else result["original_final_acc"] - result["ref_final_acc"])
        rows.append(result)

        print(f"seed{seed}: n_train={len(train_x)} init_same={init_same} "
              f"ref={result['ref_final_acc']} "
              f"orig={result['original_final_acc']:.2f} "
              f"scratch200={result['scratch200']['test']['acc']:.2f} "
              f"scratch20={result['scratch20']['test']['acc']:.2f}",
              flush=True)

    gains_val = [r["scratch200"]["val"]["acc"] -
                 r["original"]["val"]["acc"] for r in rows]
    gains_test = [r["scratch200"]["test"]["acc"] -
                  r["original"]["test"]["acc"] for r in rows]
    gains_short = [r["scratch20"]["val"]["acc"] -
                   r["original"]["val"]["acc"] for r in rows]
    gains_int8 = [r["scratch200"]["test_int8"]["acc"] -
                  r["original"]["test_int8"]["acc"] for r in rows]
    mean_val = float(np.mean(gains_val)) if gains_val else None
    worth = bool(mean_val is not None and mean_val >= 0.5
                 and all(g > 0 for g in gains_val))
    summary = {
        "tag": args.tag, "ref_tag": args.ref_tag, "round": args.round,
        "method": args.method, "seeds": seeds, "full_steps": steps_full,
        "short_steps": int(args.short_steps),
        "device": str(device),
        "gains_val": gains_val, "gains_test": gains_test,
        "gains_val_short": gains_short,
        "gains_test_int8": gains_int8,
        "mean_gain_val": mean_val,
        "mean_gain_test": float(np.mean(gains_test)) if gains_test else None,
        "mean_gain_test_int8": (float(np.mean(gains_int8))
                                if gains_int8 else None),
        "mean_gain_val_short": (float(np.mean(gains_short))
                                if gains_short else None),
        "all_seeds_positive_val": bool(all(g > 0 for g in gains_val)),
        "worth_online_test": worth,
        "criterion": "从头 200 步相对原增量：三 seed 验证增益均 >0 且均值 >= +0.5 pp",
        "rows": rows,
    }
    out = args.out or result_path(
        cfg, f"{args.dataset}/iid/final_cloud_refit_"
             f"{args.method}_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
