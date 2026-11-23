"""配对回放校正历史均值诊断（CIFAR-100 Plain DM, seed 42）

对照三种 DM 目标（其余流程完全一致：代表集初始化/对齐/锚定/20 步优化/
int8 full_sync/云端 200 步增量训练）:

  plain  原 Plain DM 目标：Replay + 当前轮（各样本只计一次）
  exact  精确历史目标（离线 oracle）：用当轮 encoder 重编码 base + 0..t 全部图像
  paired 配对回放校正目标（仅用在线可用信息）:
         ĥ_{t,c} = ā_{all history,c} + mean_{i∈R_c}[u_t(x_i) − a_i]
         target_c = (N_c·ĥ_{t,c} + Σ_{本轮 c} u_t) / (N_c + n_c)

其中 a_i = φ(E_arrival(x_i))，u_t(x_i) = φ(E_t(x_i))，N_c 为当前轮之前
该类的全历史计数（当前轮通过第二项合并，保证每样本只计一次）。

流程:
  Stage 1  重跑到第 60 轮，在 20/40/60 保存 DM 更新前完整快照
  Stage 2  对每个快照计算三种目标的逐类 MSE，并在相同条件下各跑一次
           代表集更新 + 同步 + 云端训练 + 验证/测试评估

用法:
  python -m vae_streaming.experiments.diag_dm_target
  python -m vae_streaming.experiments.diag_dm_target --skip-stage1
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from random_feature import RandomMLP  # noqa: E402

from vae_bdm.quant import Quantizer  # noqa: E402

from vae_streaming.cloud.incremental_trainer import IncrementalCloudLearner  # noqa: E402
from vae_streaming.cloud.state_sync import CloudState  # noqa: E402
from vae_streaming.communication.payload_accounting import (  # noqa: E402
    build_sync_message)
from vae_streaming.communication.quantization import decode_features  # noqa: E402
from vae_streaming.config_utils import checkpoint_path, result_path  # noqa: E402
from vae_streaming.distillation.streaming_dm import StreamingDM  # noqa: E402
from vae_streaming.experiments.engine import _build_stream, run_method  # noqa: E402
from vae_streaming.memory.representative_memory import (  # noqa: E402
    RepresentativeMemory)
from vae_streaming.models.online_vae import (  # noqa: E402
    FeatureProtocol, build_vae, encode_mu)

DIAG_ROUNDS = (20, 40, 60)
ORIGINAL_RESULT = ("vae_streaming/results/cifar100/iid/"
                   "dm_fixinit80c100_seed42.json")


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def _load_original():
    with open(os.path.join(ROOT, ORIGINAL_RESULT), "r", encoding="utf-8") as f:
        return json.load(f)


def _cfg_from_original(orig):
    """直接复用原 80 轮 Plain DM 的完整（已合并方法覆盖）配置。"""
    cfg = copy.deepcopy(orig["config"])
    cfg["_run_tag"] = "fixinit80c100_diag"
    cfg["_run_name"] = None
    return cfg


@torch.no_grad()
def _encode_norm(protocol, encoder, images, device):
    return protocol.normalize(encode_mu(encoder, images, device=device))


@torch.no_grad()
def _phi(phi, feats, device, chunk=8192):
    """φ 作用在归一化特征上；返回 CPU float32。"""
    outs = []
    for i in range(0, len(feats), chunk):
        outs.append(phi(feats[i:i + chunk].to(device)).float().cpu())
    if not outs:
        return torch.zeros(0, phi.net[-1].out_features)
    return torch.cat(outs, 0)


def _class_sum(feats, labels, num_classes):
    s = torch.zeros(num_classes, feats.size(1))
    s.index_add_(0, labels.long(), feats)
    cnt = torch.bincount(labels.long(), minlength=num_classes).long()
    return s, cnt


def _mse_metrics(target, exact, counts):
    mask = counts > 0
    if int(mask.sum()) == 0:
        return {"mse": None, "relative_mse": None, "cosine": None}
    t, e = target[mask], exact[mask]
    mse = float((t - e).pow(2).mean())
    denom = float(e.pow(2).mean())
    cos = torch.nn.functional.cosine_similarity(t, e, dim=1).mean()
    per_class = (t - e).pow(2).mean(1)
    return {"mse": mse,
            "relative_mse": mse / max(denom, 1e-12),
            "cosine": float(cos),
            "per_class_mse_mean": float(per_class.mean()),
            "per_class_mse_median": float(per_class.median()),
            "n_classes": int(mask.sum())}


def _load_vae_state(ckpt_dir, version, cfg, device, cache):
    if version in cache:
        return cache[version]
    path = os.path.join(ckpt_dir, f"vae_v{int(version)}.pt")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    vae = build_vae(int(cfg["num_classes"]), int(cfg["latent_dim"]),
                    int(cfg["base_channels"]), seed=None, device=device)
    vae.load_state_dict(obj["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    cache[version] = vae
    return vae


# ----------------------------------------------------------------------
# Stage 1: 重跑并保存 DM 更新前快照
# ----------------------------------------------------------------------
def stage1(cfg, seed=42):
    print("=" * 70)
    print("Stage 1: 重跑 Plain DM 到第 60 轮并在 20/40/60 保存快照")
    print("=" * 70, flush=True)
    return run_method(cfg, "dm", seed=seed, verbose=True, save=True,
                      stop_after=max(DIAG_ROUNDS),
                      snapshot_rounds=list(DIAG_ROUNDS))


def _check_repro(diag_result, orig):
    ref = {int(r["round"]): float(r["acc"]) for r in orig["rounds"]}
    diffs = []
    for rec in diag_result["rounds"]:
        t = int(rec["round"])
        if t in ref:
            diffs.append(abs(float(rec["acc"]) - ref[t]))
    return {"n_rounds": len(diffs),
            "max_abs_acc_diff": float(max(diffs)) if diffs else None,
            "mean_abs_acc_diff": float(np.mean(diffs)) if diffs else None}


# ----------------------------------------------------------------------
# Stage 2: 目标计算 + 同条件下游评估
# ----------------------------------------------------------------------
def _snapshot_state(snap, cfg, device):
    vae = build_vae(int(cfg["num_classes"]), int(cfg["latent_dim"]),
                    int(cfg["base_channels"]), seed=None, device=device)
    vae.load_state_dict(snap["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    protocol = FeatureProtocol(snap["protocol_mean"], snap["protocol_std"])
    return vae, protocol


def _run_variant(snap, cfg, stream, vae, protocol, device, round_idx,
                 target=None, counts=None):
    """在与快照完全相同的条件下执行一次 DM 更新 + 同步 + 云端训练。"""
    memory = RepresentativeMemory.from_state_dict(snap["memory"])
    real_f = torch.cat([snap["rep_feats_reenc"], snap["cur_cls_f"]], 0)
    real_y = torch.cat([snap["rep_labels"], snap["cur_cls_y"]], 0)
    real_src = torch.zeros(len(real_y), dtype=torch.long)

    sdm = StreamingDM(memory.features, memory.labels, memory.quotas,
                      cfg["distillation"], seed=int(snap["seed"]),
                      device=device)
    if target is None:
        out = sdm.update(round_idx=round_idx, real_features=real_f,
                         real_labels=real_y, real_sources=real_src)
    else:
        out = sdm.update(round_idx=round_idx, real_features=real_f,
                         real_labels=real_y, real_sources=real_src,
                         real_mean_override=target, target_counts=counts)
    memory.features = out["features"]

    # 同步（与 engine 相同的 full_sync + int8）
    msg, _ = build_sync_message("full_sync", round_idx, memory, None,
                                cfg["communication"]["quantization"],
                                float(cfg["communication"].get(
                                    "patch_fraction", 0.1)))
    cloud = CloudState.from_state_dict(snap["cloud"])
    changed = cloud.apply(msg)
    header = json.loads(msg["meta"])
    dec = decode_features(msg["features"], header["quant"])
    memory.set_submitted(np.arange(memory.budget), dec,
                         int(snap["new_version"]))
    verify = cloud.verify_consistent(memory)
    if not verify["all_ok"]:
        raise RuntimeError(f"r{round_idx} 同步不一致: {verify}")

    # 云端增量训练（与原流程同一步数/种子）
    learner = IncrementalCloudLearner(cfg, int(cfg["feature_dim"]),
                                      int(cfg["num_classes"]), device=device)
    learner.load_state_dict(snap["learner"])
    cloud_mode = "incremental" if cfg["cloud"].get(
        "incremental_training", True) else "scratch"
    learner.update(cloud.features, cloud.labels, None, None,
                   round_idx=round_idx, mode=cloud_mode)

    test_feats = _encode_norm(protocol, vae.encoder, stream.test_images, device)
    val_feats = _encode_norm(protocol, vae.encoder, stream.val_images, device)
    test_met = learner.evaluate(test_feats, stream.test_labels)
    val_met = learner.evaluate(val_feats, stream.val_labels)

    q = Quantizer("int8").fit(memory.features)
    test_int8 = learner.evaluate(q.dequantize(q.quantize(test_feats)),
                                 stream.test_labels)["acc"]
    return {"val_acc": float(val_met["acc"]),
            "test_acc": float(test_met["acc"]),
            "test_int8_acc": float(test_int8),
            "sync_changed": bool(changed),
            "n_features": int(len(memory.features)),
            "mean_loss": float(out["info"].get("mean_loss", 0.0))}


def stage2(cfg, orig, device, seed=42):
    print("=" * 70)
    print("Stage 2: 三种目标的计算与下游评估")
    print("=" * 70, flush=True)
    C = int(cfg["num_classes"])
    # run_method 命名: dm + _run_tag
    ckpt_dir = checkpoint_path(
        cfg, f"{cfg['dataset']}/iid/seed{seed}/dm_{cfg['_run_tag']}")

    # 数据流与原始运行完全一致
    stream = _build_stream(cfg)

    phi = RandomMLP(input_dim=int(cfg["feature_dim"]),
                    hidden_dim=int(cfg["distillation"].get(
                        "feature_hidden", 2048)),
                    output_dim=int(cfg["distillation"].get(
                        "feature_out", 1024)),
                    seed=int(cfg["distillation"].get("feature_seed", 0)))
    phi = phi.to(device).eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    vae_cache = {}
    vae0 = _load_vae_state(ckpt_dir, 0, cfg, device, vae_cache)
    protocol0 = None  # 由第一个快照给出（冻结协议，始终相同）

    # 全历史 arrival 特征累计（base + 已处理轮）
    base_images, base_labels = stream.base.images, stream.base.labels
    hist_sum = hist_cnt = None
    round_images = {}
    round_labels = {}
    next_hist_round = 0

    ref = {int(r["round"]): r for r in orig["rounds"]}
    out = {"checkpoints": {}, "meta": {
        "rounds": list(DIAG_ROUNDS), "seed": seed,
        "definition": {
            "a_i": "phi(normalize(E_arrival(x_i))), arrival version = round//2 (base=0)",
            "u_t": "phi(normalize(E_t(x)))",
            "paired": "h_hat_c = a_hist_mean_c + mean_{R_c}(u_t-a_i); "
                      "target = (N_c*h_hat_c + sum_cur_c u_t)/(N_c+n_c)",
            "N_c": "全历史计数（不含当前轮，当前轮单独合并）",
        }}}

    for t in DIAG_ROUNDS:
        snap_path = os.path.join(ckpt_dir, f"diag_snapshot_r{t}.pt")
        if not os.path.exists(snap_path):
            raise FileNotFoundError(f"缺少快照 {snap_path}，请先运行 Stage 1")
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        if int(snap["round"]) != int(t):
            raise ValueError(f"快照轮次不一致: {snap['round']} != {t}")

        # 访问数据流到第 t 轮（严格顺序）
        for r in range(stream._max_accessed + 1, t + 1):
            rb = stream.round(r)
            round_images[r] = rb.images
            round_labels[r] = rb.labels
        vae_t, protocol = _snapshot_state(snap, cfg, device)
        if protocol0 is None:
            protocol0 = protocol
            a_base = _phi(phi, _encode_norm(protocol, vae0.encoder,
                                            base_images, device), device)
            hist_sum, hist_cnt = _class_sum(a_base, base_labels, C)
        assert snap["version"] == snap["new_version"] == t // 2

        # ---- 全历史 arrival 累计推进到 0..t-1 ----
        while next_hist_round < t:
            r = next_hist_round
            v_arr = _load_vae_state(ckpt_dir, r // 2, cfg, device, vae_cache)
            a_r = _phi(phi, _encode_norm(protocol, v_arr.encoder,
                                         round_images[r], device), device)
            s_r, c_r = _class_sum(a_r, round_labels[r], C)
            hist_sum += s_r
            hist_cnt += c_r
            next_hist_round += 1

        # ---- 当前轮 u_t（快照中已由当轮 encoder 编码）----
        u_cur = _phi(phi, snap["cur_cls_f"], device)
        cur_sum, cur_cnt = _class_sum(u_cur, snap["cur_cls_y"], C)

        # ---- Replay 的 u_t 与 a_i ----
        per_class = snap["replay"]["per_class"]
        rep_images = torch.stack([e[0] for buf in per_class for e in buf], 0)
        rep_labels = torch.tensor([int(e[1]) for buf in per_class for e in buf],
                                  dtype=torch.long)
        rep_rounds = np.asarray([int(e[2]) for buf in per_class for e in buf],
                                dtype=np.int64)
        if not torch.equal(rep_labels, snap["rep_labels"]):
            raise RuntimeError("Replay 顺序与快照 rep_labels 不一致")
        u_rep = _phi(phi, snap["rep_feats_reenc"], device)
        a_rep = torch.zeros_like(u_rep)
        for rnd in sorted(set(rep_rounds.tolist())):
            sel = np.where(rep_rounds == rnd)[0]
            v_arr = _load_vae_state(ckpt_dir, 0 if rnd < 0 else rnd // 2,
                                    cfg, device, vae_cache)
            a_rep[torch.from_numpy(sel)] = _phi(
                phi, _encode_norm(protocol, v_arr.encoder,
                                  rep_images[torch.from_numpy(sel)], device),
                device)

        # ---- 三种目标 ----
        plain_sum = torch.zeros(C, u_rep.size(1))
        plain_cnt = torch.zeros(C, dtype=torch.long)
        for feats, labels, s_obj, c_obj in (
                (u_rep, rep_labels, None, None),
                (u_cur, snap["cur_cls_y"], None, None)):
            s_, c_ = _class_sum(feats, labels, C)
            plain_sum += s_
            plain_cnt += c_
        plain_target = plain_sum / plain_cnt.clamp_min(1)[:, None]

        # exact: 当轮 encoder 重编码 base + 0..t
        all_imgs = torch.cat([base_images] + [round_images[r] for r in range(t + 1)], 0)
        all_y = torch.cat([base_labels] + [round_labels[r] for r in range(t + 1)], 0)
        u_all = _phi(phi, _encode_norm(protocol, vae_t.encoder, all_imgs, device),
                     device)
        exact_sum, exact_cnt = _class_sum(u_all, all_y, C)
        exact_target = exact_sum / exact_cnt.clamp_min(1)[:, None]

        # paired: 全历史 ā + Replay 漂移校正，再合并当前轮
        r_sum, r_cnt = _class_sum(u_rep - a_rep, rep_labels, C)
        a_hist_mean = hist_sum / hist_cnt.clamp_min(1)[:, None]
        h_hat = a_hist_mean.clone()
        valid_r = r_cnt > 0
        h_hat[valid_r] = a_hist_mean[valid_r] + r_sum[valid_r] / r_cnt[
            valid_r, None].float()
        total_cnt = hist_cnt + cur_cnt
        paired_target = (hist_cnt[:, None].float() * h_hat + cur_sum) / \
            total_cnt.clamp_min(1)[:, None].float()

        counts = total_cnt

        # ---- 目标误差 ----
        target_report = {
            "plain": _mse_metrics(plain_target, exact_target, counts),
            "exact": _mse_metrics(exact_target, exact_target, counts),
            "paired": _mse_metrics(paired_target, exact_target, counts),
        }

        # ---- 下游准确率：三种目标各跑一次完全相同的流程 ----
        acc = {}
        for name, tgt in (("plain", None), ("exact", exact_target),
                          ("paired", paired_target)):
            use_counts = None if tgt is None else counts
            acc[name] = _run_variant(snap, cfg, stream, vae_t, protocol,
                                     device, t, target=tgt,
                                     counts=use_counts)
            print(f"  r{t} [{name:6s}] "
                  f"target_mse={target_report[name]['mse']:.4f} "
                  f"val={acc[name]['val_acc']:.2f} "
                  f"test={acc[name]['test_acc']:.2f}", flush=True)

        # ---- 与原 Plain DM 参考对比 ----
        orig_ref = ref.get(t, {})
        out["checkpoints"][str(t)] = {
            "snapshot": {"version": int(snap["version"]),
                         "do_update": bool(snap["do_update"]),
                         "n_history": int(hist_cnt.sum()),
                         "n_current": int(cur_cnt.sum()),
                         "n_replay": int(len(rep_labels))},
            "targets": target_report,
            "accuracy": acc,
            "reference_plain_dm": {
                "acc": orig_ref.get("acc"),
                "test_int8_acc": orig_ref.get("test_int8_acc")},
            "delta_vs_plain": {
                "exact": {k: acc["exact"][k] - acc["plain"][k]
                          for k in ("val_acc", "test_acc")},
                "paired": {k: acc["paired"][k] - acc["plain"][k]
                           for k in ("val_acc", "test_acc")},
            },
        }

    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    orig = _load_original()
    cfg = _cfg_from_original(orig)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    diag_result = None
    if not args.skip_stage1:
        diag_result = stage1(cfg, seed=args.seed)

    out = stage2(cfg, orig, device, seed=args.seed)
    if diag_result is not None:
        out["reproducibility"] = _check_repro(diag_result, orig)

    path = result_path(cfg, f"{cfg['dataset']}/iid/"
                            f"diag_dm_target_seed{args.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n-> {path}", flush=True)

    print("\n" + "=" * 70)
    print("三检查点结果（目标相对 exact 的 MSE / 验证准确率 / 测试准确率）")
    print("=" * 70)
    header = f"{'round':>5} {'variant':>7} {'mse':>9} {'rel_mse':>9} " \
             f"{'val':>7} {'test':>7}"
    print(header)
    for t in DIAG_ROUNDS:
        c = out["checkpoints"][str(t)]
        for name in ("plain", "exact", "paired"):
            m = c["targets"][name]
            a = c["accuracy"][name]
            print(f"{t:>5} {name:>7} {m['mse']:>9.4f} "
                  f"{m['relative_mse']:>9.4f} {a['val_acc']:>7.2f} "
                  f"{a['test_acc']:>7.2f}")
        print(f"{t:>5} {'ref':>7} {'-':>9} {'-':>9} {'-':>7} "
              f"{c['reference_plain_dm']['acc']:>7.2f}")


if __name__ == "__main__":
    main()
