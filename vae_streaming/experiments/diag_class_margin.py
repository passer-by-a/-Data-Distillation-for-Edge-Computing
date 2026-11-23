"""CIFAR-10 分类间隔诊断（决定是否值得做 margin 分位数匹配短试）

Stage 1  同一 seed / 数据流下重跑 Plain DM 与 online_reservoir_latent 到第 60 轮，
         在 20/40/60 保存代表集更新前快照；按原 80 轮结果的 acc 曲线校验复现。
Stage 2  每个快照用同一冻结的边缘分类头 (eval) 计算间隔
             m(z, c) = h_c(z) - max_{k != c} h_k(z)
         比较三组分布:
             真实样本 = 当前轮 + Replay
             Plain DM 代表（按当轮普通 DM 目标确定性更新后）
             潜变量 Reservoir 代表（按当轮 reservoir_latent 逻辑更新后）
         逐类做 (i-0.5)/n 分位数匹配，量出 MAD；用 bootstrap 得到"代表预算
         下真实样本自身的抽样波动"作为噪声基线。
         另用当轮云端模型在验证集上统计低间隔样本是否更易分错（AUC / 分箱）。

判据:
  A  代表集间隔分布与真实的差距明显超过真实样本自身的抽样波动
  B  验证集上边缘分类头给出低间隔的样本更容易被云端分错
  A 且 B 才值得做 20 轮配对短试；只看到间隔误差下降而云端准确率无改善则停。

用法:
  python -m vae_streaming.experiments.diag_class_margin
  python -m vae_streaming.experiments.diag_class_margin --skip-stage1
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

from sklearn.metrics import roc_auc_score  # noqa: E402

from vae_streaming.cloud.incremental_trainer import IncrementalCloudLearner  # noqa: E402
from vae_streaming.cloud.state_sync import CloudState  # noqa: E402
from vae_streaming.communication.payload_accounting import (  # noqa: E402
    build_sync_message)
from vae_streaming.communication.quantization import decode_features  # noqa: E402
from vae_streaming.config_utils import checkpoint_path, result_path  # noqa: E402
from vae_streaming.distillation.streaming_dm import StreamingDM  # noqa: E402
from vae_streaming.experiments.engine import _build_stream, run_method  # noqa: E402
from vae_streaming.memory.replay_buffer import FeatureReservoir  # noqa: E402
from vae_streaming.memory.representative_memory import (  # noqa: E402
    RepresentativeMemory)
from vae_streaming.models.online_vae import (  # noqa: E402
    FeatureProtocol, build_vae, encode_mu)

DIAG_ROUNDS = (20, 40, 60)
METHODS = ("dm", "online_reservoir_latent")
REF_RESULTS = {
    "dm": "vae_streaming/results/cifar10/iid/dm_sup80_seed42.json",
    "online_reservoir_latent":
        "vae_streaming/results/cifar10/iid/"
        "online_reservoir_latent_fixinit80_seed42.json",
}
RUN_TAGS = {"dm": "margin_dm", "online_reservoir_latent": "margin_rsvlat"}
N_BOOT_NULL = 20      # 每类噪声基线的 bootstrap 次数
N_SPLIT_HALF = 10     # 真实样本对半分的重复次数
N_BOOT_CI = 400       # 验证集 AUC / 错误率差的 bootstrap 次数


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def _load_ref(method):
    with open(os.path.join(ROOT, REF_RESULTS[method]), "r",
              encoding="utf-8") as f:
        return json.load(f)


def _cfg_from_ref(ref, method):
    cfg = copy.deepcopy(ref["config"])
    cfg["_run_tag"] = RUN_TAGS[method]
    cfg["_run_name"] = None
    return cfg


def _run_tag(method):
    return f"{method}_{RUN_TAGS[method]}"


def _check_repro(diag_result, ref):
    ref_acc = {int(r["round"]): float(r["acc"]) for r in ref["rounds"]}
    diffs = [abs(float(rec["acc"]) - ref_acc[int(rec["round"])])
             for rec in diag_result["rounds"] if int(rec["round"]) in ref_acc]
    return {"n_rounds": len(diffs),
            "max_abs_acc_diff": float(max(diffs)) if diffs else None,
            "mean_abs_acc_diff": float(np.mean(diffs)) if diffs else None}


# ----------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------
def stage1(cfg, method, seed):
    print("=" * 70)
    print(f"Stage 1: 重跑 {method} 到第 {max(DIAG_ROUNDS)} 轮并保存快照")
    print("=" * 70, flush=True)
    return run_method(cfg, method, seed=seed, verbose=True, save=True,
                      stop_after=max(DIAG_ROUNDS),
                      snapshot_rounds=list(DIAG_ROUNDS))


# ----------------------------------------------------------------------
# 分类间隔
# ----------------------------------------------------------------------
def _snapshot_vae(snap, cfg, device):
    vae = build_vae(int(cfg["num_classes"]), int(cfg["latent_dim"]),
                    int(cfg["base_channels"]), seed=None, device=device)
    vae.load_state_dict(snap["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    protocol = FeatureProtocol(snap["protocol_mean"], snap["protocol_std"])
    return vae, protocol


@torch.no_grad()
def _classify(feats, labels, vae, protocol, device, chunk=2048):
    """归一化潜特征 → (间隔 margin, 分类头预测 pred)

    先把 (raw - mean) / std 还原到分类头训练时的尺度与 16×8×8 布局。
    margin 为真类 logit 减最大异类 logit：接近 0 表示靠近分类边界。
    """
    C, H, W = vae.latent_dim, vae.spatial_size, vae.spatial_size
    mean = protocol.mean.to(device)
    std = protocol.std.to(device)
    margins, preds = [], []
    for i in range(0, len(feats), chunk):
        raw = feats[i:i + chunk].float().to(device).view(-1, C, H, W)
        logits = vae.classifier(raw * std + mean)
        y = labels[i:i + chunk].long().to(device)
        true = logits.gather(1, y[:, None]).squeeze(1)
        other = logits.clone()
        other.scatter_(1, y[:, None], float("-inf"))
        margins.append((true - other.max(1).values).float().cpu())
        preds.append(logits.argmax(1).cpu())
    if not margins:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(margins), torch.cat(preds)


def _stats(x):
    a = np.asarray(x, dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "q10": float(np.quantile(a, 0.1)),
            "q50": float(np.quantile(a, 0.5)),
            "q90": float(np.quantile(a, 0.9)), "max": float(a.max()),
            "frac_neg": float((a < 0).mean())}


def _quantile_match(rep_vals, real_vals):
    """代表集第 i 个排序间隔 ↔ 真实分布的 (i-0.5)/n 分位数。"""
    q = np.sort(np.asarray(rep_vals, dtype=np.float64))
    r = np.sort(np.asarray(real_vals, dtype=np.float64))
    if len(q) == 0 or len(r) < 2:
        return None
    p = (np.arange(len(q)) + 0.5) / len(q)
    tgt = np.quantile(r, p)
    d = np.abs(q - tgt)
    sd = max(float(r.std(ddof=1)), 1e-12)
    return {"n_rep": int(len(q)), "n_real": int(len(r)),
            "mad": float(d.mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "max_abs": float(d.max()),
            "mad_z": float(d.mean() / sd),
            "bias": float(q.mean() - r.mean()),
            "bias_z": float((q.mean() - r.mean()) / sd)}


def _null_floor(real_vals, n_rep, reps=N_BOOT_NULL, seed=0):
    """若代表集是来自真实分布的 n_rep 个 iid 样本，分位数匹配误差的期望尺度。"""
    r = np.asarray(real_vals, dtype=np.float64)
    if len(r) < 2 or n_rep <= 0:
        return None
    rng = np.random.RandomState(int(seed))
    mads = []
    for _ in range(reps):
        draw = rng.choice(r, size=n_rep, replace=True)
        qm = _quantile_match(draw, r)
        if qm is not None:
            mads.append(qm["mad"])
    if not mads:
        return None
    return {"reps": len(mads), "mean": float(np.mean(mads)),
            "std": float(np.std(mads, ddof=1)) if len(mads) > 1 else 0.0,
            "p95": float(np.quantile(mads, 0.95))}


def _split_half_noise(real_vals, reps=N_SPLIT_HALF, seed=0):
    """真实样本按类随机对半分，量出真实样本自身的大样本抽样波动。"""
    r = np.asarray(real_vals, dtype=np.float64)
    if len(r) < 8:
        return None
    rng = np.random.RandomState(int(seed))
    mads = []
    for _ in range(reps):
        perm = rng.permutation(len(r))
        a, b = r[perm[:len(r) // 2]], r[perm[len(r) // 2:]]
        qm = _quantile_match(a, b)
        if qm is not None:
            mads.append(qm["mad"])
    if not mads:
        return None
    return {"reps": len(mads), "mean": float(np.mean(mads)),
            "max": float(np.max(mads))}


def _class_margin_report(real_m, real_y, rep_m, rep_y, C, seed):
    real_m = np.asarray(real_m, dtype=np.float64)
    rep_m = np.asarray(rep_m, dtype=np.float64)
    per_class = {}
    for c in range(C):
        rm = real_m[np.asarray(real_y) == c]
        pm = rep_m[np.asarray(rep_y) == c]
        row = {"real": _stats(rm), "rep": _stats(pm)}
        if len(rm) >= 5 and len(pm) >= 5:
            qm = _quantile_match(pm, rm)
            null = _null_floor(rm, len(pm), seed=seed + c)
            sh = _split_half_noise(rm, seed=seed + c)
            row["quantile"] = qm
            row["null_budget"] = null
            row["split_half"] = sh
            if null is not None and null["mean"] > 0:
                row["ratio_vs_null"] = qm["mad"] / null["mean"]
                row["exceeds_null_p95"] = bool(qm["mad"] > null["p95"])
        per_class[str(c)] = row

    valid = [per_class[str(c)] for c in range(C)
             if "quantile" in per_class[str(c)]]
    agg = {"n_classes": len(valid)}
    if valid:
        mads = np.array([r["quantile"]["mad"] for r in valid])
        nulls = np.array([r["null_budget"]["mean"] for r in valid])
        p95 = np.array([r["null_budget"]["p95"] for r in valid])
        ratios = mads / np.maximum(nulls, 1e-12)
        agg.update({
            "mad_mean": float(mads.mean()),
            "mad_median": float(np.median(mads)),
            "null_mean": float(nulls.mean()),
            "ratio_mean": float(ratios.mean()),
            "ratio_median": float(np.median(ratios)),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "n_exceed_null_p95": int((mads > p95).sum()),
            "bias_mean": float(np.mean(
                [r["quantile"]["bias"] for r in valid])),
            "bias_z_mean": float(np.mean(
                [r["quantile"]["bias_z"] for r in valid])),
            "split_half_mean": float(np.mean(
                [r["split_half"]["mean"] for r in valid]))})
    return {"per_class": per_class, "aggregate": agg}


# ----------------------------------------------------------------------
# 验证集: 间隔 vs 云端错误
# ----------------------------------------------------------------------
def _margin_bins(margins, correct, n_bins=5):
    m = np.asarray(margins, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for i in range(n_bins):
        sel = (m > edges[i]) & (m <= edges[i + 1])
        out.append({"bin": i, "lo": float(edges[i]), "hi": float(edges[i + 1]),
                    "n": int(sel.sum()),
                    "err": float(1 - c[sel].mean()) if sel.any() else None,
                    "mean_margin": float(m[sel].mean()) if sel.any() else None})
    return out


def _boot_auc_ci(m, c, n_boot=N_BOOT_CI, seed=0):
    rng = np.random.RandomState(int(seed))
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(m), len(m))
        cc = c[idx]
        if cc.all() or (~cc).all():
            continue
        vals.append(roc_auc_score(cc, m[idx]))
    if not vals:
        return None, None
    return (float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975)))


def _boot_err_diff(low, high, c, n_boot=N_BOOT_CI, seed=0):
    """P(bottom-decile 错误率 <= 中位数以上错误率) 的 bootstrap 估计。"""
    rng = np.random.RandomState(int(seed))
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(c), len(c))
        lo, hi = idx[low[idx]], idx[high[idx]]
        if len(lo) == 0 or len(hi) == 0:
            continue
        diffs.append((1 - c[lo].mean()) - (1 - c[hi].mean()))
    if not diffs:
        return None, None
    return float(np.mean(diffs)), float(np.mean(np.asarray(diffs) <= 0))


def _error_analysis(margins, cloud_correct, edge_pred, labels):
    m = np.asarray(margins, dtype=np.float64)
    c = np.asarray(cloud_correct, dtype=bool)
    y = np.asarray(labels)
    out = {"n": int(len(m)),
           "cloud_acc": float(c.mean() * 100),
           "edge_acc": float((np.asarray(edge_pred) == y).mean() * 100),
           "corr_margin_correct": float(np.corrcoef(m, c.astype(float))[0, 1])
           if c.std() > 0 else None}
    if c.all() or (~c).all():
        return out
    auc = float(roc_auc_score(c, m))
    lo, hi = _boot_auc_ci(m, c)
    q10, q50 = np.quantile(m, [0.1, 0.5])
    low = m <= q10
    high = m >= q50
    diff, p = _boot_err_diff(low, high, c)
    out.update({
        "auc_margin_correct": auc,
        "auc_ci95": [lo, hi],
        "low_decile": {"thr": float(q10), "n": int(low.sum()),
                       "err": float(1 - c[low].mean()),
                       "mean_margin": float(m[low].mean())},
        "top_half": {"thr": float(q50), "n": int(high.sum()),
                     "err": float(1 - c[high].mean()),
                     "mean_margin": float(m[high].mean())},
        "err_diff_low_minus_top": diff,
        "p_err_diff_le_0": p,
        "bins": _margin_bins(m, c)})
    return out


# ----------------------------------------------------------------------
# 代表集更新（与 engine 同轮逻辑） + 当轮云端模型
# ----------------------------------------------------------------------
def _post_update_memory(snap, method, cfg, device):
    memory = RepresentativeMemory.from_state_dict(snap["memory"])
    if int(snap["new_version"]) > int(snap["version"]):
        memory.align(snap["alignment"])

    round_idx = int(snap["round"])
    if method == "dm":
        real_f = torch.cat([snap["rep_feats_reenc"], snap["cur_cls_f"]], 0)
        real_y = torch.cat([snap["rep_labels"], snap["cur_cls_y"]], 0)
        real_src = torch.zeros(len(real_y), dtype=torch.long)
        sdm = StreamingDM(memory.features, memory.labels, memory.quotas,
                          cfg["distillation"], seed=int(snap["seed"]),
                          device=device)
        out = sdm.update(round_idx=round_idx, real_features=real_f,
                         real_labels=real_y, real_sources=real_src)
        memory.features = out["features"]
        info = {**out["info"], "n_target": int(len(real_y))}
    elif method == "online_reservoir_latent":
        if snap.get("latent_reservoir") is None:
            raise RuntimeError("快照缺少 latent_reservoir，请重跑 Stage 1")
        lr = FeatureReservoir.from_state_dict(snap["latent_reservoir"])
        slots = FeatureReservoir.slots(memory.labels,
                                       int(cfg["num_classes"]))
        idx, vals = lr.offer(snap["cur_cls_f"].detach().cpu(),
                             snap["cur_cls_y"].cpu(), slots)
        if len(idx) > 0:
            memory.features[idx] = vals
        info = {"method": "reservoir_latent", "n_replaced": int(len(idx))}
    else:
        raise ValueError(f"未知方法 {method}")

    updated_idx = np.arange(memory.budget)
    memory.update_features(updated_idx, memory.features[updated_idx],
                           int(snap["new_version"]), round_idx)
    return memory, info


def _sync_and_train(snap, cfg, memory, device):
    round_idx = int(snap["round"])
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

    learner = IncrementalCloudLearner(cfg, int(cfg["feature_dim"]),
                                      int(cfg["num_classes"]), device=device)
    learner.load_state_dict(snap["learner"])
    cloud_mode = "incremental" if cfg["cloud"].get(
        "incremental_training", True) else "scratch"
    learner.update(cloud.features, cloud.labels, None, None,
                   round_idx=round_idx, mode=cloud_mode)
    return learner, bool(changed)


# ----------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------
def stage2(cfgs, refs, device, seed=42):
    print("=" * 70)
    print("Stage 2: 间隔分布比较 + 验证集错误分析")
    print("=" * 70, flush=True)
    cfg_dm = cfgs["dm"]
    C = int(cfg_dm["num_classes"])
    stream = _build_stream(cfg_dm)

    out = {"checkpoints": {},
           "meta": {"rounds": list(DIAG_ROUNDS), "seed": seed,
                    "methods": list(METHODS),
                    "definition": {
                        "margin": "h_c(z) - max_{k!=c} h_k(z), 冻结边缘分类头 eval",
                        "real": "当前轮当前样本 + Replay（当轮 encoder 重编码）",
                        "rep_dm": "当轮 Plain DM 目标确定性更新后的 980 个代表",
                        "rep_latent": "当轮 latent reservoir 更新后的 980 个代表",
                        "quantile_match": "代表排序第 i 个 ↔ 真实 (i-0.5)/n 分位",
                        "null_budget": "从真实样本 iid 抽 n_rep 个的 MAD 分布",
                        "split_half": "真实样本按类随机对半分的 MAD"}}}

    for method in METHODS:
        cfg = cfgs[method]
        ckpt_dir = checkpoint_path(
            cfg, f"{cfg['dataset']}/iid/seed{seed}/{_run_tag(method)}")
        print(f"\n[{method}] 快照目录 {ckpt_dir}", flush=True)
        out["checkpoints"][method] = {}
        for t in DIAG_ROUNDS:
            snap_path = os.path.join(ckpt_dir, f"diag_snapshot_r{t}.pt")
            if not os.path.exists(snap_path):
                raise FileNotFoundError(f"缺少快照 {snap_path}，请先运行 Stage 1")
            snap = torch.load(snap_path, map_location="cpu",
                              weights_only=False)
            if int(snap["round"]) != int(t):
                raise ValueError(f"快照轮次不一致: {snap['round']} != {t}")

            vae, protocol = _snapshot_vae(snap, cfg, device)
            memory, dist_info = _post_update_memory(snap, method, cfg, device)

            # 真实样本：当前轮 + Replay
            # reservoir_latent 分支不预计算 rep_feats_reenc，这里用当轮
            # encoder 对快照中的 replay 图像重新编码，保证两组可比。
            if snap.get("rep_feats_reenc") is not None:
                rep_f = snap["rep_feats_reenc"]
            else:
                rep_f = protocol.normalize(
                    encode_mu(vae.encoder, snap["rep_images"], device=device))
            real_feats = torch.cat([rep_f, snap["cur_cls_f"]], 0)
            real_y = torch.cat([snap["rep_labels"], snap["cur_cls_y"]], 0)
            real_m, _ = _classify(real_feats, real_y, vae, protocol, device)
            real_y_np = real_y.numpy()

            # 代表集（更新后）
            rep_m, rep_pred = _classify(memory.features, memory.labels, vae,
                                        protocol, device)
            rep_y_np = memory.labels.numpy()

            cls_report = _class_margin_report(real_m, real_y_np, rep_m,
                                              rep_y_np, C, seed=seed + t)
            rep_edge_acc = float((rep_pred.numpy() == rep_y_np).mean() * 100)

            # 当轮云端模型 → 验证集错误分析
            learner, changed = _sync_and_train(snap, cfg, memory, device)
            val_feats = protocol.normalize(
                encode_mu(vae.encoder, stream.val_images, device=device))
            val_met = learner.evaluate(val_feats, stream.val_labels)
            test_met = learner.evaluate(
                protocol.normalize(encode_mu(vae.encoder, stream.test_images,
                                             device=device)),
                stream.test_labels)
            val_m, val_pred = _classify(val_feats, stream.val_labels, vae,
                                        protocol, device)
            cloud_correct = (val_met["pred"].numpy()
                             == stream.val_labels.numpy())
            err = _error_analysis(val_m, cloud_correct, val_pred,
                                  stream.val_labels.numpy())

            # 边缘分类头自身的间隔 vs 自身错误（同类的"上限"检查）
            edge_correct = val_pred.numpy() == stream.val_labels.numpy()
            edge_err = _error_analysis(val_m, edge_correct, val_pred,
                                       stream.val_labels.numpy())

            out["checkpoints"][method][str(t)] = {
                "round": int(t),
                "encoder_version": int(snap["new_version"]),
                "n_real": int(len(real_y)), "n_rep": int(len(memory.labels)),
                "distill": dist_info,
                "rep_edge_acc": rep_edge_acc,
                "real_margin": _stats(real_m.numpy()),
                "rep_margin": _stats(rep_m.numpy()),
                "class_report": cls_report,
                "val": {"cloud_acc": float(val_met["acc"]),
                        "cloud_error_analysis": err,
                        "edge_error_analysis": edge_err},
                "test_acc": float(test_met["acc"]),
                "sync_changed": changed,
            }
            agg = cls_report["aggregate"]
            print(f"  r{t:3d} real={len(real_y)} rep={len(memory.labels)} "
                  f"MAD={agg.get('mad_mean', float('nan')):.4f} "
                  f"null={agg.get('null_mean', float('nan')):.4f} "
                  f"ratio={agg.get('ratio_median', float('nan')):.2f} "
                  f"exceed95={agg.get('n_exceed_null_p95', -1)}/{C} "
                  f"auc={err.get('auc_margin_correct', float('nan')):.3f} "
                  f"val_acc={val_met['acc']:.2f} test={test_met['acc']:.2f}",
                  flush=True)

    out["judgement"] = _judge(out)
    return out


def _judge(out):
    dm = out["checkpoints"]["dm"]
    agg = [dm[str(t)]["class_report"]["aggregate"] for t in DIAG_ROUNDS]
    ratio_med = float(np.median([a["ratio_median"] for a in agg]))
    n_exceed = [a["n_exceed_null_p95"] for a in agg]
    cond_a = bool(ratio_med >= 1.5 and min(n_exceed) >= 7)

    aucs = [dm[str(t)]["val"]["cloud_error_analysis"].get(
        "auc_margin_correct") for t in DIAG_ROUNDS]
    ps = [dm[str(t)]["val"]["cloud_error_analysis"].get(
        "p_err_diff_le_0") for t in DIAG_ROUNDS]
    aucs_ok = [a for a in aucs if a is not None]
    ps_ok = [p for p in ps if p is not None]
    cond_b = bool(aucs_ok and min(aucs_ok) >= 0.55
                  and ps_ok and max(ps_ok) < 0.05)

    rsv = out["checkpoints"]["online_reservoir_latent"]
    rsv_aucs = [rsv[str(t)]["val"]["cloud_error_analysis"].get(
        "auc_margin_correct") for t in DIAG_ROUNDS]
    rsv_ratios = [rsv[str(t)]["class_report"]["aggregate"].get(
        "ratio_median") for t in DIAG_ROUNDS]
    return {
        "cond_a_rep_gap_exceeds_noise": cond_a,
        "cond_a_detail": {"ratio_median_over_rounds": ratio_med,
                          "n_exceed_null_p95_per_round": n_exceed,
                          "threshold": "median ratio>=1.5 且每轮 >=7/10 类超过 null p95"},
        "cond_b_low_margin_more_cloud_errors": cond_b,
        "cond_b_detail": {"auc_margin_correct": aucs,
                          "bootstrap_p_diff_le_0": ps,
                          "threshold": "每轮 AUC>=0.55 且 p<0.05"},
        "worth_margin_training": bool(cond_a and cond_b),
        "reference_reservoir_latent": {"auc_margin_correct": rsv_aucs,
                                       "ratio_median": rsv_ratios},
        "stop_rule": "只看到间隔误差下降、云端验证准确率无改善就停",
    }


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfgs, refs, repro = {}, {}, {}
    for method in METHODS:
        refs[method] = _load_ref(method)
        cfgs[method] = _cfg_from_ref(refs[method], method)

    if args.skip_stage1:
        print("跳过 Stage 1：复用已有快照（不重复校验复现）", flush=True)
    else:
        for method in METHODS:
            res = stage1(cfgs[method], method, args.seed)
            repro[method] = _check_repro(res, refs[method])
            print(f"  复现对照 {method}: {repro[method]}", flush=True)

    out = stage2(cfgs, refs, device, seed=args.seed)
    out["repro"] = repro
    path = result_path(cfgs["dm"],
                       "cifar10/iid/diag_class_margin_seed42.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 70)
    print("结论:", json.dumps(out["judgement"], ensure_ascii=False, indent=1))
    print(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
