"""Streaming 实验主引擎

单个方法的一次完整因果运行:
  Initial Base VAE → 逐轮 (Online VAE → Alignment → Statistics →
  Streaming DM / Coreset → Sync → Cloud 增量训练 → 评估)

所有方法共享同一数据流、预算、代表集容量与云端训练预算。
"""
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

from vae_bdm.baselines import allocate_quotas
from vae_bdm.quant import Quantizer

from ..cloud.incremental_trainer import IncrementalCloudLearner
from ..cloud.state_sync import CloudState
from ..communication.payload_accounting import (CommunicationLedger,
                                                build_sync_message,
                                                message_size,
                                                select_patch_indices)
from ..communication.quantization import decode_features, encode_features
from ..data.causal_stream import CausalStream
from ..distillation.moment_calibration import streaming_moment_calibrate
from ..distillation.online_coreset import OnlineKMeans, build_coreset
from ..distillation.streaming_dm import StreamingDM
from ..memory.replay_buffer import ClassBalancedReservoir
from ..memory.representative_memory import RepresentativeMemory
from ..memory.running_statistics import StreamingStatistics
from ..models.feature_alignment import (estimate_feature_alignment,
                                        identity_alignment)
from ..models.online_vae import (FeatureProtocol, build_vae, encode_mu,
                                 evaluate_vae, online_vae_update,
                                 snapshot_encoder, train_initial_vae)
from ..config_utils import checkpoint_path, result_path


METHOD_OVERRIDES = {
    # 原版 DM：固定随机 MLP 特征空间中的均值匹配（唯一 DM 目标）
    "dm": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        # 与 K-means / Moment 使用同一套 drift 对齐机制，代表集跨版本继承，
        # 不再每轮用 K-means 重建（避免丢失上一轮蒸馏结果）
        "alignment": {"method": "diagonal_affine"},
        # DM 目标每轮都用当前 encoder 精确重编码 replay+current，
        # 保证更新轮与非更新轮的目标集合一致
        "distillation": {"method": "dm", "update_fraction": 1.0,
                         "exact_target": True},
        "communication": {"mode": "full_sync"}},
    # 在线 K-means：流式质心
    "online_kmeans": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "kmeans"},
        "communication": {"mode": "full_sync"}},
    # 在线矩匹配：解析式均值/方差校准
    "online_moment": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "moment"},
        "communication": {"mode": "full_sync"}},
    # 在线 Reservoir：固定预算的类平衡原始样本缓冲，每轮用当前 Encoder
    # 精确重编码缓冲内容作为代表特征（不做任何质心/优化蒸馏）
    "online_reservoir": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "reservoir"},
        "communication": {"mode": "full_sync"}},
}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# DM 的两个增强变体：只替换特征映射 / 增加潜空间均值项
for _name, _extra in {
        "dm_multi": {"feature_type": "multi_mlp", "num_mappings": 3},
        "dm_residual": {"feature_type": "residual_mlp", "lambda_latent": 1.0},
}.items():
    METHOD_OVERRIDES[_name] = deep_merge(
        METHOD_OVERRIDES["dm"], {"distillation": _extra})

# 诊断用：冻结 Encoder（不做在线 VAE 更新），其余流程保持不变
METHOD_OVERRIDES["dm_freeze"] = deep_merge(
    METHOD_OVERRIDES["dm"], {"vae": {"online_update": False}})
# 诊断用：冻结 Encoder + 关闭 DM 优化（代表集只经跨版本对齐/重编码）
METHOD_OVERRIDES["dm_freeze_nodistill"] = deep_merge(
    METHOD_OVERRIDES["dm"], {"vae": {"online_update": False},
                             "distillation": {"updates_per_class": 0}})

# lr / 分类权重扫描：只改在线阶段，不触发 Initial Base VAE 缓存失效
for _name, _lr in {"g0_res_lr1e3": 1e-3,
                   "g1_res_lr3e4": 3e-4,
                   "g2_res_lr1e4": 1e-4}.items():
    METHOD_OVERRIDES[_name] = deep_merge(
        METHOD_OVERRIDES["dm_residual"], {"vae": {"online_lr": _lr}})

# G3 / G4 使用 G0/G1 中较优的 3e-4：分类权重加倍 / 弱潜空间均值
METHOD_OVERRIDES["g3_res_lr3e4_cls2x"] = deep_merge(
    METHOD_OVERRIDES["dm_residual"],
    {"vae": {"online_lr": 3e-4, "online_cls_lambda": 0.2}})
METHOD_OVERRIDES["g4_res_lr3e4_lat"] = deep_merge(
    METHOD_OVERRIDES["dm_residual"],
    {"vae": {"online_lr": 3e-4}, "distillation": {"lambda_latent": 0.1}})
# G5：G3（3e-4 + 分类权重 ×2）与 G4（弱潜空间均值）的合并
METHOD_OVERRIDES["g5_res_lr3e4_cls2x_lat"] = deep_merge(
    METHOD_OVERRIDES["dm_residual"],
    {"vae": {"online_lr": 3e-4, "online_cls_lambda": 0.2},
     "distillation": {"lambda_latent": 0.1}})

# 公平比较：K-means / Moment 使用与 G5 完全相同的 VAE 配置
for _name, _base in {"kmeans_g5": "online_kmeans",
                     "moment_g5": "online_moment"}.items():
    METHOD_OVERRIDES[_name] = deep_merge(
        METHOD_OVERRIDES[_base],
        {"vae": {"online_lr": 3e-4, "online_cls_lambda": 0.2}})


def _seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 跨进程可复现（不同方法必须共享完全相同的 Initial Base VAE）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _mask_acc(pred, labels, mask):
    if mask is None or int(mask.sum()) == 0:
        return None
    return float((pred[mask] == labels[mask]).float().mean().item() * 100)


def _encoder_sha1(encoder):
    """Encoder 参数指纹：用于核对三种方法是否共享同一 VAE 轨迹"""
    h = hashlib.sha1()
    sd = encoder.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def _stratified_anchor_indices(labels, n_a, num_classes):
    """按类别分层抽取锚点索引（Replay 按类别连续存储）。

    配额按类别样本量分配；某类容量不足时把剩余名额重新分配给仍有余量的
    类别（water-filling），保证锚点数量尽量达到 n_a。
    预算不足以覆盖所有类别时退化为均匀抽样。
    """
    labels = torch.as_tensor(labels).long().flatten()
    n = int(len(labels))
    n_a = min(int(n_a), n)
    if n_a <= 0:
        return torch.empty(0, dtype=torch.long)
    present = [c for c in range(int(num_classes)) if bool((labels == c).any())]
    if not present:
        return torch.empty(0, dtype=torch.long)
    if n_a < len(present):
        return torch.linspace(0, n - 1, n_a).long().unique()
    idx_by_class = [torch.nonzero(labels == c, as_tuple=False).flatten()
                    for c in present]
    counts = torch.tensor([len(x) for x in idx_by_class], dtype=torch.long)
    quota = torch.ones(len(present), dtype=torch.long)
    left = n_a - len(present)
    while left > 0:
        spare = counts - quota                      # 各类还能再分配的名额
        total_spare = int(spare.sum())
        if total_spare <= 0:
            break
        share = spare.double() / total_spare * left
        add = torch.minimum(share.floor().long(), spare)
        if int(add.sum()) == 0:
            # 名额过少时按剩余容量逐个补齐
            add = torch.zeros_like(spare)
            order = torch.argsort(-spare)
            for j in range(min(left, total_spare)):
                add[int(order[j])] += 1
        quota += add
        left -= int(add.sum())
    sel = []
    for i, idx in enumerate(idx_by_class):
        k = int(quota[i])
        sel.append(idx[torch.linspace(0, len(idx) - 1, k).long().unique()])
    return torch.cat(sel)


def _save_vae_ckpt(path, vae, version, tag, seed, round_idx, n_seen, stream):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": vae.state_dict(),
                "encoder_version": int(version), "method": tag,
                "seed": int(seed), "round": int(round_idx),
                "accessible_rounds": f"initial_base + rounds 0..{round_idx}",
                "n_seen": int(n_seen), "stream": stream.summary()}, path)


def _build_stream(cfg):
    s = cfg["stream"]
    return CausalStream(
        dataset=cfg["dataset"],
        initial_fraction=float(s["initial_fraction"]),
        num_rounds=int(s["num_rounds"]),
        seed=int(s["shuffle_seed"]),
        class_balanced_initial=bool(s.get("class_balanced_initial", True)),
        scenario=s.get("scenario", "iid"),
        val_ratio=float(cfg["val_ratio"]),
        split_seed=int(cfg["split_seed"]),
        class_quota=s.get("class_quota"))


def _state_signature(cfg, tag, seed):
    """对完整配置做哈希：任何影响训练结果的参数变化都会使断点失效"""
    c = copy.deepcopy(cfg)
    for k in ("_path", "results_dir", "checkpoints_dir", "figures_dir",
              "_run_tag", "_run_name"):
        c.pop(k, None)
    payload = {"tag": tag, "seed": int(seed), "config": c}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _base_signature(cfg, seed, stream):
    """Initial Base VAE 缓存签名：包含数据划分、架构与训练超参"""
    v = cfg["vae"]
    s = cfg["stream"]
    base_idx = np.asarray(sorted(int(i) for i in stream.base_indices),
                          dtype=np.int64)
    return {
        "cache_version": 3,
        "dataset": cfg["dataset"], "num_classes": int(cfg["num_classes"]),
        "latent_dim": int(cfg["latent_dim"]),
        "base_channels": int(cfg["base_channels"]),
        "seed": int(seed),
        "initial_fraction": float(s["initial_fraction"]),
        "shuffle_seed": int(s["shuffle_seed"]),
        "split_seed": int(cfg["split_seed"]),
        "val_ratio": float(cfg["val_ratio"]),
        "class_balanced_initial": bool(s.get("class_balanced_initial", True)),
        "class_quota": s.get("class_quota"),
        # 配额模式下 base 大小依赖轮数
        "class_quota_rounds": int(s["num_rounds"]) if s.get("class_quota") else None,
        # 直接校验 base 样本索引本身，避免数据划分实现变化后误用旧缓存
        "base_indices_sha1": hashlib.sha1(base_idx.tobytes()).hexdigest()[:16],
        "n_base": int(len(base_idx)),
        "initial_epochs": int(v["initial_epochs"]),
        "batch_size": int(v["batch_size"]),
        "lr": float(v["lr"]), "beta": float(v["beta"]),
        "cls_lambda": float(v["cls_lambda"]),
        "augment": bool(v.get("augment", True)),
    }


def _save_run_state(path, *, vae, version, last_round, protocol, memory,
                    statistics, replay, cloud, learner, coreset, ledger,
                    rounds_log, edge_sec, peak_gpu, signature, wall_sec=0.0,
                    pending_vae=None, pending_rounds=None, reservoir=None):
    torch.save({
        "signature": signature, "vae": vae.state_dict(),
        "version": int(version), "last_round": int(last_round),
        "pending_vae": pending_vae,
        "pending_rounds": list(pending_rounds or []),
        "protocol_mean": protocol.mean.clone(),
        "protocol_std": protocol.std.clone(),
        "memory": memory.state_dict(), "statistics": statistics.state_dict(),
        "replay": replay.state_dict(), "cloud": cloud.state_dict(),
        "learner": learner.state_dict(), "coreset": coreset.state_dict()
        if coreset is not None else None,
        "reservoir": reservoir.state_dict()
        if reservoir is not None else None,
        "ledger": copy.deepcopy(ledger.entries),
        "rounds_log": copy.deepcopy(rounds_log),
        "edge_sec": float(edge_sec), "peak_gpu": float(peak_gpu),
        "wall_sec": float(wall_sec),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
        "np_rng": np.random.get_state(),
    }, path)


def run_method(cfg, method, seed=None, verbose=True, save=True, resume=False,
               stop_after=None):
    if method is not None and method not in METHOD_OVERRIDES:
        raise ValueError(f"未知方法 {method}; 可选: {sorted(METHOD_OVERRIDES)}")
    if method is not None:
        cfg = deep_merge(cfg, METHOD_OVERRIDES[method])
        tag = method
    else:
        cfg = copy.deepcopy(cfg)
        tag = str(cfg.get("_run_name", "custom"))
    if seed is None:
        seed = int(cfg["vae"].get("seed", 42))
    if cfg.get("_run_tag"):
        tag = f"{tag}_{cfg['_run_tag']}"
    cfg["vae"]["seed"] = int(seed)
    _seed_everything(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C = int(cfg["num_classes"])
    dim = int(cfg["feature_dim"])
    budget = int(cfg["memory"]["representative_budget"])
    vcfg = cfg["vae"]
    align_mode = cfg["alignment"]["method"]
    distill = cfg["distillation"]["method"]
    sync_mode = cfg["communication"]["mode"]
    quant_mode = cfg["communication"]["quantization"]
    cloud_mode = "incremental" if cfg["cloud"].get(
        "incremental_training", True) else "scratch"
    patch_fraction = float(cfg["communication"].get("patch_fraction", 0.10))
    update_fraction = cfg["distillation"].get("update_fraction")
    if update_fraction is None:
        # 兼容旧配置：synced_only 表示更新子集 = 上传子集
        if sync_mode == "patch" and cfg["distillation"].get(
                "patch_optimize", "synced_only") == "synced_only":
            update_fraction = patch_fraction
        else:
            update_fraction = 1.0
    update_fraction = float(update_fraction)
    if not 0.0 < update_fraction <= 1.0:
        raise ValueError(f"update_fraction 必须在 (0,1]，实际 {update_fraction}")
    vae_update = bool(vcfg.get("online_update", True))
    # 累积若干轮再做一次在线更新：小批量流式场景下避免反复看极少数样本
    update_every = max(1, int(vcfg.get("update_every_rounds", 1)))
    replay_enabled = bool(vcfg.get("replay_enabled", True))
    anchor_count = int(cfg["alignment"].get("anchor_count", 200))
    anchor_count = min(anchor_count, int(cfg["memory"]["raw_replay_capacity"]))
    # 历史预算口径: full=累计全部历史（base+每轮）；replay=固定预算 replay+current
    history_mode = str(cfg["stream"].get("history_mode", "full"))
    if history_mode not in ("full", "replay"):
        raise ValueError(f"history_mode 只能是 full / replay，实际 {history_mode}")
    # DM 精确目标：每轮都用当前 encoder 重编码 replay+current 作为同一目标集合
    dm_exact = distill == "dm" and bool(
        cfg["distillation"].get("exact_target", False))
    signature = _state_signature(cfg, tag, seed)

    stream = _build_stream(cfg)
    base_sig = _base_signature(cfg, seed, stream)
    ckpt0 = checkpoint_path(
        cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/vae_v0.pt")
    if verbose:
        print(f"[{tag}] {cfg['dataset']} {stream.scenario} "
              f"base={len(stream.base_indices)} rounds={stream.num_rounds} "
              f"budget={budget} align={align_mode} distill={distill} "
              f"sync={sync_mode} quant={quant_mode} seed={seed}", flush=True)

    t_start = time.time()
    base = stream.base
    state_path = checkpoint_path(
        cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/stream_state.pt")
    # 先探测断点：命中时跳过 base 训练、初始同步与初始云端训练
    st = None
    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["signature"] != signature:
            raise ValueError("断点状态与当前配置签名不一致，拒绝续跑")
    resuming = st is not None

    vae = build_vae(C, int(cfg["latent_dim"]), int(cfg["base_channels"]),
                    seed=int(seed), device=device)
    if resuming:
        vae.load_state_dict(st["vae"])
        base_log = copy.deepcopy(st["rounds_log"][0].get("base_vae_log", {}))
    else:
        base_hash = hashlib.sha1(
            json.dumps(base_sig, sort_keys=True).encode()).hexdigest()[:16]
        base_ckpt = os.path.join(cfg["checkpoints_dir"], cfg["dataset"],
                                 "base_vae",
                                 f"base_{cfg['dataset']}_{base_hash}.pt")
        cached = False
        if bool(vcfg.get("cache_base", True)) and os.path.exists(base_ckpt):
            obj = torch.load(base_ckpt, map_location=device, weights_only=True)
            if obj.get("base_signature") == base_sig:
                vae.load_state_dict(obj["model_state_dict"])
                base_log = obj["base_log"]
                base_log["cached"] = True
                cached = True
                if verbose:
                    print(f"  [base VAE] 复用缓存 "
                          f"{os.path.relpath(base_ckpt, os.getcwd())}",
                          flush=True)
        if not cached:
            base_log = train_initial_vae(vae, base.images, base.labels, cfg,
                                         device, verbose=verbose)
            os.makedirs(os.path.dirname(base_ckpt), exist_ok=True)
            torch.save({"model_state_dict": vae.state_dict(),
                        "base_log": base_log, "base_signature": base_sig,
                        "seed": int(seed),
                        "accessible": "initial_base_only"}, base_ckpt)
    vae.eval()

    quotas = allocate_quotas(base.labels.numpy(), budget, C)
    peak_gpu = 0.0
    if resuming:
        protocol = FeatureProtocol(st["protocol_mean"], st["protocol_std"])
        memory = RepresentativeMemory.from_state_dict(st["memory"])
        stats = StreamingStatistics.from_state_dict(st["statistics"])
        replay = ClassBalancedReservoir.from_state_dict(st["replay"])
        cloud = CloudState.from_state_dict(st["cloud"])
        learner = IncrementalCloudLearner(cfg, dim, C, device=device)
        learner.load_state_dict(st["learner"])
        ledger = CommunicationLedger()
        ledger.entries = copy.deepcopy(st["ledger"])
        if distill == "kmeans":
            if st.get("coreset") is None:
                raise ValueError("断点缺少 kmeans coreset 状态")
            coreset = OnlineKMeans.from_state_dict(st["coreset"])
        else:
            coreset = None
        if distill == "reservoir":
            if st.get("reservoir") is None:
                raise ValueError("断点缺少 reservoir 状态")
            reservoir = ClassBalancedReservoir.from_state_dict(st["reservoir"])
        else:
            reservoir = None
        rounds_log = copy.deepcopy(st["rounds_log"])
        pending_vae = st.get("pending_vae")
        pending_rounds = [int(r) for r in (st.get("pending_rounds") or [])]
        if pending_vae is not None and not pending_rounds:
            # 旧断点无轮次记录：按 update_every 反推
            k = (int(st["last_round"]) + 1) % update_every or update_every
            pending_rounds = list(range(int(st["last_round"]) - k + 1,
                                        int(st["last_round"]) + 1))
        edge_sec = float(st["edge_sec"])
        wall_sec = float(st.get("wall_sec", 0.0))
        version = int(st["version"])
        start_round = int(st["last_round"]) + 1
        stream._base_accessed = True
        stream._max_accessed = int(st["last_round"])
        # 让在线阶段的随机性独立于 base 权重是否来自缓存
        _seed_everything(int(seed) + 1_000_003)
        torch.set_rng_state(st["torch_rng"])
        if device == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        peak_gpu = max(peak_gpu, float(st["peak_gpu"]))
        vae.eval()
        if verbose:
            print(f"  [resume] 从第 {start_round} 轮继续 "
                  f"(已完成 {start_round}/{stream.num_rounds})", flush=True)
    else:
        base_raw = encode_mu(vae.encoder, base.images, device=device)
        protocol = FeatureProtocol.fit(base_raw)
        base_feats = protocol.normalize(base_raw)

        memory = RepresentativeMemory(budget, C, quotas, dim)
        memory.initialize_from_features(
            base_feats, base.labels,
            method=cfg["memory"].get("representative_init", "kmeans"),
            version=0, seed=int(seed), device=device,
            cfg=cfg["distillation"])

        stats = StreamingStatistics(C, dim, version=0)
        stats.update(base_feats, base.labels, 0, round_idx=-1)

        replay = ClassBalancedReservoir(
            int(cfg["memory"]["raw_replay_capacity"]), C,
            class_balanced=cfg["memory"].get("replay_strategy",
                                             "class_balanced") == "class_balanced",
            seed=int(seed))
        replay.add_batch(base.images, base.labels)

        coreset = None
        if distill == "kmeans":
            coreset = build_coreset(distill, memory.features, memory.labels,
                                    quotas, dim=dim, seed=int(seed))

        # Online Reservoir 基线：与代表集同预算的类平衡原始样本缓冲，
        # 初始内容取自 Initial Base Set，代表特征由当前 Encoder 精确重编码
        reservoir = None
        if distill == "reservoir":
            reservoir = ClassBalancedReservoir(
                budget, C, class_balanced=True, seed=int(seed) + 13)
            reservoir.caps = [int(q) for q in quotas]
            reservoir.add_batch(base.images, base.labels)
            r_imgs, r_labels = reservoir.get()
            if len(r_labels) != budget or not torch.equal(r_labels,
                                                          memory.labels):
                raise RuntimeError(
                    f"Reservoir 初始内容与代表集不一致: "
                    f"n={len(r_labels)} budget={budget}")
            memory.features = protocol.normalize(
                encode_mu(vae.encoder, r_imgs, device=device)).clone()

        cloud = CloudState.empty(budget, dim)
        learner = IncrementalCloudLearner(cfg, dim, C, device=device)
        ledger = CommunicationLedger()

        # ---------------- 初始完整同步 ----------------
        msg, _ = build_sync_message("full_sync", -1, memory, None, quant_mode,
                                    patch_fraction)
        size = message_size(msg)
        ledger.add("initial_sync", -1, size)
        cloud.apply(msg)
        header = json.loads(msg["meta"])
        decoded = decode_features(msg["features"], header["quant"])
        memory.set_submitted(np.arange(budget), decoded, 0)
        init_verify = cloud.verify_consistent(memory)
        if not init_verify["all_ok"]:
            raise RuntimeError(f"初始同步不一致: {init_verify}")

        if save:
            _save_vae_ckpt(ckpt0, vae, 0, tag, seed, -1,
                           int(len(base.labels)), stream)

        val_feats = protocol.normalize(
            encode_mu(vae.encoder, stream.val_images, device=device))
        test_feats = protocol.normalize(
            encode_mu(vae.encoder, stream.test_images, device=device))
        cloud_init = learner.update(cloud.features, cloud.labels, val_feats,
                                    stream.val_labels, round_idx=-1,
                                    mode=cloud_mode)
        init_metrics = learner.evaluate(test_feats, stream.test_labels)
        if device == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated() / 1e9

        initial_record = {
            "round": -1, "encoder_version": 0,
            "base_vae_log": base_log,
            "feat_norm_protocol": protocol.meta(),
            "memory": memory.summary(stats),
            "statistics": stats.summary(),
            "cloud": cloud_init,
            "acc": init_metrics["acc"], "f1": init_metrics["f1"],
            "sync_bytes": size,
            "sync_verify": init_verify,
        }
        rounds_log = [initial_record]
        edge_sec = base_log["train_sec"]
        # 让在线阶段的随机性独立于 base 权重是否来自缓存
        _seed_everything(int(seed) + 1_000_003)

        version = 0
        start_round = 0
        wall_sec = 0.0
        pending_vae = None
        pending_rounds = []
        if save:
            _save_run_state(state_path, vae=vae, version=version, last_round=-1,
                            protocol=protocol, memory=memory, statistics=stats,
                            replay=replay, cloud=cloud, learner=learner,
                            coreset=coreset, ledger=ledger, rounds_log=rounds_log,
                            edge_sec=edge_sec, peak_gpu=peak_gpu,
                            signature=signature,
                            wall_sec=wall_sec + (time.time() - t_start),
                            pending_rounds=pending_rounds,
                            reservoir=reservoir)

    # 统计来源（跨轮保持）：只有真正发生 reset 才降级，不因当前轮未更新而误标
    stats_origin = "aligned_full_history"
    if resuming and rounds_log:
        stats_origin = rounds_log[-1].get("stats_scope", stats_origin)
    ck = ckpt0
    for t in range(start_round, stream.num_rounds):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr = time.time()
        current = stream.round(t)
        prev_encoder = snapshot_encoder(vae, version)

        # 累积本轮样本；每 update_every 轮做一次在线更新。
        # 末轮强制 flush：392 轮下余数轮（2 轮）的样本也必须进入一次更新，
        # 否则这部分数据不参与任何 VAE 梯度（batch 会小于常规更新）。
        if vae_update:
            if pending_vae is None:
                pending_vae = (current.images.clone(), current.labels.clone())
            else:
                pending_vae = (torch.cat([pending_vae[0], current.images], 0),
                               torch.cat([pending_vae[1], current.labels], 0))
            pending_rounds.append(int(t))
        final_flush = bool(vae_update and (t + 1) % update_every != 0
                           and t == stream.num_rounds - 1)
        do_update = vae_update and ((t + 1) % update_every == 0 or final_flush)

        # ---------------- Online VAE ----------------
        vae_sec = 0.0
        n_update_batch = int(len(pending_vae[1])) \
            if (do_update and pending_vae is not None) else 0
        vlog = {"skipped": True, "update_every_rounds": update_every}
        rep_imgs, rep_labels = replay.get()   # DM / 锚点：允许含当前窗口
        if do_update:
            # 历史 Replay 必须排除仍在 pending 中的轮次，否则同一批样本会
            # 同时作为“当前数据”和“replay”进入同一个训练 batch
            if replay_enabled and pending_rounds:
                vae_rep_imgs, vae_rep_labels = replay.get(
                    exclude_rounds=pending_rounds)
            else:
                vae_rep_imgs, vae_rep_labels = rep_imgs, rep_labels
            if not replay_enabled:
                vae_rep_imgs = torch.empty(0, 3, 32, 32, dtype=torch.uint8)
                vae_rep_labels = torch.empty(0, dtype=torch.long)
            # 独立随机流：在线更新（含 torchvision 增强）不得推进全局 RNG，
            # 否则三种方法会得到不同的后续 VAE 随机轨迹
            rng_seed = int(seed) + 1_000_003 + 7919 * t
            rng_devs = [] if device == "cpu" else [torch.cuda.current_device()]
            with torch.random.fork_rng(devices=rng_devs):
                torch.manual_seed(rng_seed)
                if device != "cpu":
                    torch.cuda.manual_seed(rng_seed)
                vlog = online_vae_update(vae, pending_vae[0], pending_vae[1],
                                         vae_rep_imgs, vae_rep_labels,
                                         prev_encoder, cfg,
                                         device=device, round_idx=t)
            vlog["update_every_rounds"] = update_every
            vlog["final_flush"] = bool(final_flush)
            vlog["n_accumulated"] = n_update_batch
            vlog["encoder_sha1"] = _encoder_sha1(vae.encoder)
            vae_sec = vlog["train_sec"]
            edge_sec += vae_sec
            pending_vae = None
            pending_rounds = []
        vae.eval()
        new_version = version + 1 if do_update else version

        # ---------------- Anchors / Drift / Alignment ----------------
        anchors_img, anchors_y = replay.get()
        if len(anchors_img) == 0:
            anchors_img, anchors_y = current.images, current.labels
        n_a = min(anchor_count, len(anchors_img))
        # 类别分层抽样，保证每个有样本的类别都被锚点覆盖
        sel = _stratified_anchor_indices(anchors_y, n_a, C)
        anchors_img, anchors_y = anchors_img[sel], anchors_y[sel]
        n_a = len(sel)
        ta = time.time()
        if n_a == 0:
            anchor_drift = {"mse": 0.0, "relative": 0.0,
                            "relative_norm": 0.0, "cosine": 1.0,
                            "n_anchors": 0}
            alignment = identity_alignment(dim, version, new_version)
            align_sec = time.time() - ta
        else:
            raw_old = encode_mu(prev_encoder, anchors_img, device=device)
            raw_new = encode_mu(vae.encoder, anchors_img, device=device)
            z_old = raw_old.flatten(1)
            z_new = raw_new.flatten(1)
            n_old = protocol.normalize(raw_old)
            n_new = protocol.normalize(raw_new)
            anchor_drift = {
                "mse": float((z_new - z_old).pow(2).mean()),
                "relative": float((z_new - z_old).pow(2).mean() /
                                  z_new.pow(2).mean().clamp_min(1e-8)),
                "relative_norm": float((n_new - n_old).pow(2).mean() /
                                       n_new.pow(2).mean().clamp_min(1e-8)),
                "cosine": float(torch.nn.functional.cosine_similarity(
                    z_old, z_new, dim=1).mean()),
                "n_anchors": int(n_a)}
            if do_update and align_mode == "diagonal_affine":
                alignment = estimate_feature_alignment(
                    prev_encoder, vae.encoder, anchors_img, anchors_y,
                    version_from=version, version_to=new_version,
                    max_scale=float(cfg["alignment"].get("max_scale", 3.0)),
                    method="diagonal_affine", device=device, protocol=protocol)
            else:
                alignment = identity_alignment(dim, version, new_version)
            align_sec = time.time() - ta

        # ---------------- 当前轮特征 ----------------
        te = time.time()
        cur_raw = encode_mu(vae.encoder, current.images, device=device)
        cur_feats = protocol.normalize(cur_raw)
        # DM 精确目标 / history_mode=replay / 旧 reencode 模式都需要用当前
        # encoder 重编码历史 replay，保证目标集合与代表特征处于同一版本
        need_rep_now = dm_exact or history_mode == "replay" or (
            do_update and align_mode == "reencode")
        rep_feats_reenc = None
        if need_rep_now and len(rep_imgs) > 0:
            rep_feats_reenc = protocol.normalize(
                encode_mu(vae.encoder, rep_imgs, device=device))
        encode_sec = time.time() - te

        # ---------------- 统计记忆 ----------------
        ts = time.time()
        if not do_update:
            stats.update(cur_feats, current.labels, version, round_idx=t)
        elif align_mode == "diagonal_affine":
            stats.align_to(new_version, alignment)
            stats.update(cur_feats, current.labels, new_version, round_idx=t)
        elif align_mode == "reencode":
            stats.reset(new_version)
            all_f = torch.cat([rep_feats_reenc, cur_feats], 0) \
                if rep_feats_reenc is not None else cur_feats
            all_y = torch.cat([rep_labels, current.labels], 0) \
                if rep_feats_reenc is not None else current.labels
            stats.update(all_f, all_y, new_version, round_idx=t)
            stats_origin = "replay_plus_current"
        else:  # none: 跨版本直接混用是禁止的，只能丢弃历史统计
            stats.reset(new_version)
            stats.update(cur_feats, current.labels, new_version, round_idx=t)
            stats_origin = "current_only"
        if history_mode == "replay" and not (do_update and
                                             align_mode == "reencode"):
            # 消融：与 DM / K-means 相同的固定历史预算（replay + current），
            # 而非累计全部历史样本
            stats.reset(new_version)
            all_f = torch.cat([rep_feats_reenc, cur_feats], 0) \
                if rep_feats_reenc is not None else cur_feats
            all_y = torch.cat([rep_labels, current.labels], 0) \
                if rep_feats_reenc is not None else current.labels
            stats.update(all_f, all_y, new_version, round_idx=t)
            stats_origin = "replay_plus_current"
        stats_sec = time.time() - ts

        # ---------------- 代表集更新 ----------------
        patch_idx = select_patch_indices(memory.quotas, patch_fraction, t) \
            if sync_mode == "patch" else None
        # 更新预算与上传预算相互独立：
        #   100% 更新 + 100% 上传 -> 标准全同步
        #   100% 更新 + 10%  上传 -> 本地领先云端（显式记录 local_unsynced）
        #   10%  更新 + 100% 上传 -> 全量上传，边缘/云端一致
        #   10%  更新 + 10%  上传 -> 更新/上传同一子集，严格一致
        if update_fraction >= 1.0:
            opt_idx = None
        else:
            # 优化子集与上传子集相互独立：比例不同时必须各自抽样
            opt_idx = select_patch_indices(memory.quotas, update_fraction, t)
        updated_idx = np.arange(budget) if opt_idx is None else opt_idx
        td = time.time()
        if do_update and align_mode == "reencode":
            all_f = torch.cat([rep_feats_reenc, cur_feats], 0) \
                if rep_feats_reenc is not None else cur_feats
            all_y = torch.cat([rep_labels, current.labels], 0) \
                if rep_feats_reenc is not None else current.labels
            init_method = cfg["memory"].get("representative_init", "kmeans")
            memory.reinitialize_from_features(
                all_f, all_y,
                method=init_method,
                version=new_version, round_idx=t, seed=int(seed),
                device=device, cfg=cfg["distillation"])
            if distill == "dm":
                sdm = StreamingDM(memory.features, memory.labels,
                                  memory.quotas, cfg["distillation"],
                                  seed=int(seed), device=device)
                out = sdm.update(allowed_idx=opt_idx, round_idx=t,
                                 real_features=all_f, real_labels=all_y)
                memory.features = out["features"]
                distill_info = {**out["info"],
                                "init": f"reencode_{init_method}",
                                "target_scope": "replay_plus_current",
                                "n_target": int(len(all_y))}
            else:
                distill_info = {"method": "reencode"}
        else:
            if new_version > version:
                memory.align(alignment)
            if distill == "dm":
                if rep_feats_reenc is not None:
                    real_f = torch.cat([rep_feats_reenc, cur_feats], 0)
                    real_y = torch.cat([rep_labels, current.labels], 0)
                    target_scope = "replay_plus_current"
                else:
                    real_f, real_y = cur_feats, current.labels
                    target_scope = "current_only"
                sdm = StreamingDM(memory.features, memory.labels,
                                  memory.quotas, cfg["distillation"],
                                  seed=int(seed), device=device)
                out = sdm.update(allowed_idx=opt_idx, round_idx=t,
                                 real_features=real_f, real_labels=real_y)
                memory.features = out["features"]
                distill_info = {**out["info"], "target_scope": target_scope,
                                "n_target": int(len(real_y))}
            elif distill == "kmeans":
                if new_version > version:
                    coreset.align(alignment)
                res = coreset.update(cur_feats, current.labels, round_idx=t,
                                     allowed_idx=opt_idx)
                memory.features = res["features"].clone()
                distill_info = res["info"]
            elif distill == "moment":
                feats, minfo = streaming_moment_calibrate(
                    memory.features, memory.labels, stats,
                    allowed_idx=opt_idx, round_idx=t,
                    max_scale=float(cfg["distillation"].get("max_scale", 3.0)))
                memory.features = feats
                distill_info = minfo
            elif distill == "reservoir":
                reservoir.add_batch(current.images, current.labels,
                                    round_idx=t)
                r_imgs, r_labels = reservoir.get()
                if len(r_labels) != budget:
                    raise RuntimeError(
                        f"Reservoir 内容不足: {len(r_labels)} < {budget}")
                # 用当前 Encoder 精确重编码缓冲内容：无质心/无梯度优化
                memory.features = protocol.normalize(
                    encode_mu(vae.encoder, r_imgs, device=device)).clone()
                distill_info = {"method": "reservoir",
                                "n_buffer": int(len(r_labels)),
                                "per_class": reservoir.per_class_counts(),
                                "target_scope": "reservoir_exact"}
            else:
                raise ValueError(f"未知蒸馏方法: {distill}")
            memory.update_features(updated_idx, memory.features[updated_idx],
                                   new_version, t)
        distill_sec = time.time() - td
        edge_sec += distill_sec + encode_sec + align_sec + stats_sec

        # ---------------- Replay ----------------
        if replay_enabled:
            replay.add_batch(current.images, current.labels, round_idx=t)
        replay.assert_capacity()

        # ---------------- 同步 ----------------
        ts = time.time()
        msg, _ = build_sync_message(sync_mode, t, memory, alignment,
                                    quant_mode, patch_fraction, patch_idx)
        size = message_size(msg)
        ledger.add("round_sync", t, size)
        changed = cloud.apply(msg)
        header = json.loads(msg["meta"])
        dec = decode_features(msg["features"], header["quant"])
        if sync_mode == "full_sync":
            memory.set_submitted(np.arange(budget), dec, new_version)
        else:
            memory.set_submitted(patch_idx, dec, new_version)
        verify = cloud.verify_consistent(memory)
        if not verify["all_ok"]:
            raise RuntimeError(f"第 {t} 轮同步不一致: {verify}")
        if patch_idx is not None and opt_idx is None:
            mask = torch.ones(budget, dtype=torch.bool)
            mask[torch.from_numpy(np.asarray(patch_idx))] = False
            d = (memory.features[mask] - memory.submitted[mask]).abs()
            local_unsynced = {"max": float(d.max()), "mean": float(d.mean()),
                              "n": int(mask.sum())}
        else:
            local_unsynced = {"max": 0.0, "mean": 0.0, "n": 0}
        sync_sec = time.time() - ts

        # ---------------- 云端训练与评估 ----------------
        te = time.time()
        val_feats = protocol.normalize(
            encode_mu(vae.encoder, stream.val_images, device=device))
        test_feats = protocol.normalize(
            encode_mu(vae.encoder, stream.test_images, device=device))
        eval_encode_sec = time.time() - te
        cloud_info = learner.update(cloud.features, cloud.labels, val_feats,
                                    stream.val_labels, round_idx=t,
                                    mode=cloud_mode)
        met = learner.evaluate(test_feats, stream.test_labels)

        # 历史域 / 当前域
        group_metrics = {}
        if stream.scenario == "covariate_shift":
            level = stream.round_domain_level(t)
            dom_x, dom_y = stream.test_domain(level)
            te = time.time()
            dom_f = protocol.normalize(
                encode_mu(vae.encoder, dom_x, device=device))
            eval_encode_sec += time.time() - te
            dm_met = learner.evaluate(dom_f, dom_y)
            group_metrics["current_domain"] = {"acc": dm_met["acc"],
                                               "f1": dm_met["f1"],
                                               "level": level}
            group_metrics["historical_domain"] = {"acc": met["acc"],
                                                  "f1": met["f1"],
                                                  "level": 0.0}
        elif stream.scenario == "label_shift":
            groups = stream.scenario_groups(t)
            pred = met["pred"]
            for gname, gmask in groups.items():
                group_metrics[gname] = {
                    "acc": _mask_acc(pred, stream.test_labels, gmask),
                    "n": int(gmask.sum())}

        test_int8 = None
        int8_sec = 0.0
        if cfg.get("evaluation", {}).get("test_int8", True):
            tq0 = time.time()
            q = Quantizer("int8").fit(memory.features)
            tq = q.dequantize(q.quantize(test_feats))
            test_int8 = learner.evaluate(tq, stream.test_labels)["acc"]
            int8_sec = time.time() - tq0
        stats_scope = stats_origin

        vm = evaluate_vae(vae, stream.val_images, stream.val_labels, cfg,
                          device=device) if do_update else None
        mem_summary = memory.summary(stats)
        peak = torch.cuda.max_memory_allocated() / 1e9 \
            if device == "cuda" else 0.0
        peak_gpu = max(peak_gpu, peak)

        rec = {
            "round": t, "encoder_version": new_version,
            "vae": vlog, "vae_sec": vae_sec,
            "vae_updated": bool(do_update),
            "n_vae_batch": n_update_batch,
            "n_vae_pending": int(len(pending_vae[1])) if pending_vae else 0,
            "anchor_drift": anchor_drift,
            "alignment": alignment.meta() if align_mode != "reencode" else
            {"method": "reencode"},
            "align_sec": align_sec,
            "statistics": stats.summary(),
            "stats_scope": stats_scope,
            "history_mode": history_mode,
            "stats_sec": stats_sec,
            "n_current": int(len(current.labels)),
            "n_cumulative_seen": int(len(base.labels) + sum(
                len(stream.round_indices[r]) for r in range(t + 1))),
            "n_effective_stats": int(stats.n_seen),
            "distill": distill_info, "distill_sec": distill_sec,
            "encode_sec": encode_sec,
            "eval_encode_sec": eval_encode_sec,
            "sync": {"mode": sync_mode, "bytes": size,
                     "verify": verify, "applied": bool(changed),
                     "sec": sync_sec,
                     "align_sent": sync_mode == "patch",
                     "update_fraction": update_fraction,
                     "n_updated": int(len(updated_idx)),
                     "n_uploaded": int(len(patch_idx)) if patch_idx is not None
                     else int(budget),
                     "local_unsynced": local_unsynced},
            "cloud": cloud_info,
            "acc": met["acc"], "f1": met["f1"],
            "groups": group_metrics,
            "test_int8_acc": test_int8,
            "test_int8_sec": int8_sec,
            "test_int8_note": "测试特征用本轮代表集的全局 INT8 量化器；"
                              "云端训练用的是逐轮同步消息的反量化特征",
            "vae_val": vm,
            "memory": mem_summary,
            "replay_bytes": replay.nbytes(),
            "n_replay": len(replay),
            "peak_gpu_gb": peak, "round_sec": time.time() - tr,
        }
        rounds_log.append(rec)

        # 只在 Encoder 真正更新时写盘，避免轮次间用同一版本号反复覆盖
        n_seen_so_far = int(len(base.labels) + sum(
            len(stream.round_indices[r]) for r in range(t + 1)))
        if save and do_update:
            ck = checkpoint_path(
                cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/"
                     f"vae_v{new_version}.pt")
            _save_vae_ckpt(ck, vae, new_version, tag, seed, t, n_seen_so_far,
                           stream)

        if verbose:
            print(f"  r{t} v{new_version} acc {met['acc']:.2f} "
                  f"f1 {met['f1']:.2f} | drift {anchor_drift['relative']:.4f} "
                  f"align {alignment.relative_err:.4f} | "
                  f"stats n={stats.n_seen} | {sync_mode} "
                  f"{size['theoretical_bytes']/1024:.1f}KB | "
                  f"cloud val {cloud_info['best_val_acc']} | {rec['round_sec']:.1f}s",
                  flush=True)
        version = new_version
        if save:
            _save_run_state(state_path, vae=vae, version=version,
                            last_round=t, protocol=protocol, memory=memory,
                            statistics=stats, replay=replay, cloud=cloud,
                            learner=learner, coreset=coreset, ledger=ledger,
                            rounds_log=rounds_log, edge_sec=edge_sec,
                            peak_gpu=peak_gpu, signature=signature,
                            wall_sec=wall_sec + (time.time() - t_start),
                            pending_vae=pending_vae,
                            pending_rounds=pending_rounds,
                            reservoir=reservoir)
        if stop_after is not None and t >= int(stop_after):
            break

    # 按实际完成轮次统计（stop_after / 提前 break 时不再假设跑满）
    final_round = int(rounds_log[-1]["round"])
    completed_rounds = final_round + 1
    partial = completed_rounds < stream.num_rounds
    n_seen_final = int(len(base.labels) + sum(
        len(stream.round_indices[r]) for r in range(completed_rounds)))
    pending_leftover = int(len(pending_vae[1])) if pending_vae else 0
    if save:
        # 部分运行（stop_after / 中断）不得命名为 final，避免被当成完整结果
        ck_name = "vae_partial.pt" if partial else "vae_final.pt"
        ck = checkpoint_path(
            cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/"
                 f"{ck_name}")
        _save_vae_ckpt(ck, vae, version, tag, seed, final_round, n_seen_final,
                       stream)
    accs = [r["acc"] for r in rounds_log[1:]]
    f1s = [r["f1"] for r in rounds_log[1:]]
    final = rounds_log[-1]
    # 时间拆账：边缘计算 / 评估 / 云端训练 / 总墙钟互不混淆
    rl = rounds_log[1:]
    edge_vae_sec = float(sum(r.get("vae_sec", 0.0) or 0.0 for r in rl))
    edge_encode_sec = float(sum(r.get("encode_sec", 0.0) or 0.0 for r in rl))
    edge_align_sec = float(sum(r.get("align_sec", 0.0) or 0.0 for r in rl))
    edge_stats_sec = float(sum(r.get("stats_sec", 0.0) or 0.0 for r in rl))
    edge_distill_sec = float(sum(r.get("distill_sec", 0.0) or 0.0 for r in rl))
    edge_sync_sec = float(sum((r.get("sync") or {}).get("sec", 0.0) for r in rl))
    eval_encode_sec = float(sum(r.get("eval_encode_sec", 0.0) or 0.0 for r in rl))
    int8_eval_sec = float(sum(r.get("test_int8_sec", 0.0) or 0.0 for r in rl))
    base_train_sec = float(rounds_log[0].get("base_vae_log", {}).get(
        "train_sec", 0.0))
    edge_compute_sec = (base_train_sec + edge_vae_sec + edge_encode_sec +
                        edge_align_sec + edge_stats_sec + edge_distill_sec +
                        edge_sync_sec)
    wall_total_sec = wall_sec + (time.time() - t_start)
    summary = {
        "method": tag, "seed": int(seed),
        "partial": bool(partial),
        "completed_rounds": int(completed_rounds),
        "dataset": cfg["dataset"], "scenario": stream.scenario,
        "num_rounds": stream.num_rounds,
        "class_quota": stream.class_quota,
        "update_every_rounds": update_every,
        "history_mode": history_mode,
        "vae_steps_per_round": int(vcfg.get("steps_per_round", 0)),
        "initial_acc": rounds_log[0]["acc"],
        "final_acc": final["acc"], "final_f1": final["f1"],
        "avg_online_acc": float(np.mean(accs)) if accs else None,
        "avg_online_f1": float(np.mean(f1s)) if f1s else None,
        "acc_curve": accs, "f1_curve": f1s,
        "final_test_int8_acc": final.get("test_int8_acc"),
        "initial_upload_bytes": ledger.init_bytes,
        "cumulative_upload_bytes": ledger.cumulative_bytes,
        "cumulative_file_bytes": ledger.cumulative_file_bytes,
        "sync_mode": sync_mode, "quantization": quant_mode,
        "update_fraction": update_fraction,
        "patch_fraction": patch_fraction if sync_mode == "patch" else 1.0,
        "n_reps": budget, "dim": dim,
        # 口径区分：实际已见样本 / 统计有效样本 / Replay 样本
        "n_train_used": int(n_seen_final),
        "cumulative_seen_samples": int(n_seen_final),
        "effective_stats_samples": int(stats.n_seen),
        "replay_samples_final": int(len(replay)),
        "vae_pending_leftover": int(pending_leftover),
        "stats_scope": final.get("stats_scope"),
        "vae_ckpt": ck,
        "vae_accessible": f"initial_base + rounds 0..{final['round']}",
        "base_cache_signature": base_sig,
        "raw_replay_capacity": int(cfg["memory"]["raw_replay_capacity"]),
        "replay_bytes": replay.nbytes(),
        "edge_base_train_sec": base_train_sec,
        "edge_vae_update_sec": edge_vae_sec,
        "edge_encode_sec": edge_encode_sec,
        "edge_align_sec": edge_align_sec,
        "edge_stats_sec": edge_stats_sec,
        "edge_distill_sec": edge_distill_sec,
        "edge_sync_sec": edge_sync_sec,
        # edge_train_sec 只含 VAE 训练；edge_total_sec 为边缘侧计算总耗时
        "edge_train_sec": base_train_sec + edge_vae_sec,
        "edge_total_sec": edge_compute_sec,
        "eval_encode_sec_total": eval_encode_sec,
        "int8_eval_sec_total": int8_eval_sec,
        "eval_total_sec": eval_encode_sec + int8_eval_sec,
        "cloud_total_sec": learner.total_train_sec,
        "wall_total_sec": wall_total_sec,
        "peak_gpu_gb": peak_gpu,
        "stats_seen_final": int(stats.n_seen),
        "expected_seen_final": int(n_seen_final),
        "final_alignment_err": final.get("alignment", {}).get("relative_err"),
        "final_no_align_heldout_mse": final.get("alignment", {}).get(
            "no_align_heldout_mse"),
        "final_alignment_improvement": final.get("alignment", {}).get(
            "improvement_heldout"),
        "final_class_center_improvement": final.get("alignment", {}).get(
            "class_center_improvement"),
        "final_fit_class_center_err": final.get("alignment", {}).get(
            "fit_class_center_err"),
        "final_heldout_class_center_err": final.get("alignment", {}).get(
            "class_center_err"),
        "final_group_var_ratio": final["memory"].get("group_single_var_ratio"),
        "final_class_mean_distance": final["memory"].get("class_mean_distance"),
        "final_local_unsynced_max": final.get("sync", {}).get(
            "local_unsynced", {}).get("max"),
        "test_int8_note": final.get("test_int8_note"),
        "ckpt_initial": ckpt0,
    }
    result = {
        "summary": summary,
        "config": cfg,
        "stream": stream.summary(),
        "communication": ledger.summary(),
        "rounds": rounds_log,
        "cloud_history": learner.history,
    }
    if save and not partial:
        p = result_path(cfg, f"{cfg['dataset']}/{stream.scenario}/"
                             f"{tag}_seed{seed}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result["path"] = p
        if verbose:
            print(f"  -> {p}", flush=True)
    return result
