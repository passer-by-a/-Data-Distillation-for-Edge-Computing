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
from ..distillation.online_coreset import (OnlineHerding, OnlineKMeans,
                                           ReservoirCoreset, build_coreset)
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
    "frozen_stream_dm": {
        "vae": {"online_update": False, "replay_enabled": False,
                "replay_ratio": 0.0},
        "alignment": {"method": "none"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "full_sync"}},
    "online_naive_dm": {
        "vae": {"online_update": True, "replay_enabled": False,
                "replay_ratio": 0.0},
        "alignment": {"method": "none"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "full_sync"}},
    "online_replay_dm": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "reencode"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "full_sync"}},
    # ---- 拆变量后的显式方法名 ----
    "replay_reinit": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "reencode"},
        "distillation": {"method": "reinit"},
        "communication": {"mode": "full_sync"}},
    "replay_dm": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "reencode"},
        "distillation": {"method": "dm", "update_fraction": 1.0},
        "communication": {"mode": "full_sync"}},
    "affine_dm": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm", "update_fraction": 1.0},
        "communication": {"mode": "full_sync"}},
    # 原版 DM（线性核 MMD = 特征均值匹配，无方差/anchor 项）
    "replay_dm_mean": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "reencode"},
        "distillation": {"method": "dm", "update_fraction": 1.0,
                         "lambda_var": 0.0, "lambda_anchor": 0.0},
        "communication": {"mode": "full_sync"}},
    "affine_dm_mean": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm", "update_fraction": 1.0,
                         "lambda_var": 0.0, "lambda_anchor": 0.0},
        "communication": {"mode": "full_sync"}},
    "affine_dm_patch_100up_10tx": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm", "update_fraction": 1.0},
        "communication": {"mode": "patch", "patch_fraction": 0.10}},
    "affine_dm_partial_10up_100tx": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm", "update_fraction": 0.10},
        "communication": {"mode": "full_sync"}},
    "affine_dm_partial_10up_10tx": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm", "update_fraction": 0.10},
        "communication": {"mode": "patch", "patch_fraction": 0.10}},
    "online_replay_align_dm": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "full_sync"}},
    "online_kmeans": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "kmeans"},
        "communication": {"mode": "full_sync"}},
    "online_herding": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "herding"},
        "communication": {"mode": "full_sync"}},
    "online_reservoir": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "reservoir"},
        "communication": {"mode": "full_sync"}},
    "online_moment": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "moment"},
        "communication": {"mode": "full_sync"}},
    "full_method": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "patch"}},
    "full_method_full_sync": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "full_sync"}},
    "online_replay_align_scratch_cloud": {
        "vae": {"online_update": True, "replay_enabled": True,
                "replay_ratio": 0.25},
        "alignment": {"method": "diagonal_affine"},
        "distillation": {"method": "dm"},
        "communication": {"mode": "full_sync"},
        "cloud": {"incremental_training": False}},
}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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
        split_seed=int(cfg["split_seed"]))


def _state_signature(cfg, tag, seed):
    """对完整配置做哈希：任何影响训练结果的参数变化都会使断点失效"""
    c = copy.deepcopy(cfg)
    for k in ("_path", "results_dir", "checkpoints_dir", "figures_dir",
              "_run_tag", "_run_name"):
        c.pop(k, None)
    payload = {"tag": tag, "seed": int(seed), "config": c}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _base_signature(cfg, seed):
    """Initial Base VAE 缓存签名：包含数据划分、架构与训练超参"""
    v = cfg["vae"]
    s = cfg["stream"]
    return {
        "cache_version": 2,
        "dataset": cfg["dataset"], "num_classes": int(cfg["num_classes"]),
        "latent_dim": int(cfg["latent_dim"]),
        "base_channels": int(cfg["base_channels"]),
        "seed": int(seed),
        "initial_fraction": float(s["initial_fraction"]),
        "shuffle_seed": int(s["shuffle_seed"]),
        "split_seed": int(cfg["split_seed"]),
        "val_ratio": float(cfg["val_ratio"]),
        "class_balanced_initial": bool(s.get("class_balanced_initial", True)),
        "initial_epochs": int(v["initial_epochs"]),
        "batch_size": int(v["batch_size"]),
        "lr": float(v["lr"]), "beta": float(v["beta"]),
        "cls_lambda": float(v["cls_lambda"]),
        "augment": bool(v.get("augment", True)),
    }


def _save_run_state(path, *, vae, version, last_round, protocol, memory,
                    statistics, replay, cloud, learner, coreset, ledger,
                    rounds_log, edge_sec, peak_gpu, signature, wall_sec=0.0):
    torch.save({
        "signature": signature, "vae": vae.state_dict(),
        "version": int(version), "last_round": int(last_round),
        "protocol_mean": protocol.mean.clone(),
        "protocol_std": protocol.std.clone(),
        "memory": memory.state_dict(), "statistics": statistics.state_dict(),
        "replay": replay.state_dict(), "cloud": cloud.state_dict(),
        "learner": learner.state_dict(), "coreset": coreset.state_dict()
        if coreset is not None else None,
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
    replay_enabled = bool(vcfg.get("replay_enabled", True))
    anchor_count = int(cfg["alignment"].get("anchor_count", 200))
    anchor_count = min(anchor_count, int(cfg["memory"]["raw_replay_capacity"]))
    signature = _state_signature(cfg, tag, seed)

    stream = _build_stream(cfg)
    if verbose:
        print(f"[{tag}] {cfg['dataset']} {stream.scenario} "
              f"base={len(stream.base_indices)} rounds={stream.num_rounds} "
              f"budget={budget} align={align_mode} distill={distill} "
              f"sync={sync_mode} quant={quant_mode} seed={seed}", flush=True)

    t_start = time.time()
    base = stream.base
    vae = build_vae(C, int(cfg["latent_dim"]), int(cfg["base_channels"]),
                    seed=int(seed), device=device)
    base_sig = _base_signature(cfg, seed)
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
                      f"{os.path.relpath(base_ckpt, os.getcwd())}", flush=True)
    if not cached:
        base_log = train_initial_vae(vae, base.images, base.labels, cfg, device,
                                     verbose=verbose)
        os.makedirs(os.path.dirname(base_ckpt), exist_ok=True)
        torch.save({"model_state_dict": vae.state_dict(),
                    "base_log": base_log, "base_signature": base_sig,
                    "seed": int(seed),
                    "accessible": "initial_base_only"}, base_ckpt)
    vae.eval()

    base_raw = encode_mu(vae.encoder, base.images, device=device)
    protocol = FeatureProtocol.fit(base_raw)
    base_feats = protocol.normalize(base_raw)

    quotas = allocate_quotas(base.labels.numpy(), budget, C)
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
    if distill in ("kmeans", "herding", "reservoir"):
        coreset = build_coreset(distill, memory.features, memory.labels,
                                quotas, dim=dim, seed=int(seed))

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

    ckpt0 = checkpoint_path(
        cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/vae_v0.pt")
    if save:
        torch.save({"model_state_dict": vae.state_dict(),
                    "encoder_version": 0, "method": tag, "seed": int(seed),
                    "accessible_rounds": "initial_base_only",
                    "n_base": int(len(base.labels)),
                    "stream": stream.summary()}, ckpt0)

    val_feats = protocol.normalize(
        encode_mu(vae.encoder, stream.val_images, device=device))
    test_feats = protocol.normalize(
        encode_mu(vae.encoder, stream.test_images, device=device))
    cloud_mode = "incremental" if cfg["cloud"].get("incremental_training",
                                                   True) else "scratch"
    cloud_init = learner.update(cloud.features, cloud.labels, val_feats,
                                stream.val_labels, round_idx=-1,
                                mode=cloud_mode)
    init_metrics = learner.evaluate(test_feats, stream.test_labels)
    peak_gpu = 0.0
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
    state_path = checkpoint_path(
        cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/stream_state.pt")
    # 让在线阶段的随机性独立于 base 权重是否来自缓存
    _seed_everything(int(seed) + 1_000_003)

    version = 0
    peak_gpu = max(peak_gpu, 0.0)
    start_round = 0
    wall_sec = 0.0
    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["signature"] != signature:
            raise ValueError("断点状态与当前配置签名不一致，拒绝续跑")
        vae.load_state_dict(st["vae"])
        protocol = FeatureProtocol(st["protocol_mean"], st["protocol_std"])
        memory = RepresentativeMemory.from_state_dict(st["memory"])
        stats = StreamingStatistics.from_state_dict(st["statistics"])
        replay = ClassBalancedReservoir.from_state_dict(st["replay"])
        cloud = CloudState.from_state_dict(st["cloud"])
        learner.load_state_dict(st["learner"])
        ledger = CommunicationLedger()
        ledger.entries = copy.deepcopy(st["ledger"])
        if st.get("coreset") is not None:
            if distill == "kmeans":
                coreset = OnlineKMeans.from_state_dict(st["coreset"])
            elif distill == "herding":
                coreset = OnlineHerding.from_state_dict(st["coreset"])
            elif distill == "reservoir":
                coreset = ReservoirCoreset.from_state_dict(st["coreset"])
        rounds_log = copy.deepcopy(st["rounds_log"])
        edge_sec = st["edge_sec"]
        peak_gpu = max(peak_gpu, st["peak_gpu"])
        wall_sec = float(st.get("wall_sec", 0.0))
        version = int(st["version"])
        start_round = int(st["last_round"]) + 1
        stream._base_accessed = True
        stream._max_accessed = int(st["last_round"])
        torch.set_rng_state(st["torch_rng"])
        if device == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        vae.eval()
        if verbose:
            print(f"  [resume] 从第 {start_round} 轮继续 "
                  f"(已完成 {start_round}/{stream.num_rounds})", flush=True)
    elif save:
        _save_run_state(state_path, vae=vae, version=version, last_round=-1,
                        protocol=protocol, memory=memory, statistics=stats,
                        replay=replay, cloud=cloud, learner=learner,
                        coreset=coreset, ledger=ledger, rounds_log=rounds_log,
                        edge_sec=edge_sec, peak_gpu=peak_gpu,
                        signature=signature,
                        wall_sec=wall_sec + (time.time() - t_start))

    for t in range(start_round, stream.num_rounds):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr = time.time()
        current = stream.round(t)
        prev_encoder = snapshot_encoder(vae, version)

        # ---------------- Online VAE ----------------
        vae_sec = 0.0
        vlog = {"skipped": True}
        rep_imgs, rep_labels = replay.get()
        if vae_update:
            if not replay_enabled:
                rep_imgs = torch.empty(0, 3, 32, 32, dtype=torch.uint8)
                rep_labels = torch.empty(0, dtype=torch.long)
            vlog = online_vae_update(vae, current.images, current.labels,
                                     rep_imgs, rep_labels, prev_encoder, cfg,
                                     device=device, round_idx=t)
            vae_sec = vlog["train_sec"]
            edge_sec += vae_sec
        vae.eval()
        new_version = version + 1 if vae_update else version

        # ---------------- Anchors / Drift / Alignment ----------------
        anchors_img, anchors_y = replay.get()
        if len(anchors_img) == 0:
            anchors_img, anchors_y = current.images, current.labels
        n_a = min(anchor_count, len(anchors_img))
        # 均匀抽取，保证锚点覆盖各类别（Replay 按类别分组存储）
        sel = torch.linspace(0, len(anchors_img) - 1, n_a).long().unique()
        anchors_img, anchors_y = anchors_img[sel], anchors_y[sel]
        n_a = len(sel)
        ta = time.time()
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
                z_old, z_new, dim=1).mean())}
        if vae_update and align_mode == "diagonal_affine":
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
        rep_feats_reenc = None
        if vae_update and align_mode == "reencode" and len(rep_imgs) > 0:
            rep_feats_reenc = protocol.normalize(
                encode_mu(vae.encoder, rep_imgs, device=device))
        encode_sec = time.time() - te

        # ---------------- 统计记忆 ----------------
        ts = time.time()
        if not vae_update:
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
        else:  # none: 跨版本直接混用是禁止的，只能丢弃历史统计
            stats.reset(new_version)
            stats.update(cur_feats, current.labels, new_version, round_idx=t)
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
        elif patch_idx is not None:
            opt_idx = patch_idx
        else:
            opt_idx = select_patch_indices(memory.quotas, update_fraction, t)
        updated_idx = np.arange(budget) if opt_idx is None else opt_idx
        td = time.time()
        if vae_update and align_mode == "reencode":
            all_f = torch.cat([rep_feats_reenc, cur_feats], 0) \
                if rep_feats_reenc is not None else cur_feats
            all_y = torch.cat([rep_labels, current.labels], 0) \
                if rep_feats_reenc is not None else current.labels
            memory.reinitialize_from_features(
                all_f, all_y,
                method=cfg["memory"].get("representative_init", "kmeans"),
                version=new_version, round_idx=t, seed=int(seed),
                device=device, cfg=cfg["distillation"])
            if distill == "dm":
                sdm = StreamingDM(memory.features, memory.labels,
                                  memory.quotas, cfg["distillation"],
                                  seed=int(seed), device=device)
                out = sdm.update(stats, allowed_idx=opt_idx, round_idx=t)
                memory.features = out["features"]
                distill_info = {**out["info"], "init": "reencode_kmeans"}
            elif distill == "reinit":
                distill_info = {"method": "reinit",
                                "init": cfg["memory"].get(
                                    "representative_init", "kmeans"),
                                "n_optimized": int(len(updated_idx))}
            else:
                distill_info = {"method": "reencode"}
        else:
            if distill == "reinit":
                raise ValueError("reinit 方法只用于 alignment.method=reencode")
            if new_version > version:
                memory.align(alignment)
            if distill == "dm":
                sdm = StreamingDM(memory.features, memory.labels,
                                  memory.quotas, cfg["distillation"],
                                  seed=int(seed), device=device)
                out = sdm.update(stats, allowed_idx=opt_idx, round_idx=t)
                memory.features = out["features"]
                distill_info = out["info"]
            elif distill in ("kmeans", "herding", "reservoir"):
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
            else:
                raise ValueError(f"未知蒸馏方法: {distill}")
            memory.update_features(updated_idx, memory.features[updated_idx],
                                   new_version, t)
        distill_sec = time.time() - td
        edge_sec += distill_sec + encode_sec + align_sec + stats_sec

        # ---------------- Replay ----------------
        if replay_enabled:
            replay.add_batch(current.images, current.labels)
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
            dm_met = learner.evaluate(dom_f, stream.test_labels)
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
        if not vae_update or align_mode == "diagonal_affine":
            stats_scope = "aligned_full_history"
        elif align_mode == "reencode":
            stats_scope = "replay_plus_current"
        else:
            stats_scope = "current_only"

        vm = evaluate_vae(vae, stream.val_images, stream.val_labels, cfg,
                          device=device) if vae_update else None
        mem_summary = memory.summary(stats)
        peak = torch.cuda.max_memory_allocated() / 1e9 \
            if device == "cuda" else 0.0
        peak_gpu = max(peak_gpu, peak)

        rec = {
            "round": t, "encoder_version": new_version,
            "vae": vlog, "vae_sec": vae_sec,
            "anchor_drift": anchor_drift,
            "alignment": alignment.meta() if align_mode != "reencode" else
            {"method": "reencode"},
            "align_sec": align_sec,
            "statistics": stats.summary(),
            "stats_scope": stats_scope,
            "stats_sec": stats_sec,
            "n_current": int(len(current.labels)),
            "n_cumulative_seen": int(len(base.labels) + sum(
                len(stream.round_indices[r]) for r in range(t + 1))),
            "n_effective_stats": int(stats.n_seen),
            "n_replay": len(replay),
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

        ck = checkpoint_path(
            cfg, f"{cfg['dataset']}/{stream.scenario}/seed{seed}/{tag}/vae_v{new_version}.pt")
        if save:
            torch.save({"model_state_dict": vae.state_dict(),
                        "encoder_version": new_version, "method": tag,
                        "seed": int(seed), "round": t,
                        "accessible_rounds": f"initial_base + rounds 0..{t}",
                        "n_seen": int(len(base.labels) + sum(
                            len(stream.round_indices[r]) for r in range(t + 1))),
                        "stream": stream.summary()}, ck)

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
                            wall_sec=wall_sec + (time.time() - t_start))
        if stop_after is not None and t >= int(stop_after):
            break

    partial = stop_after is not None and rounds_log[-1]["round"] >= int(stop_after)
    accs = [r["acc"] for r in rounds_log[1:]]
    f1s = [r["f1"] for r in rounds_log[1:]]
    final = rounds_log[-1]
    summary = {
        "method": tag, "seed": int(seed),
        "partial": bool(partial),
        "dataset": cfg["dataset"], "scenario": stream.scenario,
        "num_rounds": stream.num_rounds,
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
        "n_train_used": int(stats.n_seen),
        "cumulative_seen_samples": int(len(base.labels) + sum(
            len(stream.round_indices[r]) for r in range(stream.num_rounds))),
        "effective_stats_samples": int(stats.n_seen),
        "replay_samples_final": int(len(replay)),
        "stats_scope": final.get("stats_scope"),
        "vae_ckpt": ck,
        "vae_accessible": f"initial_base + rounds 0..{final['round']}",
        "base_cache_signature": base_sig,
        "raw_replay_capacity": int(cfg["memory"]["raw_replay_capacity"]),
        "replay_bytes": replay.nbytes(),
        "edge_train_sec": edge_sec,
        "edge_total_sec": wall_sec + (time.time() - t_start),
        "eval_encode_sec_total": float(sum(
            r.get("eval_encode_sec", 0.0) for r in rounds_log[1:])),
        "int8_eval_sec_total": float(sum(
            r.get("test_int8_sec", 0.0) for r in rounds_log[1:])),
        "cloud_total_sec": learner.total_train_sec,
        "peak_gpu_gb": peak_gpu,
        "stats_seen_final": int(stats.n_seen),
        "expected_seen_final": int(len(base.labels) + sum(
            len(stream.round_indices[r]) for r in range(stream.num_rounds))),
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
