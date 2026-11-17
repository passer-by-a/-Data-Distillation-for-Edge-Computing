"""vae_streaming 协议单元测试（CPU 可运行）

覆盖任务书要求的 19 项检查。运行:
  python -m pytest vae_streaming/tests/test_streaming_protocol.py -q
"""
import copy
import json
import os

import numpy as np
import pytest
import torch

from vae import EdgeVAE
from vae_bdm.baselines import allocate_quotas
from vae_streaming.cloud.incremental_trainer import IncrementalCloudLearner
from vae_streaming.cloud.state_sync import CloudState, apply_sync_message
from vae_streaming.communication.payload_accounting import (CommunicationLedger,
                                                            build_sync_message,
                                                            message_size,
                                                            select_patch_indices)
from vae_streaming.communication.quantization import (decode_features,
                                                      encode_features,
                                                      roundtrip_error)
from vae_streaming.data.causal_stream import CausalStream
from vae_streaming.data.stream_scenarios import apply_covariate_shift
from vae_streaming.memory.replay_buffer import ClassBalancedReservoir
from vae_streaming.memory.representative_memory import RepresentativeMemory
from vae_streaming.memory.running_statistics import StreamingStatistics
from vae_streaming.models.feature_alignment import (estimate_feature_alignment,
                                                    identity_alignment)
from vae_streaming.models.online_vae import (FeatureProtocol, build_vae,
                                             encode_mu, online_vae_update,
                                             snapshot_encoder)


DIM = 1024


def _vcfg(**kw):
    cfg = {"vae": {"online_update": True, "steps_per_round": 2,
                   "batch_size": 8, "lr": 1e-3, "beta": 1e-4,
                   "cls_lambda": 0.1, "replay_enabled": True,
                   "replay_ratio": 0.5, "stability_enabled": False,
                   "lambda_stab": 0.0, "augment": False, "seed": 0}}
    for k, v in kw.items():
        cfg["vae"][k] = v
    return cfg


def _images(n=32, seed=0, num_classes=3):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, 3, 32, 32, generator=g) * 255).byte()
    y = torch.arange(n) % num_classes
    return x, y


# ======================================================================
# 1. 当前 VAE Encoder 确实更新
# ======================================================================
def test_encoder_updates():
    torch.manual_seed(0)
    vae = build_vae(3, latent_dim=16, base_channels=8, seed=0, device="cpu")
    before = {k: v.clone() for k, v in vae.encoder.state_dict().items()}
    x, y = _images(16)
    online_vae_update(vae, x, y, x[:8], y[:8], None, _vcfg(), device="cpu")
    changed = any(not torch.equal(before[k], v)
                  for k, v in vae.encoder.state_dict().items())
    assert changed, "Online VAE 未更新 Encoder"
    assert all(p.requires_grad for p in vae.encoder.parameters())


# ======================================================================
# 2. 上一轮 Encoder 快照不接收梯度且保持不变
# ======================================================================
def test_previous_encoder_frozen():
    torch.manual_seed(0)
    vae = build_vae(3, latent_dim=16, base_channels=8, seed=0, device="cpu")
    snap = snapshot_encoder(vae, 0)
    frozen = {k: v.clone() for k, v in snap.state_dict().items()}
    assert all(not p.requires_grad for p in snap.module.parameters())
    x, y = _images(16)
    online_vae_update(vae, x, y, x[:8], y[:8], snap, _vcfg(), device="cpu")
    for k, v in snap.state_dict().items():
        assert torch.equal(frozen[k], v), "上一轮快照被意外修改"


# ======================================================================
# 3. 新批次不能访问未来轮次
# ======================================================================
@pytest.fixture(scope="module")
def stream():
    return CausalStream(dataset="cifar10", initial_fraction=0.2,
                        num_rounds=4, seed=42, scenario="iid")


def test_no_future_access(stream):
    base = stream.base
    r0 = stream.round(0)
    with pytest.raises(RuntimeError):
        stream.round(2)
    r1 = stream.round(1)
    assert set(base.indices.tolist()).isdisjoint(set(r0.indices.tolist()))
    assert set(r0.indices.tolist()).isdisjoint(set(r1.indices.tolist()))
    stream.assert_no_leakage(1, np.concatenate([r0.indices, r1.indices]))
    with pytest.raises(RuntimeError):
        stream.assert_no_leakage(0, r1.indices)


# ======================================================================
# 4. Replay Memory 不超过配置容量
# ======================================================================
def test_replay_capacity():
    buf = ClassBalancedReservoir(capacity=90, num_classes=3, seed=0)
    for i in range(5):
        x, y = _images(60, seed=i)
        buf.add_batch(x, y)
        assert len(buf) <= 90
        buf.assert_capacity()
    per = buf.per_class_counts()
    assert per == [30, 30, 30]
    assert buf.nbytes() > 0


# ======================================================================
# 5. 过去样本不会因回放被重复计入统计量
# ======================================================================
def test_replay_not_double_counted():
    stats = StreamingStatistics(2, 8, version=0)
    x, y = _images(10, num_classes=1)
    f = torch.randn(10, 8)
    stats.update(f, torch.zeros(10, dtype=torch.long), 0)
    n_before = stats.n_seen
    buf = ClassBalancedReservoir(20, 2, seed=0)
    buf.add_batch(x, y)
    buf.add_batch(x, y)          # 回放不应触发 stats.update
    assert stats.n_seen == n_before == 10
    assert int(stats.count[0]) == 10


# ======================================================================
# 6. 固定 Encoder 时，Streaming 均值/方差与离线一致
# ======================================================================
def test_streaming_stats_match_offline():
    torch.manual_seed(0)
    C, D, n, b = 3, 16, 300, 37
    f = torch.randn(n, D, dtype=torch.float64)
    y = torch.arange(n) % C
    stats = StreamingStatistics(C, D, version=0)
    for i in range(0, n, b):
        stats.update(f[i:i + b], y[i:i + b], 0)
    for c in range(C):
        cnt, mu, var = stats.get(c)
        off = f[y == c]
        assert cnt == len(off)
        assert torch.allclose(mu.double(), off.mean(0), atol=1e-10)
        assert torch.allclose(var.double(), off.var(0, unbiased=False),
                              atol=1e-10)


# ======================================================================
# 7. 更新 Encoder 后，旧统计量不会直接与新统计量混用
# ======================================================================
def test_no_mixed_versions():
    stats = StreamingStatistics(2, 8, version=0)
    f = torch.randn(5, 8)
    y = torch.zeros(5, dtype=torch.long)
    stats.update(f, y, 0)
    with pytest.raises(ValueError):
        stats.update(torch.randn(5, 8), y, 1)
    al = identity_alignment(8, 0, 1)
    stats.align_to(1, al)
    assert stats.version == 1 and bool(stats.estimated[0])
    assert int(stats.count[0]) == 5
    stats.update(torch.randn(5, 8), y, 1)
    assert int(stats.count[0]) == 10


# ======================================================================
# 8. 特征对齐只使用历史锚点
# ======================================================================
def test_alignment_uses_history_only():
    torch.manual_seed(0)
    vae = build_vae(3, latent_dim=16, base_channels=8, seed=0, device="cpu")
    prev = snapshot_encoder(vae, 0)
    cur = copy.deepcopy(vae.encoder).eval()
    with torch.no_grad():
        for p in cur.parameters():
            p.add_(0.01 * torch.randn_like(p))
    anchors, ay = _images(24, num_classes=2)
    before = anchors.clone()
    al = estimate_feature_alignment(prev, cur, anchors, ay, version_from=0,
                                    version_to=1, device="cpu")
    assert torch.equal(before, anchors), "对齐过程修改了锚点图像"
    assert al.n_anchors == 24 and al.n_fit == 12 and al.n_heldout == 12
    assert al.version_from == 0 and al.version_to == 1
    assert np.isfinite(al.relative_err)
    al2 = estimate_feature_alignment(prev, cur, anchors, ay, version_from=0,
                                     version_to=1, device="cpu")
    assert torch.allclose(al.scale, al2.scale)
    assert torch.allclose(al.bias, al2.bias)


# ======================================================================
# 9. 当前代表特征与云端同步代表特征具有相同版本
# ======================================================================
def _tiny_memory(M=4, C=2, dim=DIM, seed=0):
    x = torch.randn(M * 3, dim, generator=torch.Generator().manual_seed(seed))
    y = torch.arange(M * 3) % C
    quotas = allocate_quotas(y.numpy(), M, C)
    mem = RepresentativeMemory(M, C, quotas, dim)
    mem.initialize_from_features(x, y, method="random", version=0, seed=seed)
    return mem


def test_sync_versions_match():
    mem = _tiny_memory()
    cloud = CloudState.empty(mem.budget, mem.dim)
    msg, _ = build_sync_message("full_sync", -1, mem, None, "int8")
    cloud.apply(msg)
    dec = decode_features(msg["features"], json.loads(msg["meta"])["quant"])
    mem.set_submitted(np.arange(mem.budget), dec, 0)
    v = cloud.verify_consistent(mem)
    assert v["all_ok"] and torch.equal(cloud.versions, mem.submitted_versions)


# ======================================================================
# 10. 代表特征数量不超过固定预算
# ======================================================================
def test_budget():
    mem = _tiny_memory(M=8, C=2)
    assert mem.assert_budget()
    assert len(mem.features) == 8
    assert int(mem.quotas.sum()) == 8
    assert list(mem.quotas) == [4, 4]


# ======================================================================
# 11. 所有代表特征维度均为 1024
# ======================================================================
def test_dim_1024():
    mem = _tiny_memory(M=4, C=2)
    assert mem.features.shape == (4, 1024)
    assert mem.submitted.shape == (4, 1024)
    assert mem.cloud_view()["features"].shape == (4, 1024)
    assert mem.dim == 1024


# ======================================================================
# 12. INT8 量化能够正确解码
# ======================================================================
def test_int8_roundtrip():
    f = torch.randn(64, DIM) * 2.5
    err = roundtrip_error(f, "int8")
    assert err["bits"] == 8
    arr, meta = encode_features(f, "int8")
    rec = decode_features(arr, meta)
    assert rec.shape == f.shape
    assert torch.allclose(rec, f, atol=float(meta["scale"]) + 1e-6)
    assert err["mse"] < 1e-3


# ======================================================================
# 13. 每次同步实际字节数可复核
# ======================================================================
def test_bytes_recount():
    mem = _tiny_memory(M=4, C=2)
    msg, _ = build_sync_message("full_sync", -1, mem, None, "int8")
    size = message_size(msg)
    manual = len(msg["meta"].encode()) + sum(
        v.nbytes for k, v in msg.items() if k != "meta")
    assert size["theoretical_bytes"] == manual
    assert size["file_bytes"] > 0
    assert size["file_bytes"] >= size["theoretical_bytes"]
    ledger = CommunicationLedger().add("initial_sync", -1, size)
    assert ledger.cumulative_bytes == manual


# ======================================================================
# 14. 固定补丁预算下边缘与云端状态一致
# ======================================================================
def test_patch_consistency():
    mem = _tiny_memory(M=20, C=2)
    cloud = CloudState.empty(mem.budget, mem.dim)
    m0, _ = build_sync_message("full_sync", -1, mem, None, "int8")
    cloud.apply(m0)
    dec = decode_features(m0["features"], json.loads(m0["meta"])["quant"])
    mem.set_submitted(np.arange(mem.budget), dec, 0)

    al = identity_alignment(mem.dim, 0, 1)
    mem.align(al)
    mem.features[::2] = mem.features[::2] + 0.01
    mem.versions.fill_(1)
    idx = select_patch_indices(mem.quotas, 0.25, 0)
    assert len(idx) == 5
    msg, info = build_sync_message("patch", 0, mem, al, "int8", 0.25, idx)
    cloud.apply(msg)
    dec = decode_features(msg["features"], json.loads(msg["meta"])["quant"])
    mem.set_submitted(idx, dec, 1)
    v = cloud.verify_consistent(mem)
    assert v["all_ok"], v
    assert torch.equal(cloud.versions, torch.ones(mem.budget, dtype=torch.long))


# ======================================================================
# 15. 同步消息重复应用不会重复修改状态
# ======================================================================
def test_duplicate_sync_idempotent():
    mem = _tiny_memory(M=4, C=2)
    cloud = CloudState.empty(mem.budget, mem.dim)
    msg, _ = build_sync_message("full_sync", -1, mem, None, "int8")
    assert cloud.apply(msg) is True
    before = cloud.state_dict()
    changed = cloud.apply(msg)
    after = cloud.state_dict()
    assert changed is False
    assert torch.equal(before["features"], after["features"])
    assert before["last_sync_id"] == after["last_sync_id"]


# ======================================================================
# 16. 云端分类器保留上一轮模型与优化器状态
# ======================================================================
def test_cloud_state_preserved():
    cfg = {"cloud": {"head": "linear", "lr": 1e-3, "wd": 1e-4, "batch": 8,
                     "steps_per_round": 5, "seed": 0, "epochs": 2}}
    learner = IncrementalCloudLearner(cfg, dim=8, num_classes=2, device="cpu")
    x = torch.randn(16, 8)
    y = torch.arange(16) % 2
    xv = torch.randn(8, 8)
    yv = torch.arange(8) % 2
    learner.update(x, y, xv, yv, round_idx=0)
    assert len(learner.opt.state_dict()["state"]) > 0
    w1 = copy.deepcopy(learner.model.state_dict())
    assert learner.model is not None
    learner.update(x, y, xv, yv, round_idx=1)
    assert learner.steps_done == 10
    assert learner.history[0]["mode"] == "incremental"
    assert len(learner.opt.state_dict()["state"]) > 0
    assert any(not torch.equal(w1[k], learner.model.state_dict()[k])
               for k in w1)
    st = learner.state_dict()
    learner2 = IncrementalCloudLearner(cfg, dim=8, num_classes=2, device="cpu")
    learner2.load_state_dict(st)
    assert learner2.steps_done == learner.steps_done
    assert torch.equal(learner2.model.state_dict()["fc.weight"],
                       learner.model.state_dict()["fc.weight"])


# ======================================================================
# 17. 测试集不参与流式训练
# ======================================================================
def test_test_set_isolated(stream):
    base = stream.base
    assert len(base.labels) == len(stream.base_indices)
    assert torch.equal(base.images,
                       stream.train_images[torch.from_numpy(stream.base_indices)])
    assert base.indices.max() < len(stream.train_images)
    assert len(stream.test_images) == 10000
    assert stream.train_images.data_ptr() != stream.test_images.data_ptr()
    for t in range(stream.num_rounds):
        assert stream.round_indices[t].max() < len(stream.train_images)


# ======================================================================
# 18. 完整训练与断点恢复产生一致结果（组件级）
# ======================================================================
def test_state_roundtrip():
    buf = ClassBalancedReservoir(12, 2, seed=0)
    x, y = _images(40, num_classes=2)
    buf.add_batch(x, y)
    buf2 = ClassBalancedReservoir.from_state_dict(buf.state_dict())
    xa, ya = buf.get()
    xb, yb = buf2.get()
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    buf.add_batch(x, y)
    buf2.add_batch(x, y)
    assert torch.equal(buf.get()[0], buf2.get()[0])

    stats = StreamingStatistics(2, 8, version=0)
    stats.update(torch.randn(6, 8), torch.zeros(6, dtype=torch.long), 0)
    stats2 = StreamingStatistics.from_state_dict(stats.state_dict())
    f = torch.randn(4, 8)
    yb2 = torch.ones(4, dtype=torch.long)
    stats.update(f, yb2, 0)
    stats2.update(f, yb2, 0)
    assert torch.allclose(stats.mean, stats2.mean)
    assert torch.equal(torch.from_numpy(stats.count),
                       torch.from_numpy(stats2.count))

    mem = _tiny_memory(M=4, C=2)
    mem2 = RepresentativeMemory.from_state_dict(mem.state_dict())
    assert torch.equal(mem.features, mem2.features)
    assert torch.equal(mem.labels, mem2.labels)

    cloud = CloudState.empty(4, 4)
    msg, _ = build_sync_message("full_sync", -1, mem, None, "fp32")
    cloud.apply(msg)
    c2 = CloudState.from_state_dict(cloud.state_dict())
    assert torch.equal(cloud.features, c2.features)
    assert cloud.last_sync_id == c2.last_sync_id


# ======================================================================
# 19. 第一阶段代码与 checkpoint 仍可正常运行
# ======================================================================
def test_stage1_intact():
    ck = "checkpoints/autoencoder_edge_s8_c16.pt"
    if not os.path.exists(ck):
        pytest.skip("第一阶段 checkpoint 不存在")
    obj = torch.load(ck, map_location="cpu", weights_only=True)
    ae = EdgeVAE(latent_dim=16, num_classes=10, base_channels=64)
    ae.load_state_dict(obj["model_state_dict"])
    mu = ae.encode_mu(torch.randn(2, 3, 32, 32))
    assert mu.shape == (2, 16, 8, 8) and mu.numel() // 2 == 1024
    old = "vae_bdm/results/cifar10_main_r10_summary.csv"
    if os.path.exists(old):
        assert os.path.getsize(old) > 0
    from vae_bdm.evaluate import eval_reps  # noqa: F401
    assert callable(eval_reps)


# ======================================================================
# 附加: 场景构造与域变换
# ======================================================================
def test_covariate_shift_deterministic():
    x, _ = _images(8)
    a = apply_covariate_shift(x, 0.5)
    b = apply_covariate_shift(x, 0.5)
    assert torch.equal(a, b)
    assert a.dtype == torch.uint8
    assert torch.equal(apply_covariate_shift(x, 0.0), x)
    assert not torch.equal(a, x)


def test_label_shift_stream():
    s = CausalStream(dataset="cifar10", initial_fraction=0.2, num_rounds=3,
                     seed=7, scenario="label_shift")
    base = s.base
    counts = [s.round_class_counts[r] for r in range(s.num_rounds)]
    assert len(s.blocks) == 3
    assert all(int(c.sum()) > 0 for c in counts)
    # 每类仍然出现在 base 中
    assert np.bincount(base.labels.numpy(), minlength=10).min() > 0


# ======================================================================
# 20. 完整训练与断点恢复产生一致结果（整条流水线）
# ======================================================================
def test_full_pipeline_resume(tmp_path):
    from vae_streaming.config_utils import load_config
    from vae_streaming.experiments.engine import run_method

    cfg = load_config("vae_streaming/configs/cifar10.yaml")
    cfg["stream"]["initial_fraction"] = 0.10
    cfg["stream"]["num_rounds"] = 2
    cfg["memory"]["representative_budget"] = 20
    cfg["vae"]["initial_epochs"] = 1
    cfg["vae"]["steps_per_round"] = 2
    cfg["cloud"]["steps_per_round"] = 3
    cfg["checkpoints_dir"] = str(tmp_path / "ckpt")
    cfg["results_dir"] = str(tmp_path / "res")
    cfg["figures_dir"] = str(tmp_path / "fig")
    # 用确定性的 random 初始化，隔离 sklearn KMeans 的线程级浮点差异
    cfg["memory"]["representative_init"] = "random"

    kw = dict(method="dm", seed=7, verbose=False, save=True)
    full = run_method(cfg, **kw)
    partial = run_method(cfg, stop_after=0, **kw)
    assert len(partial["rounds"]) == 2
    resumed = run_method(cfg, resume=True, **kw)
    assert len(full["rounds"]) == len(resumed["rounds"]) == 3
    for a, b in zip(full["rounds"], resumed["rounds"]):
        assert a["round"] == b["round"]
        assert a["acc"] == pytest.approx(b["acc"], abs=1e-6)
        assert a["f1"] == pytest.approx(b["f1"], abs=1e-6)
        assert a["statistics"]["n_seen"] == b["statistics"]["n_seen"]
    assert (full["communication"]["cumulative_upload_bytes"] ==
            resumed["communication"]["cumulative_upload_bytes"])


# ======================================================================
# 21. 对齐必须在 Normalized 特征空间拟合与应用
# ======================================================================
def test_alignment_normalized_space():
    torch.manual_seed(0)
    vae = build_vae(3, latent_dim=16, base_channels=8, seed=0, device="cpu")
    prev = snapshot_encoder(vae, 0)
    cur = copy.deepcopy(vae.encoder).eval()
    with torch.no_grad():
        for p in cur.parameters():
            p.add_(0.01 * torch.randn_like(p))
    anchors, ay = _images(24, num_classes=2)
    raw_old = encode_mu(prev.module, anchors, device="cpu")
    protocol = FeatureProtocol.fit(raw_old)
    n_old = protocol.normalize(raw_old)
    n_new = protocol.normalize(encode_mu(cur, anchors, device="cpu"))
    al_norm = estimate_feature_alignment(prev, cur, anchors, ay,
                                        protocol=protocol, device="cpu")
    al_raw = estimate_feature_alignment(prev, cur, anchors, ay, device="cpu")
    err_norm = float((al_norm.apply(n_old) - n_new).pow(2).mean())
    err_raw = float((al_raw.apply(n_old) - n_new).pow(2).mean())
    assert err_norm <= err_raw, (err_norm, err_raw)
    assert torch.allclose(al_norm.scale, al_raw.scale, atol=1e-5)
    assert al_norm.n_classes_center_heldout > 0
    assert np.isfinite(al_norm.fit_class_center_err)
    assert al_norm.heldout_mse < al_norm.no_align_heldout_mse
    assert al_norm.improvement_heldout > 0


# ======================================================================
# 22. Base VAE 缓存签名覆盖数据划分与架构
# ======================================================================
def test_base_cache_signature_changes():
    from vae_streaming.config_utils import load_config
    from vae_streaming.experiments.engine import _base_signature, _build_stream

    cfg = load_config("vae_streaming/configs/cifar10.yaml")
    stream = _build_stream(cfg)
    s0 = _base_signature(cfg, 42, stream)
    assert s0["n_base"] == len(stream.base_indices)
    assert "base_indices_sha1" in s0
    # 影响 base 划分的参数必须重建 stream 才能反映到签名
    for key, val in (("shuffle_seed", 99), ("initial_fraction", 0.3)):
        c = copy.deepcopy(cfg)
        c["stream"][key] = val
        assert _base_signature(c, 42, _build_stream(c)) != s0, key
    c = copy.deepcopy(cfg)
    c["split_seed"] = 7
    assert _base_signature(c, 42, _build_stream(c)) != s0
    for path, val in ((("vae", "batch_size"), 64),
                      (("vae", "beta"), 0.5)):
        c = copy.deepcopy(cfg)
        c[path[0]][path[1]] = val
        assert _base_signature(c, 42, stream) != s0, path
    c = copy.deepcopy(cfg)
    c["base_channels"] = 32
    assert _base_signature(c, 42, stream) != s0


# ======================================================================
# 23. 断点签名覆盖主要实验参数
# ======================================================================
def test_resume_signature_covers_config():
    from vae_streaming.config_utils import load_config
    from vae_streaming.experiments.engine import _state_signature

    cfg = load_config("vae_streaming/configs/cifar10.yaml")
    s0 = _state_signature(cfg, "m", 42)
    for path, val in ((("distillation", "updates_per_class"), 99),
                      (("cloud", "lr"), 0.01),
                      (("vae", "beta"), 0.1),
                      (("memory", "representative_init"), "random")):
        c = copy.deepcopy(cfg)
        c[path[0]][path[1]] = val
        assert _state_signature(c, "m", 42) != s0, path


# ======================================================================
# 24. DM 目标集合：更新轮/非更新轮统一（replay + current）
# ======================================================================
def test_dm_target_scope_unified(tmp_path):
    from vae_streaming.experiments.engine import run_method

    cfg = _tiny_engine_cfg(tmp_path)
    cfg["stream"]["num_rounds"] = 2
    res = run_method(cfg, "dm", seed=7, verbose=False)
    rec0, rec1 = res["rounds"][1], res["rounds"][2]
    assert rec0["distill"]["method"] == "streaming_dm"
    # 首轮 replay 已含 base 样本，因此从第 0 轮起目标就是 replay + current
    assert rec0["distill"]["target_scope"] == "replay_plus_current"
    assert rec1["distill"]["target_scope"] == "replay_plus_current"
    assert rec0["distill"]["n_target"] > rec0["n_current"]
    assert rec1["distill"]["n_target"] > rec1["n_current"]
    # 代表集跨版本继承，统计历史未被重置
    assert rec1["stats_scope"] == "aligned_full_history"


def _tiny_engine_cfg(tmp_path):
    from vae_streaming.config_utils import load_config
    cfg = load_config("vae_streaming/configs/cifar10.yaml")
    cfg["stream"]["initial_fraction"] = 0.10
    cfg["stream"]["num_rounds"] = 1
    cfg["memory"]["representative_budget"] = 20
    cfg["memory"]["representative_init"] = "random"
    cfg["vae"]["initial_epochs"] = 1
    cfg["vae"]["steps_per_round"] = 2
    cfg["cloud"]["steps_per_round"] = 2
    cfg["checkpoints_dir"] = str(tmp_path / "ckpt")
    cfg["results_dir"] = str(tmp_path / "res")
    cfg["figures_dir"] = str(tmp_path / "fig")
    return cfg


# ======================================================================
# 25. 更新预算与上传预算分离
# ======================================================================
def test_update_upload_budget_separation(tmp_path):
    from vae_streaming.experiments.engine import run_method

    def _cfg(p, update_fraction, sync_mode, patch_fraction):
        cfg = _tiny_engine_cfg(p)
        cfg["alignment"]["method"] = "diagonal_affine"
        cfg["distillation"]["method"] = "dm"
        cfg["distillation"]["update_fraction"] = update_fraction
        cfg["communication"]["mode"] = sync_mode
        cfg["communication"]["patch_fraction"] = patch_fraction
        return cfg

    # 10% 更新 + 100% 上传
    cfg = _cfg(tmp_path / "u10t100", 0.10, "full_sync", 0.10)
    rec = run_method(cfg, None, seed=7, verbose=False)["rounds"][1]
    assert rec["sync"]["update_fraction"] == pytest.approx(0.10)
    assert rec["sync"]["n_updated"] < rec["sync"]["n_uploaded"]
    assert rec["sync"]["n_uploaded"] == cfg["memory"]["representative_budget"]
    assert rec["sync"]["local_unsynced"]["max"] == 0.0
    assert rec["distill"]["n_optimized"] < cfg["memory"]["representative_budget"]

    # 100% 更新 + 10% 上传
    cfg2 = _cfg(tmp_path / "u100t10", 1.0, "patch", 0.10)
    rec2 = run_method(cfg2, None, seed=7, verbose=False)["rounds"][1]
    assert rec2["sync"]["update_fraction"] == pytest.approx(1.0)
    assert rec2["sync"]["n_updated"] == cfg2["memory"]["representative_budget"]
    assert rec2["sync"]["n_uploaded"] < cfg2["memory"]["representative_budget"]
    assert rec2["sync"]["local_unsynced"]["max"] > 0


# ======================================================================
# 27. 原版 DM：固定随机特征空间均值匹配
# ======================================================================
def test_random_feature_dm_objective():
    from vae_streaming.distillation.streaming_dm import StreamingDM

    torch.manual_seed(0)
    feats = torch.randn(8, 16)
    labels = torch.tensor([0] * 4 + [1] * 4)
    cfg = {"updates_per_class": 2, "lr": 0.01,
           "feature_hidden": 32, "feature_out": 16}
    dm = StreamingDM(feats, labels, [4, 4], cfg, seed=0, device="cpu")
    out = dm.update(
        real_features=torch.randn(20, 16),
        real_labels=torch.tensor([0] * 10 + [1] * 10))
    assert out["info"]["objective"] == "random_feature_mean"
    assert torch.isfinite(out["features"]).all()
    assert not torch.allclose(out["features"], feats)
    assert all(not p.requires_grad for p in dm.feature_net.parameters())
