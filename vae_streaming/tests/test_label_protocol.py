"""无标签流式协议（unlabeled_after_base）防标签泄漏测试

对应任务书第十二节 Test 1–Test 5。运行:
  python -m pytest vae_streaming/tests/test_label_protocol.py -q
"""
import copy
import json
import os

import numpy as np
import pytest
import torch

from vae_streaming.config_utils import load_config
from vae_streaming.data.causal_stream import (CausalStream, LabeledBatch,
                                              UnlabeledBatch)
from vae_streaming.experiments.engine import (_build_stream, _encoder_sha1,
                                              _module_sha1, _tensor_sha1,
                                              run_method)
from vae_streaming.models.online_vae import build_vae, encode_mu
from vae_streaming.models.pseudo_labeler import PSEUDO_SOURCE, FixedTeacher


CONFIG = "vae_streaming/configs/cifar10_unlabeled_multi80_u2.yaml"


def _stream(num_rounds=4, seed=42):
    return CausalStream(dataset="cifar10", initial_fraction=0.2,
                        num_rounds=num_rounds, seed=seed,
                        class_balanced_initial=True, scenario="iid",
                        label_mode="unlabeled_after_base")


def _tiny_cfg(root, num_rounds=2, threshold=0.5):
    cfg = load_config(CONFIG)
    cfg["stream"]["num_rounds"] = num_rounds
    cfg["stream"]["initial_fraction"] = 0.10
    cfg["memory"]["representative_budget"] = 20
    cfg["memory"]["raw_replay_capacity"] = 40
    # 确定性初始化，隔离 sklearn KMeans 的线程级浮点差异
    cfg["memory"]["representative_init"] = "random"
    cfg["vae"]["initial_epochs"] = 1
    cfg["vae"]["steps_per_round"] = 2
    cfg["vae"]["update_every_rounds"] = 2
    cfg["distillation"]["updates_per_class"] = 2
    cfg["distillation"]["feature_hidden"] = 32
    cfg["distillation"]["feature_out"] = 64
    cfg["cloud"]["steps_per_round"] = 2
    cfg["alignment"]["anchor_count"] = 16
    cfg["pseudo_labels"]["threshold"] = threshold
    cfg["checkpoints_dir"] = str(root / "ckpt")
    cfg["results_dir"] = str(root / "res")
    cfg["figures_dir"] = str(root / "fig")
    return cfg


# ======================================================================
# Test 1: 流式标签不可访问
# ======================================================================
def test_1_stream_labels_inaccessible():
    s = _stream()
    assert isinstance(s.base, LabeledBatch)
    assert s.base.label_source == "ground_truth"
    assert int((s.base.labels >= 0).all())

    stream_idx = np.concatenate(s.round_indices)
    # 流式位置的真实标签被密封为 -1
    sealed = s.train_labels[torch.from_numpy(stream_idx)]
    assert bool((sealed == -1).all()), "流式真实标签未密封"
    # 只有 base 位置保留真实标签
    kept = s.train_labels[torch.from_numpy(s.base_indices)]
    assert bool((kept >= 0).all())

    for t in range(s.num_rounds):
        b = s.round(t)
        assert isinstance(b, UnlabeledBatch)
        assert not hasattr(b, "labels")
        assert not hasattr(b, "y") and not hasattr(b, "targets")
        assert np.array_equal(b.sample_ids, s.round_indices[t])
    # summary 不泄露逐样本类别
    sm = s.summary()
    assert sm["streaming_labels_sealed"] is True
    assert sm["label_mode"] == "unlabeled_after_base"


# ======================================================================
# Test 4: 伪标签来源检查（pseudo_label == teacher_prediction）
# ======================================================================
def test_4_pseudo_labels_come_from_fixed_teacher():
    torch.manual_seed(0)
    vae = build_vae(10, latent_dim=16, base_channels=8, seed=0, device="cpu")
    teacher = FixedTeacher(vae, version=0)
    s = _stream(num_rounds=1)
    _ = s.base                       # 因果访问：base 之后才允许读流式轮次
    batch = s.round(0)

    tb = teacher.make_batch(batch, threshold=0.5, device="cpu")
    pred, conf = teacher.predict(batch.images, device="cpu")
    assert torch.equal(tb.pseudo_labels, pred), "伪标签与 Teacher 预测不一致"
    assert torch.allclose(tb.confidence, conf)
    assert torch.equal(tb.accept, conf >= 0.5)
    assert tb.label_source == PSEUDO_SOURCE
    assert tb.teacher_version == 0

    # 伪标签不能与流式真实标签产生任何交互：密封字段仍为 -1
    sealed = s.train_labels[torch.from_numpy(tb.sample_ids)]
    assert bool((sealed == -1).all())

    # Teacher 参数冻结，重复预测完全一致
    assert all(not p.requires_grad for p in teacher.module.parameters())
    fp = teacher.fingerprint()
    tb2 = teacher.make_batch(batch, threshold=0.5, device="cpu")
    assert torch.equal(tb.pseudo_labels, tb2.pseudo_labels)
    assert fp == teacher.fingerprint()


# ======================================================================
# Test 3: 缺失标签运行（训练接口没有 labels 字段，跑完整流水线不报错）
# ======================================================================
def test_3_runs_without_stream_labels(tmp_path):
    cfg = _tiny_cfg(tmp_path, num_rounds=2)
    res = run_method(cfg, "dm_multi", seed=7, verbose=False, save=False)
    assert len(res["rounds"]) == 3
    for rec in res["rounds"][1:]:
        assert rec["label_protocol"] == "unlabeled_after_base"
        assert rec["pseudo"]["label_source"] == PSEUDO_SOURCE
        assert rec["distill"]["target_scope"] == "base_replay_plus_pseudo"
    # 真实流式标签从未参与：密封字段在全程保持 -1
    stream = _build_stream(cfg)
    idx = np.concatenate(stream.round_indices)
    assert bool((stream.train_labels[torch.from_numpy(idx)] == -1).all())


# ======================================================================
# Test 2: 流式真实标签置乱后训练轨迹完全不变
# ======================================================================
class _LeakProbeStream:
    """代理对象：数据流/索引完全固定，只在每个 UnlabeledBatch 上额外挂一个
    随机的 `labels` 字段，模拟“最可能被误用的流式真实标签”存在时的情形。
    引擎若以任何方式读取该字段，轨迹就会改变。"""

    def __init__(self, inner, seed=123):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_g", np.random.RandomState(seed))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def round(self, t):
        b = object.__getattribute__(self, "_inner").round(t)
        g = object.__getattribute__(self, "_g")
        n = len(b)
        b.labels = torch.from_numpy(
            g.randint(0, 10, size=n)).long()
        return b


def test_2_stream_label_shuffle_invariance(tmp_path):
    cfg_a = _tiny_cfg(tmp_path / "a", num_rounds=2)
    cfg_b = _tiny_cfg(tmp_path / "b", num_rounds=2)
    plain = _build_stream(cfg_a)
    probe = _LeakProbeStream(plain, seed=999)

    ra = run_method(cfg_a, "dm_multi", seed=7, verbose=False, save=False,
                    stream=plain)
    rb = run_method(cfg_b, "dm_multi", seed=7, verbose=False, save=False,
                    stream=probe)

    # 数据流本身必须完全一致（没有因为置乱重新分组）
    assert ra["summary"]["stream_sha1"] == rb["summary"]["stream_sha1"]
    # Initial Base VAE / Teacher 不变
    assert ra["rounds"][0]["encoder_sha1"] == rb["rounds"][0]["encoder_sha1"]
    assert ra["summary"]["teacher_fingerprint"] == \
        rb["summary"]["teacher_fingerprint"]
    assert ra["summary"]["initial_acc"] == rb["summary"]["initial_acc"]
    # 每轮 Encoder / 代表集 / 云端参数轨迹完全一致
    for x, y in zip(ra["rounds"][1:], rb["rounds"][1:]):
        assert x["encoder_sha1"] == y["encoder_sha1"]
        assert x["memory_sha1"] == y["memory_sha1"]
        assert x["cloud_param_sha1"] == y["cloud_param_sha1"]
        assert x["acc"] == pytest.approx(y["acc"], abs=1e-9)
        assert x["f1"] == pytest.approx(y["f1"], abs=1e-9)
        assert x["distill"]["mean_loss"] == pytest.approx(
            y["distill"]["mean_loss"], rel=1e-7, abs=1e-9)


# ======================================================================
# Test 5: 预算检查
# ======================================================================
def test_5_budget_config():
    cfg = load_config(CONFIG)
    assert cfg["stream"]["label_mode"] == "unlabeled_after_base"
    assert cfg["stream"]["num_rounds"] == 80
    assert cfg["vae"]["update_every_rounds"] == 2
    assert cfg["vae"]["steps_per_round"] == 150
    assert cfg["memory"]["representative_budget"] == 980
    assert cfg["memory"]["raw_replay_capacity"] == 1000
    assert cfg["distillation"]["feature_type"] == "multi_mlp"
    assert cfg["distillation"]["num_mappings"] == 3
    assert cfg["cloud"]["select_best_on_val"] is False

    s = _build_stream(cfg)
    assert len(s.base_indices) == 9800
    assert len(s.val_labels) == 1000 and len(s.test_labels) == 10000
    sizes = [len(x) for x in s.round_indices]
    assert len(sizes) == 80 and set(sizes) == {490}
    assert sum(sizes) == 39200
    all_idx = np.concatenate(s.round_indices)
    assert len(set(all_idx.tolist())) == 39200
    assert set(all_idx.tolist()) & set(s.base_indices.tolist()) == set()

    # VAE 更新次数 / 优化步数
    n_updates = sum(1 for t in range(80)
                    if (t + 1) % cfg["vae"]["update_every_rounds"] == 0)
    assert n_updates == 40
    assert n_updates * cfg["vae"]["steps_per_round"] == 6000


def test_5_budget_from_full_run():
    """完整 80 轮结果存在时，逐项核对实际预算。"""
    path = os.path.join("vae_streaming", "results", "cifar10", "iid",
                        "dm_multi_u2_seed42.json")
    if not os.path.exists(path):
        pytest.skip("完整 80 轮结果尚未生成")
    with open(path) as f:
        res = json.load(f)
    rounds = res["rounds"][1:]
    assert res["summary"]["label_mode"] == "unlabeled_after_base"
    assert len(rounds) == 80
    assert sum(int(r["n_current"]) for r in rounds) == 39200
    assert sum(1 for r in rounds if r["vae_updated"]) == 40
    assert sum(int(r["vae"].get("steps", 0)) for r in rounds
               if r["vae_updated"]) == 6000
    assert sum(1 for r in rounds
               if r["distill"].get("method") == "streaming_dm") == 80
    assert max(int(r["n_replay"]) for r in rounds) <= 1000
    assert all(int(r["memory"]["n_features"]) == 980 for r in rounds)
    for i, r in enumerate(rounds):
        assert int(r["n_cumulative_seen"]) == 9800 + 490 * (i + 1)
    # 伪标签来源与配比审计字段齐全
    for r in rounds:
        p = r["pseudo"]
        assert p["label_source"] == PSEUDO_SOURCE
        assert 0.0 <= p["accept_rate"] <= 1.0
        assert len(p["class_counts"]) == 10
        assert int(r["distill"]["n_base_samples"]) > 0
