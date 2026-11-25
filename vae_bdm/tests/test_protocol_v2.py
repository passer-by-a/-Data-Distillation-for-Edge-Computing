"""VAE-BDM v2 协议单元测试（CPU 可运行）

运行:
  python -m pytest vae_bdm/tests/test_protocol_v2.py -q
"""
import json
import os

import pytest
import torch

from vae_bdm.baselines import moment_calibration
from vae_bdm.bdm import BatchDM
from vae_bdm.cloud import (LinearCloud, MLPCloud, quick_val_acc,
                           train_cloud_classifier)
from vae_bdm.evaluate import eval_reps


DIM = 1024


def _data(C=3, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(C * n, DIM, generator=g)
    y = torch.arange(C).repeat_interleave(n)
    return x, y


def _cfg(**kw):
    cfg = dict(feature_dim=DIM, updates_per_class=2,
               lr=0.01, weight_decay=0.0, lambda_mean=1.0,
               selection="final", kmeans_seed=0, kmeans_n_init=1,
               kmeans_max_iter=10, real_batch_size=16, log_every=10,
               eval_every=100)
    cfg.update(kw)
    return cfg


def test_class_isolation():
    x, y = _data()
    dm = BatchDM(x, y, [3, 3, 3], 3, _cfg(), seed=42, device="cpu")
    before = [p.detach().clone() for p in dm.S_by_class]
    dm._step(1)
    assert torch.allclose(before[1], dm.S_by_class[1].detach()) is False
    for c in (0, 2):
        assert torch.equal(before[c], dm.S_by_class[c].detach())


def test_balanced_update_counts():
    x, y = _data()
    dm = BatchDM(x, y, [3, 3, 3], 3, _cfg(updates_per_class=5),
                 seed=42, device="cpu")
    out = dm.fit(verbose=False)
    assert out["class_update_counts"] == [5, 5, 5]
    assert out["total_optimizer_steps"] == 15
    assert dm.class_update_counts.min() == dm.class_update_counts.max()


def test_gradients_flow():
    x, y = _data()
    dm = BatchDM(x, y, [3, 3, 3], 3, _cfg(), seed=42, device="cpu")
    loss, stats = dm._losses(0)
    dm.optimizers[0].zero_grad(set_to_none=True)
    loss.backward()
    g = dm.S_by_class[0].grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert torch.isfinite(loss).all()


def test_random_feature_mean_match():
    """原版 DM：损失等于固定随机特征空间中的逐类均值匹配。"""
    x, y = _data()
    dm = BatchDM(x, y, [3, 3, 3], 3, _cfg(real_batch_size=30),
                 seed=42, device="cpu")
    loss, _ = dm._losses(1)
    manual = (dm._class_feat(1).mean(0)
              - dm.feature_net(dm.S_by_class[1]).mean(0)).pow(2).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    dm.optimizers[1].zero_grad(set_to_none=True)
    loss.backward()
    assert dm.S_by_class[1].grad is not None
    assert all(dm.S_by_class[c].grad is None for c in (0, 2))


def test_external_validation_n_train():
    x, y = _data(C=3, n=20)
    xv, yv = _data(C=3, n=20, seed=9)
    model, info = train_cloud_classifier(
        x, y, head="linear", num_classes=3, epochs=2, device="cpu",
        val_features=xv, val_labels=yv)
    assert info["validation_mode"] == "external"
    assert info["n_train"] == len(x)
    assert info["n_val"] == len(xv)


def test_low_ipc_cifar100():
    C, q = 100, 5
    x, y = _data(C=C, n=q, seed=1)
    dm = BatchDM(x, y, [q] * C, C, _cfg(updates_per_class=1,
                                        real_batch_size=5, log_every=1000),
                 seed=42, device="cpu")
    out = dm.fit(verbose=False)
    assert out["features"].shape == (C * q, DIM)
    assert out["class_update_counts"] == [1] * C


def test_validation_heads():
    x, y = _data(C=3, n=15)
    xv, yv = _data(C=3, n=10, seed=7)
    for head, cls in (("linear", LinearCloud), ("mlp", MLPCloud)):
        model, info = train_cloud_classifier(
            x, y, head=head, num_classes=3, epochs=2, device="cpu",
            val_features=xv, val_labels=yv)
        assert isinstance(model, cls)
        assert info["head"] == head
        assert info["validation_mode"] == "external"


def test_rng_isolation():
    x, y = _data(C=3, n=10)
    xv, yv = _data(C=3, n=10, seed=3)
    before = torch.get_rng_state().clone()
    quick_val_acc(x, y, xv, yv, num_classes=3, epochs=2, seed=0, device="cpu")
    assert torch.equal(before, torch.get_rng_state())


def test_moment_calibration():
    real_stats = {0: (torch.full((DIM,), 2.0), torch.full((DIM,), 4.0)),
                  1: (torch.full((DIM,), -1.0), torch.full((DIM,), 0.25))}
    g = torch.Generator().manual_seed(0)
    reps = torch.cat([torch.randn(3, DIM, generator=g),
                      torch.randn(3, DIM, generator=g) * 2.0], 0)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    out, info = moment_calibration(reps, labels, real_stats, eps=1e-8,
                                   max_scale=1e6)
    for c, (mu, var) in real_stats.items():
        z = out[labels == c]
        assert torch.allclose(z.mean(0), mu, atol=1e-4)
        assert torch.allclose(z.var(0, unbiased=False), var, atol=1e-4)
    assert info["post_mean_mse"] < info["pre_mean_mse"]
    assert info["post_var_mse"] < info["pre_var_mse"]

    # 缩放截断时不保证逐维方差完全匹配，但误差应下降
    _, info_c = moment_calibration(reps, labels, real_stats, eps=1e-8,
                                   max_scale=1.5)
    assert info_c["clamped_frac"] > 0
    assert info_c["post_var_mse"] < info_c["pre_var_mse"]

    # 少于 2 个代表特征的类跳过
    out2, info2 = moment_calibration(torch.zeros(3, DIM),
                                     torch.tensor([0, 1, 1]), real_stats)
    assert info2["skipped"] == [0]
    assert torch.allclose(out2[0], torch.zeros(DIM))


def test_rejects_legacy_cloud_protocol():
    x, y = _data(C=3, n=10)
    xv, yv = _data(C=3, n=5, seed=2)
    cfg = {"num_classes": 3,
           "cloud": {"head": "linear", "lr": 1e-3, "wd": 1e-4, "batch": 8,
                     "val_mode": "internal", "val_frac": 0.1,
                     "training_budget": {"mode": "fixed_epochs", "epochs": 1}}}
    with pytest.raises(ValueError):
        eval_reps(x, y, xv, yv, cfg, [0], val={"x": xv, "y": yv})


def test_rejects_legacy_bdm_config():
    x, y = _data()
    cfg = _cfg()
    cfg.pop("updates_per_class")
    with pytest.raises(ValueError):
        BatchDM(x, y, [3, 3, 3], 3, cfg, seed=42, device="cpu")


def test_legacy_results_still_readable():
    old = "vae_bdm/results/cifar10_bdm_r10.json"
    if os.path.exists(old):
        rows = json.load(open(old))["rows"]
        assert rows and "acc_mean" in rows[0]
