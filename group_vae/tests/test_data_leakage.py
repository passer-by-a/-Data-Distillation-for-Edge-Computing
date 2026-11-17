"""分组只使用训练索引、类内同质、预算精确"""
import numpy as np

from group_vae.data.group_sampler import build_groups


def test_group_leakage_and_budget():
    rng = np.random.RandomState(0)
    n, c, k, m = 120, 3, 4, 30
    labels = np.repeat(np.arange(c), n // c)
    g = build_groups(labels, m, k, "random", seed=0)
    assert len(g["labels"]) == m
    assert g["indices"].shape == (m, k)
    for i in range(m):
        valid = g["indices"][i][g["mask"][i]]
        assert len(valid) >= 1
        assert (valid >= 0).all() and (valid < len(labels)).all()
        assert len(set(labels[valid].tolist())) == 1
        assert (labels[valid] == g["labels"][i]).all()


def test_group_mask():
    labels = np.repeat(np.arange(2), 10)
    g = build_groups(labels, 5, 5, "random", seed=1)
    total = 0
    for i in range(len(g["labels"])):
        assert 1 <= g["mask"][i].sum() <= 5
        total += int(g["mask"][i].sum())
    assert total == 20
