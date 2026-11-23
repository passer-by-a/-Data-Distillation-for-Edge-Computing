# 边缘 VAE 流式蒸馏（两阶段）

本仓库只保留论文主线的两阶段代码。

## 阶段一：静态潜空间蒸馏（`vae_bdm/`）

在冻结 EdgeVAE 的 16×8×8 潜特征上做数据集蒸馏，包括 Batch DM、K-means、
Moment、Reservoir 等方法。入口与配置见 `vae_bdm/README.md`。

## 阶段二：因果流式蒸馏（`vae_streaming/`）

流式场景下的 Online VAE、双层记忆、跨版本对齐与 Streaming DM。
各场景的 80 轮 / 392 轮协议、结果表和复现命令见
`vae_streaming/README.md`。

## 共享模块

| 文件 | 用途 |
| --- | --- |
| `vae.py` | EdgeVAE 定义与权重加载 |
| `models.py` | 冻结的 `VAEBackbone` |
| `data.py` | CIFAR-10/100 固定划分 |
| `train.py` | 特征通道统计 |
| `common.py` | 设备、随机种子、指标 |
| `random_feature.py` | DM 随机特征映射 |
| `config.py` | 路径与 DataLoader |

## 快速开始

```bash
# 单元测试
python -m pytest vae_streaming/tests/test_streaming_protocol.py -q

# 阶段二：CIFAR-10 监督流式 80 轮
python -m vae_streaming.experiments.run_iid \
  --config vae_streaming/configs/cifar10_supervised_multi80_u2.yaml \
  --methods online_batch_dm,online_multi_mlp_dm,online_kmeans,online_moment,online_reservoir \
  --seeds 42,43,44 --tag sup80
```

## 文档

- `EXPERIMENT_SUMMARY.md`：两阶段实验结果与结论
- `LABEL_PROTOCOL_AUDIT.md`：流式标签协议审计
- `U2_UNLABELED_REPORT.md`：无标签流式实验报告

## 不入库内容

`data/`、`checkpoints/`、`pt/`、`logs/`、`results/`、`figures/`、
`vae_bdm/cache/` 及各类 `__pycache__` 均通过 `.gitignore` 排除。
