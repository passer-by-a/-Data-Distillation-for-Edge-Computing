# Group-VAE：共享 Stem 的多图联合编码

目标：把 K 张同类别训练图像在 **VAE 编码器内部**融合成一个 1024 维组级潜变量，
替代“先单图编码、再 K-means/BDM 聚合”的两阶段流程。

## 1. 现有 VAE 审计与拆分点

`vae.py` 的 `EdgeEncoder`：

```text
输入 [B,3,32,32]
  conv1                         -> [B,64,32,32]
  down[0] (ResBlock+Conv4x4s2)  -> [B,64,16,16]
  down[1] (ResBlock+Conv4x4s2)  -> [B,64,8,8]
  mid (ResBlock×2)              -> [B,64,8,8]
  mu_conv / logvar_conv         -> [B,16,8,8]
```

| 融合尺度 | Shared Stem | 组级融合输入 | Shared Deep Tail |
|---|---|---|---|
| **8×8（默认）** | conv1 + down[0] + down[1] | `[B,K,64,8,8]` | mid + mu/logvar 头 |
| 16×16 | conv1 + down[0] | `[B,K,64,16,16]` | down[1] + mid + mu/logvar 头 |

融合发生在 `mu/logvar` 头之前，不会先生成 K 个 1024 维潜变量。

## 2. 模型

- `models/shared_stem.py`：`StemTail` 直接引用原 VAE 模块，不复制参数。
- `models/set_attention.py`：1 层 4 头 masked MHA + FFN，只在 K 个 token 上做注意力，无位置编码。
- `models/group_fusion.py`：FiLM（gate=0，初始化恒等）+ masked attention pooling（score 初始为 0，初始等价均值池化）。
- `models/group_vae.py`：
  - `encode_single(x)`：stem → tail → `mu` → 逐通道 z-score → 1024 维。
  - `encode_group(x, mask)`：stem → Set Attention/FiLM/Pooling → tail → `mu` → 同一个 z-score → 1024 维。
  - `forward_joint`：一次 stem 前向同时产出单图 VAE 路径与组级路径。
- K=1 时三条融合路径在初始化时与 `encode_single` 数值一致（测试覆盖）。

特征归一化与 `vae_bdm` 完全一致：`vae_bdm/cache/{dataset}/norm_stats.pt` 的逐通道 z-score。

## 3. 分组

`data/group_sampler.py`：
- 类内分组，`M = round(49000 / ratio)`，配额用最大余数法，保证总数精确。
- `group.size` 默认 10；20:1 时建议 `--group_size 20`。
- 策略：`random`、`nearest`（MiniBatchKMeans 簇内最近 K）、`diverse`（簇内最远 K）。
- 每组返回 `images [K,3,32,32]`、`mask [K]`、`label`；padding 不参与注意力与池化。
- 只使用训练索引；测试集与验证集不参与分组。

## 4. 训练

### Stage 1：Group module warm-up

```bash
python -m group_vae.training.train_group_warmup \
  --config group_vae/configs/cifar10.yaml --ratio 10 \
  --fusion film --lambda_dm 1 --tag gvae_film_dm_c10_r10 --epochs 20
```

- VAE 冻结（`eval()`，BN 固定），只训练 Set Attention / FiLM / Pooling / `group_head`。
- 损失：`λ_dm·(mean + λ_var·var) + λ_gce·CE + λ_gkl·KL`。
- `--lambda_dm 0` 关闭 DM，用于 GroupVAE-Mean / Attention / FiLM 的纯 CE 对照。
- 5% 训练组做 holdout，按 holdout group CE 选 checkpoint。

### Stage 2：Joint fine-tuning

```bash
python -m group_vae.training.train_group_joint \
  --config group_vae/configs/cifar10.yaml --ratio 10 \
  --fusion attention --init_group group_vae/results/cifar10/gvae_attn_dm_c10_r10.pt \
  --tag gvae_joint_c10_r10 --epochs 5
```

- 解冻 VAE；单图路径使用重建 + KL + CE，组级路径使用 DM + CE + KL。
- DM 的单图参考特征使用 stop-grad 当前编码器。

## 5. 评估

```bash
# 1) 导出组级代表特征与单图 val/test/train 特征
python -m group_vae.evaluation.encode_groups \
  --config group_vae/configs/cifar10.yaml --ratio 10 \
  --fusion film --ckpt group_vae/results/cifar10/gvae_film_dm_c10_r10.pt \
  --tag gvae_film_dm_c10_r10

# 2) 云端训练（v2 external validation，FP32/FP16/INT8）
python -m group_vae.evaluation.train_cloud \
  --config group_vae/configs/cifar10.yaml --tag gvae_film_dm_c10_r10

# 3) 当前编码器空间中的 K-means/Herding/BDM 基线
python -m group_vae.evaluation.compare_baselines \
  --config group_vae/configs/cifar10.yaml --ratio 10 \
  --tag gvae_film_dm_c10_r10 --methods kmeans,herding,bdm

# 4) 分布差异
python -m group_vae.evaluation.feature_analysis \
  --config group_vae/configs/cifar10.yaml --tag gvae_film_dm_c10_r10

# 5) 汇总
python -m group_vae.evaluation.summarize_results \
  --config group_vae/configs/cifar10.yaml --ratio 10
```

## 6. 测试

```bash
python -m pytest group_vae/tests -q
```

覆盖：输出形状、K=1 单图一致性、置换不变性、padding 不变性、
冻结/联合梯度、无潜变量聚合、分组泄漏与预算、mask。

## 7. 当前结果

运行后写入 `group_vae/results/` 与 `group_vae/figures/`；本 README 不预填数字。
