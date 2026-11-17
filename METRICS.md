# 指标与计算方法汇总 (Metrics)

> 项目: 潜空间数据集蒸馏 → 边缘应用
> 本文档汇总所有对比实验使用的指标口径与计算公式, 与代码一一对应, 可直接用作论文方法/实验部分素材。
> **口径边界**: 除注明外, 能耗/延迟均只含下游分类器 (不含 EdgeVAE 编码器), 均为理论解析估算 (非实测)。

---

## 1. 实验体系与结论图

| 脚本 | 对比内容 | 产出 |
|---|---|---|
| `legacy/exp_ipc.py` | 压缩率-精度曲线 (固定 subset=1000/类), 多 backbone 同图 (默认 resnet+vit) | `figures/accuracy_compression_tradeoff.pdf` (+`.png`, 论文主图) |
| `legacy/exp_method_compare.py` | 与 Gradient Matching / EL2N / Forgetting 对比 (像素或潜空间); `--methods` 可追加 3 个生成式先验 | `results/method_compare_*.json`, `figures/method_x_backbone_*.pdf` |
| `legacy/latent_prior.py` | 潜空间生成式先验 (高斯 / RealNVP / latent DDPM) 训练与采样, 输出与 K-means 同尺度的 `[C,IPC,16,8,8]` | `pt/syn_latents_ipc*_{prior}_edge_s*.pt`, `results/latent_prior_*.json` |
| `legacy/exp_energy_methods.py` | 分阶段 FLOPs / 能耗 / 传输时延 (自动吸收 latent_prior 的成本元数据) | `figures/{flops,energy,transfer_latency}_by_method_*.pdf` |

默认配置: EdgeVAE (16ch×8×8, `edge`), 中心=mean, 源子集=1000/类, backbone 50 epoch 余弦衰减 Adam(lr=1e-3), seeds=3, Dropout(0.3)@分类头前。

---

## 2. 精度类指标

| 指标 | 方法 | 代码位置 |
|---|---|---|
| CNN acc | 4 backbone (vgg/resnet/mobilenet/vit), 测试集 argmax 正确率 (潜空间直接分类) | `backbones.py::train_eval_backbone` |

测试集统一: CIFAR-10 测试集 10,000 张 → 经同一冻结编码器 → 潜空间 (或原图直入 CNN)。

---

## 3. 压缩率与传输带宽 (云 → 边缘)

| 指标 | 公式 | 例 (distill, ipc=100) |
|---|---|---|
| 压缩率 | 源子集点数/类 ÷ IPC | 1000 ÷ 100 = **10:1** (可到 2:1~1000:1) |
| 总标量 | 点数 × 每样本维度 | 1000 × 1024 = 1.02M |
| fp32 字节 | 总标量 × 4 B | 4.19 MB |
| int8 字节 | 总标量 × 1 B | 1.05 MB |
| 传输时间 | 字节 × 8 ÷ 带宽 (默认 10 Mbps) | 0.84 s |

数据形式对照 (默认 ipc=100, subset=1000, C=10):

| 数据形式 | 点数 | 每样本维度 | 总标量 | fp32 | int8 | int8 耗时@10Mbps |
|---|---|---|---|---|---|---|
| 原图子集 (pixel) | 10,000 | 3072 | 30.72M | 122.88 MB | 30.72 MB | 24.58 s |
| 未压缩潜空间 (1:1) | 10,000 | 1024 | 10.24M | 40.96 MB | 10.24 MB | 8.19 s |
| 蒸馏潜空间 | 1,000 | 1024 | 1.02M | 4.19 MB | 1.05 MB | 0.84 s |

- 原图按 uint8 (1 B/标量); 潜变量默认 fp32 (4 B), 可量化 int8。
- 保守口径 (fp32 vs uint8 原图): 省 ~7.3×; 一致 int8 口径: 省 ~29×。
- 代码: `legacy/analysis.py::bytes_of_image / bytes_of_latent` (压缩率与训练能耗见 `plot_training_cost`)。

---

## 4. 计算量 (FLOPs)

- 工具: thop 库 `profile`, 输出 MACs, FLOPs = MACs × 2。
- 输入: 单张, eval 模式。
- 每管线/形态各统计一次:

| 输入形态 | 每图 FLOPs (参考) |
|---|---|
| 像素 3×32×32 | ~219-249M |
| 潜空间 16×8×8 | resnet/vgg/mobilenet ~3.6-4.2M, vit ~0.9M |

- 训练总计算量 (如需): `每图 FLOPs × (前向 1 + 反向 ~2) × 点数 × epochs`。
- 训练成本 vs 压缩率曲线 (只计下游训练, 不含 VAE 编码/蒸馏与传输): `results/training_cost_vs_ratio_{downstream}.png`
  横轴压缩率 = 源子集点数/类 ÷ IPC; 参考线为 原始全量 50000 点与 1:1 源子集 10000 点。
- 代码: `backbones.py::count_flops` (底层为 thop) + `legacy/analysis.py::plot_training_cost`。

---

## 5. 推理延迟 (EdgeProfiler 解析模型)

```
t_compute = FLOPs / (峰值FLOPs × U)
t_mem     = DRAM流量 / (带宽 × U)      # DRAM流量 = 权重字节 + 激活字节
latency   = max(t_compute, t_mem)       # 计算与内存搬运重叠, 谁慢谁主导
```

- U = 利用率 = 0.6 (常数, 可标定替换)。
- 示例: vgg 潜空间 int8 @ Jetson:
  - FLOPs = 16.18M, 峰值 33 TOPS → t_compute = 16.18e6 / (33e12×0.6) ≈ 0.82 µs
  - DRAM ≈ 200 KB, 带宽 102 GB/s → t_mem = 200e3 / (102e9×0.6) ≈ 3.3 µs
  - latency ≈ 3.3 µs (内存瓶颈)

- 代码: `utils.py::profile_inference`。

---

## 6. 推理能耗 (计算 + 内存两笔账)

```
E = FLOPs × pJ_flop + DRAM流量 × pJ_byte        # pJ = 10⁻¹² J
```

- 权重字节 = 参数量 × B (B: fp32=4, fp16=2, int8=1, int4=0.5)。
- 激活字节 ≈ (输入 numel + 2×Σ每层输出 numel − 末层输出 numel) × B (每层写一次 + 下一层读一次)。
- 示例: vgg 潜空间 int8 @ Jetson: 16.18M×2 pJ + 200KB×6 pJ ≈ **33.6 µJ**; 像素版 ~507 µJ (≈15×)。
- 仅计动态能耗, 不含静态功耗; 只计下游分类器 (不含编码器)。
- 代码: `utils.py::profile_inference`。

---

## 7. 硬件常数表 (`utils.py::DEVICES`, 可改)

| 常数 | RPi 4 | Jetson Orin Nano Super |
|---|---|---|
| 峰值算力 fp32 | 6.75 GFLOPs/s | 0.9 TFLOPs/s |
| 峰值算力 int8 | 13.5 TOPS | 33 TOPS |
| DRAM 带宽 | 4.4 GB/s | 102 GB/s |
| pJ_flop (fp32/int8) | 30 / 15 | 6 / 2 |
| pJ_byte | 15 | 6 |
| 利用率 U | 0.6 | 0.6 |

峰值/带宽取自规格, pJ 取自文献量级 (1~100 pJ/flop 中位区间), 上论文前应在目标设备实测标定。

---

## 8. 生成式先验对照口径 (legacy/latent_prior.py)

冻结同一个 EdgeVAE (16×8×8, 1024 维), 只替换「如何从类条件分布取 IPC 个点」:

| 方法 | 取点方式 | 每样本采样前向次数 (NFE) |
|---|---|---|
| Ours | 逐类 K-means 簇均值 (确定性) | 0 (只算聚类) |
| Gaussian | 逐类对角高斯采样 | 0 (无网络) |
| Flow | RealNVP 通道耦合, 逆变换采样 | 1 |
| Diffusion | latent DDPM (1024 维 MLP 去噪, x0 预测), DDIM | 50 (可扫 20/50/100) |

- 先验只在 `real_subset` (每类 1000 张) 的潜变量上拟合, 不额外用训练集标签; 采样后反标准化回 VAE `mu` 尺度。
- 精度 = 「采样 seed × 分类器 seed」全部组合的均值±标准差; 报告最后一个 epoch。
- 能耗: 先验方法 `pretrain = 共享 VAE 预训练 + 源池编码 + 先验自身训练`,
  `production = 点数 × NFE × 单次前向 FLOPs`; 传输/部署/推理与 Ours 完全相同 (同 payload)。
- Flow 保维 (dim(z)=dim(x)), 不做压缩: 3× 压缩由 EdgeVAE 完成, 论文中必须写明这一点。

---

## 9. 口径与注意事项 

1. 能耗/延迟**只含下游分类器**, 不含 EdgeVAE 编码器; 全链路需另加编码器一行。
2. 全部为理论解析估算 (峰值×U=0.6), 非实测; 不含 kernel 启动、buffer 复用。
3. 结构口径统一: 同 backbone/epochs/余弦衰减/seeds, **只变输入形态** (3×32×32 vs 16×8×8)。
4. 4 backbone (vgg/resnet/mobilenet/vit) 参数量对齐 ~28-32K, 分类头前加 Dropout(0.3); 结构改动后所有 acc 以最新版为准。ConvNeXt 因小数据易过拟合已移除。
5. 传输按原始数据口径 (未含 JPEG 等再压缩)。
