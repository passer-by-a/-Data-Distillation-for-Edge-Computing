"""
共享工具: 测试集缓存 + 边缘推理代价分析模型

两部分:
  1. 测试集缓存 (各对比脚本共用)
     encode_test_latents: 测试图 → 潜变量 (只编码一次, 供每 epoch test loss / 最终 acc 复用)
     flat_test_cache:     测试图 flat 像素 (供原图直训管线复用)

  2. EdgeProfiler-lite: 轻量分析模型, 估算小 CNN 在边缘设备上的推理
     能耗 / 延迟 / 内存足迹 (借鉴 arXiv:2506.09061 EdgeProfiler 的方法论, 从 LLM 移植到 CNN)
     与"能耗 = FLOPs × 10 pJ"的单点估算不同, 这里拆成:
       能耗 = 计算能耗 + DRAM 内存访问能耗
            = FLOPs × pJ/flop + DRAM字节 × pJ/byte
       延迟 = max(计算时间, 内存时间)
            = max(FLOPs / (峰值FLOPs×U), DRAM字节 / (带宽×U))
       内存足迹 = 权重字节(参数量 × 精度字节) + 每图峰值激活

运行 (纯解析, 不需 GPU, 几秒出全表):
  python utils.py --device all --precision fp32,int8
"""

import argparse

import torch
import torch.nn as nn


# ============================================================
# 1. 测试集缓存
# ============================================================

@torch.no_grad()
def encode_test_latents(test_loader, ae, device):
    """把测试集整体编码成潜变量 (只一次), 供每 epoch test loss / 最终 acc 复用"""
    latents, labels = [], []
    for bx, by in test_loader:
        latents.append(ae.encode_mu(bx.to(device)).flatten(1).cpu())
        labels.append(by)
    return torch.cat(latents, 0), torch.cat(labels, 0)


def flat_test_cache(test_loader):
    """原图像素测试集 [N, 3*32*32] + labels (flat, 供 train_eval_backbone)"""
    zs, ys = [], []
    for x, y in test_loader:
        zs.append(x.flatten(1))
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


# ============================================================
# 2. 硬件 profile (量级参考值, 可按实际设备修改)
# ============================================================
DEVICES = {
    "rpi4": {
        "name": "Raspberry Pi 4 (CPU, 4×Cortex-A72)",
        "flops": {"fp32": 6.75e9, "fp16": 6.75e9, "int8": 13.5e9, "int4": 13.5e9},
        "pj_flop": {"fp32": 30.0, "fp16": 30.0, "int8": 15.0, "int4": 15.0},
        "dram_bw": 4.4e9,          # LPDDR4 字节/s
        "pj_byte": 15.0,           # DRAM 每字节能耗
        "util": 0.6,               # 利用率 (论文 U_compute/U_memory)
    },
    "jetson_orin_nano_super": {
        "name": "Jetson Orin Nano Super (GPU)",
        "flops": {"fp32": 0.9e12, "fp16": 16e12, "int8": 33e12, "int4": 33e12},
        "pj_flop": {"fp32": 6.0, "fp16": 3.0, "int8": 2.0, "int4": 2.0},
        "dram_bw": 102e9,          # LPDDR5 字节/s
        "pj_byte": 6.0,
        "util": 0.6,
    },
    "rtx4090d": {
        # RTX 4090 D (中国特供版): 14592 CUDA 核, boost 2520MHz, TDP 425W
        # FP32 73.54 TFLOPS; FP16 张量核 (dense) 294.2 TFLOPS; INT8 张量核 (dense) 588.3 TOPS
        # 24GB GDDR6X 384bit @21Gbps -> 1008 GB/s
        # pJ/flop 按满载折算: 425W / 峰值 (fp32=5.8, fp16=1.4, int8=0.72)
        "name": "RTX 4090 D (desktop GPU)",
        "flops": {"fp32": 73.54e12, "fp16": 294.2e12, "int8": 588.3e12, "int4": 1176.6e12},
        "pj_flop": {"fp32": 5.8, "fp16": 1.5, "int8": 0.8, "int4": 0.4},
        "dram_bw": 1008e9,        # GDDR6X 字节/s
        "pj_byte": 8.0,           # GDDR6X 量级参考 (未实测)
        "util": 0.6,
    },
}

PRECISION_BYTES = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def profile_inference(model, x, precision="fp32", device="jetson_orin_nano_super"):
    """单张输入推理的 能耗/延迟/内存 解析估算

    返回 dict:
      flops, params, weight_bytes, act_bytes, dram_bytes
      t_compute_ms, t_mem_ms, latency_ms, energy_uj
    """
    dev = DEVICES[device]
    B = PRECISION_BYTES[precision]
    was_training = model.training
    model.eval()

    # 参数量
    params = sum(p.numel() for p in model.parameters())

    # 逐层输出激活统计 (Conv2d/Linear/BN 记为特征图; ReLU/GELU 同尺寸不计, 复用缓冲)
    out_numels, in_numel_first = [], x.numel()

    def hook(m, inp, out):
        out_numels.append(out.numel())

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handles.append(m.register_forward_hook(hook))

    # FLOPs (thop 库, MACs×2)
    try:
        from thop import profile
    except ImportError as e:
        for h in handles:
            h.remove()
        raise ImportError("请先安装 thop: pip install thop") from e
    macs, _ = profile(model, inputs=(x,), verbose=False)
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    flops = int(macs) * 2
    # 激活 DRAM 流量 ≈ 输入一次 + 每层输出写一次 + 下一层读一次 (末层输出只写不读)
    sum_out = sum(out_numels)
    act_elems = in_numel_first + 2 * sum_out - out_numels[-1] if out_numels else in_numel_first
    act_bytes = act_elems * B
    weight_bytes = params * B
    dram_bytes = weight_bytes + act_bytes

    t_compute = flops / (dev["flops"][precision] * dev["util"])
    t_mem = dram_bytes / (dev["dram_bw"] * dev["util"])
    latency = max(t_compute, t_mem)
    energy_j = flops * dev["pj_flop"][precision] * 1e-12 \
        + dram_bytes * dev["pj_byte"] * 1e-12

    return {
        "flops": flops, "params": params,
        "weight_bytes": weight_bytes, "act_bytes": act_bytes,
        "dram_bytes": dram_bytes,
        "t_compute_ms": t_compute * 1e3, "t_mem_ms": t_mem * 1e3,
        "latency_ms": latency * 1e3, "energy_uj": energy_j * 1e6,
    }


def print_matrix(backbones, device="jetson_orin_nano_super", precisions=("fp32", "int8")):
    """打印: 像素(3×32×32) vs 潜空间(16×8×8) × 四 backbone 的能耗/延迟/内存矩阵"""
    inputs = {"pixel 3×32×32": (3, 32, 32), "latent 16×8×8": (16, 8, 8)}
    dev = DEVICES[device]
    for prec in precisions:
        print("\n" + "=" * 118)
        print(f"设备: {dev['name']} | 精度: {prec} ({PRECISION_BYTES[prec]}B/参数)")
        print("=" * 118)
        print(f"{'backbone':<22}{'输入':>16}{'FLOPs':>11}{'权重KB':>9}{'激活KB/图':>11}"
              f"{'t计算ms':>9}{'t内存ms':>9}{'延迟ms':>9}{'能耗uJ':>9}")
        print("-" * 118)
        for name, cls in backbones.items():
            for tag, (ch, h, w) in inputs.items():
                m = cls(in_ch=ch, num_classes=10).eval()
                x = torch.randn(1, ch, h, w)
                r = profile_inference(m, x, precision=prec, device=device)
                print(f"{name:<22}{tag:>16}{r['flops']:>11,}"
                      f"{r['weight_bytes'] / 1e3:>9.1f}{r['act_bytes'] / 1e3:>10.1f}"
                      f"{r['t_compute_ms']:>9.3f}{r['t_mem_ms']:>9.3f}"
                      f"{r['latency_ms']:>9.3f}{r['energy_uj']:>9.2f}")
    print("=" * 118)


if __name__ == "__main__":
    # 延迟 import 避免 utils ↔ backbones 循环
    from backbones import BACKBONES

    parser = argparse.ArgumentParser(description="EdgeProfiler-lite: 边缘推理能耗/延迟/内存分析")
    parser.add_argument("--device", type=str, default="all",
                        choices=["all", "rpi4", "jetson_orin_nano_super", "rtx4090d"])
    parser.add_argument("--precision", type=str, default="fp32,int8",
                        help="逗号分隔: fp32/fp16/int8/int4")
    args = parser.parse_args()

    devices = list(DEVICES) if args.device == "all" else [args.device]
    precs = [p.strip() for p in args.precision.split(",") if p.strip()]
    for d in devices:
        print_matrix(BACKBONES, device=d, precisions=precs)
