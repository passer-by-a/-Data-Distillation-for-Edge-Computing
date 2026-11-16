"""代表特征量化: 与第一阶段 VAE-BDM 相同的非对称 uint8 协议

复用 vae_bdm.quant.Quantizer，不引入新的量化口径。
"""
import numpy as np
import torch

from vae_bdm.quant import Quantizer


def encode_features(features, mode="int8"):
    """返回可序列化数组与量化元数据"""
    f = torch.as_tensor(features).float().cpu()
    q = Quantizer(mode).fit(f)
    data = q.quantize(f)
    if mode == "int8":
        arr = data.numpy().astype(np.uint8)
    elif mode == "fp16":
        arr = data.numpy().astype(np.float16)
    else:
        arr = data.numpy().astype(np.float32)
    return arr, q.meta()


def decode_features(data, meta):
    """量化数组 → float32 特征"""
    mode = meta["mode"]
    q = Quantizer(mode)
    if mode == "int8":
        q.scale = float(meta["scale"])
        q.zero_point = int(meta["zero_point"])
        x = torch.from_numpy(np.asarray(data).astype(np.uint8))
        return q.dequantize(x).float()
    if mode == "fp16":
        return torch.from_numpy(np.asarray(data).astype(np.float16)).float()
    return torch.from_numpy(np.asarray(data).astype(np.float32)).float()


def roundtrip_error(features, mode="int8"):
    arr, meta = encode_features(features, mode)
    rec = decode_features(arr, meta)
    f = torch.as_tensor(features).float().cpu()
    return {"mse": float((rec - f).pow(2).mean()),
            "max_abs": float((rec - f).abs().max()),
            "bits": {"int8": 8, "fp16": 16, "fp32": 32}[mode]}
