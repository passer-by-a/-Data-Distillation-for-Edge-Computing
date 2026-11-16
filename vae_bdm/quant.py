"""FP32 / FP16 / 非对称 INT8 量化与真实 payload"""
import json
import os

import numpy as np
import torch


class Quantizer:
    def __init__(self, mode="fp32"):
        self.mode = mode
        self.scale = None
        self.zero_point = None
        self.vmin = None
        self.vmax = None
        self.calib_n = 0

    def fit(self, x):
        x = x.detach().float()
        self.calib_n = x.numel()
        if self.mode == "int8":
            self.vmin, self.vmax = float(x.min()), float(x.max())
            self.scale = max((self.vmax - self.vmin) / 255.0, 1e-8)
            self.zero_point = int(round(-self.vmin / self.scale))
        return self

    def quantize(self, x):
        x = x.detach().float()
        if self.mode == "fp32":
            return x
        if self.mode == "fp16":
            return x.half()
        q = torch.round(x / self.scale) + self.zero_point
        return q.clamp(0, 255).to(torch.uint8)

    def dequantize(self, q):
        if self.mode in ("fp32", "fp16"):
            return q.float()
        return (q.float() - self.zero_point) * self.scale

    def bits(self):
        return {"fp32": 32, "fp16": 16, "int8": 8}[self.mode]

    def meta(self):
        return {"mode": self.mode, "scheme": "asymmetric_uint8" if self.mode == "int8" else self.mode,
                "scale": self.scale, "zero_point": self.zero_point,
                "vmin": self.vmin, "vmax": self.vmax,
                "calib_numel": self.calib_n,
                "calibration": "distilled train representatives only"}


def quantize_pair(syn, test, mode):
    q = Quantizer(mode).fit(syn)
    return q.dequantize(q.quantize(syn)), q.dequantize(q.quantize(test)), q


def theoretical_bytes(n, dim=1024, bits=32, label_bytes=2, meta_bytes=0):
    return int(n) * dim * bits // 8 + int(n) * label_bytes + int(meta_bytes)


def save_payload(syn, labels, q, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    z = q.quantize(syn)
    if q.mode == "fp32":
        arr = z.numpy().astype(np.float32)
    elif q.mode == "fp16":
        arr = z.numpy().astype(np.float16)
    else:
        arr = z.numpy().astype(np.uint8)
    fp = os.path.join(out_dir, f"{tag}_features.bin")
    lp = os.path.join(out_dir, f"{tag}_labels.bin")
    mp = os.path.join(out_dir, f"{tag}_meta.json")
    arr.tofile(fp)
    labels.numpy().astype(np.int16).tofile(lp)
    with open(mp, "w") as f:
        json.dump({"shape": list(arr.shape), "dtype": str(arr.dtype),
                   "n": int(len(labels)), "dim": int(syn.size(1)),
                   "quant": q.meta()}, f, indent=2)
    sizes = {"features": os.path.getsize(fp), "labels": os.path.getsize(lp),
             "meta": os.path.getsize(mp)}
    sizes["total"] = sum(sizes.values())
    sizes["paths"] = {"features": fp, "labels": lp, "meta": mp}
    return sizes


def payload_table(n_full, m_distill, label_bytes=2, bandwidth_mbps=10.0,
                  meta_bytes=1024):
    out = {}
    for mode, bits in (("fp32", 32), ("fp16", 16), ("int8", 8)):
        full = theoretical_bytes(n_full, 1024, bits, label_bytes, 0)
        dist = theoretical_bytes(m_distill, 1024, bits, label_bytes, meta_bytes)
        out[mode] = {"full_bytes": full, "distill_bytes": dist,
                     "ratio": full / max(1, dist),
                     "upload_sec_full": full * 8 / (bandwidth_mbps * 1e6),
                     "upload_sec_distill": dist * 8 / (bandwidth_mbps * 1e6)}
    return out
