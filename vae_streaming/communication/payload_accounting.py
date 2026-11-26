"""同步消息构造与真实字节核算

消息包含: 量化代表特征、标签、稳定 ID、Encoder 版本、量化元数据、
（补丁模式下的）全局对角仿射参数与补丁位置。
字节数同时报告理论值与真实序列化文件大小。
"""
import hashlib
import io
import json

import numpy as np
import torch

from .quantization import encode_features


def select_patch_indices(quotas, fraction, round_idx):
    """确定性的类别均衡轮换: 每轮固定 Q 个可更新位置"""
    quotas = np.asarray(quotas, dtype=np.int64)
    total = int(quotas.sum())
    q_total = max(1, int(round(float(fraction) * total)))
    raw = q_total * quotas / total
    take = np.floor(raw).astype(np.int64)
    take = np.minimum(take, quotas)
    left = q_total - int(take.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while left > 0:
        c = int(order[i % len(order)])
        if take[c] < quotas[c]:
            take[c] += 1
            left -= 1
        i += 1
        if i > 100 * len(quotas):
            raise RuntimeError("补丁配额分配失败")
    idx = []
    off = 0
    for c, q in enumerate(quotas):
        if take[c] > 0:
            start = (int(round_idx) * int(take[c])) % int(q)
            pos = (start + np.arange(int(take[c]))) % int(q)
            idx.append(off + pos)
        off += int(q)
    idx = np.sort(np.concatenate(idx).astype(np.int64))
    return idx


def _sync_id(header, arrays):
    h = hashlib.sha1(json.dumps(header, sort_keys=True).encode())
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()[:20]


def build_sync_message(mode, round_idx, memory, alignment=None,
                       quant_mode="int8", patch_fraction=0.10, patch_idx=None):
    """构建上传消息。mode ∈ {full_sync, patch}。"""
    if mode not in ("full_sync", "patch"):
        raise ValueError(f"未知同步模式: {mode}")
    if not memory.initialized:
        raise RuntimeError("代表集尚未初始化")
    if mode == "patch" and alignment is None:
        raise ValueError("patch 模式必须提供跨版本对齐参数")

    if mode == "full_sync":
        idx = np.arange(memory.budget, dtype=np.int64)
    else:
        idx = select_patch_indices(memory.quotas, patch_fraction, round_idx) \
            if patch_idx is None else np.asarray(patch_idx, dtype=np.int64)
    idx_t = torch.from_numpy(idx)

    arr, qmeta = encode_features(memory.features[idx_t], quant_mode)
    version = int(memory.versions.max().item())
    header = {
        "type": mode, "round": int(round_idx), "version": version,
        "n": int(len(idx)), "dim": int(memory.dim),
        "budget": int(memory.budget), "quant": qmeta,
        "patch_fraction": float(patch_fraction),
        "feature_dtype": str(arr.dtype),
    }
    arrays = {
        "features": arr,
        "labels": memory.labels[idx_t].numpy().astype(np.int16),
        "ids": memory.ids[idx_t].numpy().astype(np.int32),
        "versions": memory.versions[idx_t].numpy().astype(np.int32),
        "patch_idx": idx.astype(np.int32),
    }
    if mode == "patch":
        a = alignment.to_arrays()
        arrays["align_scale"] = a["scale"]
        arrays["align_bias"] = a["bias"]
        header["alignment"] = alignment.meta()

    header["sync_id"] = _sync_id(header, arrays)
    msg = {"meta": json.dumps(header, ensure_ascii=False), **arrays}
    return msg, {"patch_idx": idx, "version": version,
                 "sync_id": header["sync_id"], "quant": qmeta}


def message_size(msg, tmp_dir=None):
    """理论字节、真实序列化字节与分项

    在内存中完成 npz 序列化，字节数与真实落盘一致，但不产生临时文件
    （tmp_dir 仅为兼容旧调用保留，已不再使用）。
    """
    meta_bytes = len(msg["meta"].encode("utf-8"))
    breakdown = {k: int(v.nbytes) for k, v in msg.items() if k != "meta"}
    theoretical = meta_bytes + sum(breakdown.values())
    buf = io.BytesIO()
    np.savez(buf, **{k: v for k, v in msg.items() if k != "meta"},
             meta=np.asarray(msg["meta"]))
    file_bytes = int(buf.tell())
    return {"theoretical_bytes": int(theoretical),
            "file_bytes": file_bytes,
            "overhead_bytes": int(file_bytes - theoretical),
            "meta_bytes": int(meta_bytes),
            "breakdown": breakdown}


class CommunicationLedger:
    """累计上传字节: B_cum = B_init + Σ B_sync,t"""

    def __init__(self):
        self.entries = []

    def add(self, kind, round_idx, size, extra=None):
        self.entries.append({"kind": kind, "round": int(round_idx),
                             "theoretical_bytes": int(size["theoretical_bytes"]),
                             "file_bytes": int(size["file_bytes"]),
                             "breakdown": size["breakdown"],
                             **(extra or {})})
        return self

    @property
    def init_bytes(self):
        return sum(e["theoretical_bytes"] for e in self.entries
                   if e["kind"] == "initial_sync")

    @property
    def sync_bytes(self):
        return sum(e["theoretical_bytes"] for e in self.entries
                   if e["kind"] == "round_sync")

    @property
    def cumulative_bytes(self):
        return self.init_bytes + self.sync_bytes

    @property
    def cumulative_file_bytes(self):
        return sum(e["file_bytes"] for e in self.entries)

    def summary(self):
        by_round = {}
        for e in self.entries:
            if e["kind"] == "round_sync":
                by_round[str(e["round"])] = e["theoretical_bytes"]
        return {"initial_upload_bytes": int(self.init_bytes),
                "per_round_upload_bytes": by_round,
                "cumulative_upload_bytes": int(self.cumulative_bytes),
                "cumulative_file_bytes": int(self.cumulative_file_bytes),
                "n_messages": len(self.entries),
                "downlink_bytes": 0,
                "entries": self.entries}
