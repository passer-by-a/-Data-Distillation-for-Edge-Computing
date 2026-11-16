"""云端代表集状态与同步消息应用

- 云端只持有"已同步"的代表特征，不访问边缘原始图像。
- patch 模式先对全部旧特征应用全局对角仿射，再替换本轮补丁。
- sync_id 幂等：重复应用同一消息不会再次修改状态。
"""
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from ..communication.quantization import decode_features


@dataclass
class CloudState:
    features: torch.Tensor
    labels: torch.Tensor
    ids: torch.Tensor
    versions: torch.Tensor
    last_sync_id: str = None
    initialized: bool = False
    history: list = field(default_factory=list)

    @classmethod
    def empty(cls, budget, dim=1024):
        return cls(features=torch.zeros(budget, dim),
                   labels=torch.full((budget,), -1, dtype=torch.long),
                   ids=torch.arange(budget, dtype=torch.long),
                   versions=torch.full((budget,), -1, dtype=torch.long))

    # ------------------------------------------------------------------
    def apply(self, message):
        """返回 True 表示状态被修改，False 表示重复消息被忽略"""
        header = json.loads(message["meta"])
        sync_id = header["sync_id"]
        if sync_id == self.last_sync_id:
            return False
        if not self.initialized and header["type"] != "full_sync":
            raise RuntimeError("云端尚未收到初始完整同步，不能应用补丁")

        if header["type"] == "full_sync":
            if int(header["n"]) != len(self.features):
                raise ValueError("完整同步数量与云端预算不一致")
            self.features = decode_features(message["features"],
                                            header["quant"])
            self.labels = torch.from_numpy(
                np.asarray(message["labels"]).astype(np.int64))
            self.ids = torch.from_numpy(
                np.asarray(message["ids"]).astype(np.int64))
            self.versions.fill_(int(header["version"]))
            self.initialized = True
        else:
            scale = torch.from_numpy(np.asarray(message["align_scale"],
                                                dtype=np.float32))
            bias = torch.from_numpy(np.asarray(message["align_bias"],
                                               dtype=np.float32))
            self.features = self.features * scale + bias
            idx = torch.from_numpy(np.asarray(message["patch_idx"],
                                              dtype=np.int64))
            self.features[idx] = decode_features(message["features"],
                                                 header["quant"])
            self.labels[idx] = torch.from_numpy(
                np.asarray(message["labels"]).astype(np.int64))
            self.ids[idx] = torch.from_numpy(
                np.asarray(message["ids"]).astype(np.int64))
            self.versions.fill_(int(header["version"]))

        self.last_sync_id = sync_id
        self.history.append({"round": header["round"], "type": header["type"],
                             "version": header["version"],
                             "n": header["n"], "sync_id": sync_id})
        return True

    # ------------------------------------------------------------------
    def verify_consistent(self, memory, atol=1e-4):
        """与边缘提交视图逐项核对"""
        ok_f = torch.allclose(self.features, memory.submitted, atol=atol,
                              rtol=0)
        ok_l = torch.equal(self.labels, memory.labels)
        ok_i = torch.equal(self.ids, memory.ids)
        ok_v = torch.equal(self.versions, memory.submitted_versions)
        return {"features_ok": bool(ok_f), "labels_ok": bool(ok_l),
                "ids_ok": bool(ok_i), "versions_ok": bool(ok_v),
                "max_feature_diff": float((self.features - memory.submitted)
                                          .abs().max()),
                "all_ok": bool(ok_f and ok_l and ok_i and ok_v)}

    def nbytes(self):
        return int(self.features.numel() * self.features.element_size() +
                   self.labels.numel() * self.labels.element_size() +
                   self.ids.numel() * self.ids.element_size() +
                   self.versions.numel() * self.versions.element_size())

    def state_dict(self):
        return {"features": self.features.clone(),
                "labels": self.labels.clone(), "ids": self.ids.clone(),
                "versions": self.versions.clone(),
                "last_sync_id": self.last_sync_id,
                "initialized": self.initialized,
                "history": list(self.history)}

    @classmethod
    def from_state_dict(cls, st):
        return cls(features=st["features"].clone(),
                   labels=st["labels"].clone(), ids=st["ids"].clone(),
                   versions=st["versions"].clone(),
                   last_sync_id=st["last_sync_id"],
                   initialized=bool(st["initialized"]),
                   history=list(st["history"]))


def apply_sync_message(cloud_state, message):
    changed = cloud_state.apply(message)
    return cloud_state, {"changed": bool(changed),
                         "round": json.loads(message["meta"])["round"],
                         "last_sync_id": cloud_state.last_sync_id}
