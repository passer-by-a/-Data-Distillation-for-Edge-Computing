"""固定 Initial Base Teacher 与伪标签

协议:
  - Teacher 是 Initial Base VAE 的完整快照，训练结束后冻结，不随在线更新变化。
  - 每轮只对当前 Streaming 图像做一次预测；预测结果在所有对比方法间共享。
  - 标签来源固定标记为 "initial_teacher_pseudo"，与真实标签严格区分。
  - 低置信度样本不参与任何类别条件操作（DM 目标 / 统计 / K-means / Reservoir），
    但仍可参与 VAE 的无监督（重建 + KL）训练。
"""
import copy
import hashlib
from dataclasses import dataclass

import torch

from .online_vae import to_float

PSEUDO_SOURCE = "initial_teacher_pseudo"


@dataclass
class PseudoLabeledBatch:
    round: int
    images: torch.Tensor            # uint8
    sample_ids: object              # np.ndarray
    pseudo_labels: torch.Tensor     # long [n]，全部样本的 Teacher 预测
    confidence: torch.Tensor        # float [n]
    accept: torch.Tensor            # bool [n]，高置信度筛选结果
    label_source: str = PSEUDO_SOURCE
    teacher_version: int = 0

    def __len__(self):
        return int(self.images.size(0))

    @property
    def n_accepted(self):
        return int(self.accept.sum())

    def accepted(self):
        """返回 (images, pseudo_labels, confidence) 的高置信度子集"""
        m = self.accept
        return self.images[m], self.pseudo_labels[m], self.confidence[m]

    def class_counts(self, num_classes):
        y = self.pseudo_labels[self.accept]
        return torch.bincount(y, minlength=int(num_classes)).tolist()


class FixedTeacher:
    """Initial Base VAE 的冻结快照（分类头 + Encoder）"""

    def __init__(self, vae, version=0):
        self.version = int(version)
        self.module = copy.deepcopy(vae).eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self.label_source = PSEUDO_SOURCE

    @torch.no_grad()
    def predict(self, images_u8, device="cuda", batch=256):
        """返回 (pred, confidence)，均在 CPU"""
        self.module.eval()
        preds, confs = [], []
        for i in range(0, images_u8.size(0), batch):
            x = to_float(images_u8[i:i + batch], device)
            mu = self.module.encode_mu(x)
            logits = self.module.classify(mu)
            prob = torch.softmax(logits.float(), dim=1)
            conf, pred = prob.max(dim=1)
            preds.append(pred.cpu())
            confs.append(conf.cpu())
        if not preds:
            return torch.empty(0, dtype=torch.long), torch.empty(0)
        return torch.cat(preds), torch.cat(confs)

    def make_batch(self, batch, threshold=0.9, device="cuda"):
        """UnlabeledBatch → PseudoLabeledBatch（全量预测 + 置信度掩码）"""
        pred, conf = self.predict(batch.images, device=device)
        accept = conf >= float(threshold)
        return PseudoLabeledBatch(
            round=int(batch.round), images=batch.images,
            sample_ids=batch.sample_ids, pseudo_labels=pred,
            confidence=conf, accept=accept,
            label_source=PSEUDO_SOURCE, teacher_version=self.version)

    def state_dict(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.module.state_dict().items()}

    def fingerprint(self):
        h = hashlib.sha1()
        sd = self.state_dict()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].contiguous().numpy().tobytes())
        return h.hexdigest()[:16]
