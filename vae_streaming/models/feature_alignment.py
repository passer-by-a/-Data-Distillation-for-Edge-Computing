"""Feature Drift Alignment: 跨 Encoder 版本特征/统计量转换

基础方法:
  none            : 恒等（仅作诊断，跨版本混用会产生不可控误差）
  diagonal_affine : z_new ≈ a ⊙ z_old + b，a,b ∈ R^1024，逐维对角

拟合只用历史锚点（Replay Memory），一半拟合、一半留出检验。
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .online_vae import encode_mu


@dataclass
class FeatureAlignment:
    method: str
    scale: torch.Tensor                 # [D] CPU float32
    bias: torch.Tensor                  # [D] CPU float32
    version_from: int
    version_to: int
    dim: int
    n_anchors: int = 0
    n_fit: int = 0
    n_heldout: int = 0
    fit_mse: float = 0.0
    heldout_mse: float = 0.0
    relative_err: float = 0.0
    no_align_heldout_mse: float = 0.0
    no_align_relative_err: float = 0.0
    improvement_heldout: float = 0.0
    max_scale: float = 3.0
    clamped_frac: float = 0.0
    class_center_err: float = float("nan")
    per_class_center_err: dict = field(default_factory=dict)
    fit_class_center_err: float = float("nan")
    no_align_class_center_err: float = float("nan")
    class_center_improvement: float = float("nan")
    n_classes_center_fit: int = 0
    n_classes_center_heldout: int = 0
    estimated: bool = False

    # ------------------------------------------------------------------
    def apply(self, z):
        """旧版本特征 → 新版本特征"""
        return z.float() * self.scale + self.bias

    def to_arrays(self):
        return {"scale": self.scale.numpy().astype(np.float32),
                "bias": self.bias.numpy().astype(np.float32)}

    def meta(self):
        return {"method": self.method, "dim": int(self.dim),
                "version_from": int(self.version_from),
                "version_to": int(self.version_to),
                "n_anchors": int(self.n_anchors), "n_fit": int(self.n_fit),
                "n_heldout": int(self.n_heldout),
                "fit_mse": float(self.fit_mse),
                "heldout_mse": float(self.heldout_mse),
                "relative_err": float(self.relative_err),
                "no_align_heldout_mse": float(self.no_align_heldout_mse),
                "no_align_relative_err": float(self.no_align_relative_err),
                "improvement_heldout": float(self.improvement_heldout),
                "max_scale": float(self.max_scale),
                "clamped_frac": float(self.clamped_frac),
                "class_center_err": float(self.class_center_err),
                "fit_class_center_err": float(self.fit_class_center_err),
                "no_align_class_center_err": float(
                    self.no_align_class_center_err),
                "class_center_improvement": float(
                    self.class_center_improvement),
                "n_classes_center_fit": int(self.n_classes_center_fit),
                "n_classes_center_heldout": int(self.n_classes_center_heldout),
                "estimated": bool(self.estimated)}

def identity_alignment(dim, version_from, version_to):
    return FeatureAlignment(method="none", scale=torch.ones(dim),
                            bias=torch.zeros(dim), version_from=version_from,
                            version_to=version_to, dim=dim)


def _per_class_fit_mask(labels):
    """在每个类别内部交替划分拟合锚点（True）与留出锚点（False）。

    labels 通常按类别连续排列；直接在类内取偶数位置，保证每个类（只要
    ≥2 个锚点）同时进入拟合与留出，避免"只用前一半类拟合"。
    """
    labels = torch.as_tensor(labels).long().flatten()
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for c in torch.unique(labels).tolist():
        idx_c = torch.nonzero(labels == c, as_tuple=False).flatten()
        if len(idx_c) == 1:
            mask[idx_c[0]] = True
        else:
            mask[idx_c[::2]] = True
    return mask


def estimate_feature_alignment(prev_encoder, current_encoder, anchor_images,
                               anchor_labels=None, version_from=0,
                               version_to=1, max_scale=3.0,
                               method="diagonal_affine", device="cuda",
                               protocol=None):
    """用历史锚点估计旧→新的对角仿射映射。

    anchor_images 必须来自 Replay Memory（历史图像），不得包含未来数据。
    protocol 给出时，在**归一化特征空间**拟合；该空间与代表集/统计记忆一致。
    """
    old_raw = encode_mu(prev_encoder.module, anchor_images, device=device)
    new_raw = encode_mu(current_encoder, anchor_images, device=device)
    if protocol is not None:
        z_old = protocol.normalize(old_raw).double()
        z_new = protocol.normalize(new_raw).double()
    else:
        z_old = old_raw.flatten(1).double()
        z_new = new_raw.flatten(1).double()
    dim = z_old.size(1)
    n = z_old.size(0)

    if method == "none":
        return identity_alignment(dim, version_from, version_to)

    if n < 4:
        al = identity_alignment(dim, version_from, version_to)
        al.n_anchors = n
        al.estimated = True
        return al

    # 锚点由 _stratified_anchor_indices 按类别连续拼接；若直接取前一半，
    # CIFAR-10 只用类 0–4 拟合、CIFAR-100 只用类 0–49 拟合，变换却作用于
    # 所有类。这里在每个类内部交替划分 fit / heldout。
    y_all = None
    if anchor_labels is not None:
        y_all = torch.as_tensor(np.asarray(anchor_labels)).long().flatten()
        if len(y_all) != n:
            raise ValueError(f"anchor_labels 长度 {len(y_all)} != 锚点数 {n}")
        fit_mask = _per_class_fit_mask(y_all)
    else:
        fit_mask = torch.zeros(n, dtype=torch.bool)
        fit_mask[: n // 2] = True
    if not bool(fit_mask.any()) or bool(fit_mask.all()):
        # 每类只有 1 个锚点时无法类内划分，退回简单对半切
        fit_mask = torch.zeros(n, dtype=torch.bool)
        fit_mask[: n // 2] = True
    n_fit = int(fit_mask.sum())
    zo, zn = z_old[fit_mask], z_new[fit_mask]
    zo_h, zn_h = z_old[~fit_mask], z_new[~fit_mask]

    mo, mn = zo.mean(0), zn.mean(0)
    var_o = zo.var(0, unbiased=False).clamp_min(1e-8)
    a = ((zo - mo) * (zn - mn)).mean(0) / var_o
    lo, hi = 1.0 / float(max_scale), float(max_scale)
    n_clamped = int(((a < lo) | (a > hi)).sum())
    a = a.clamp(lo, hi)
    b = mn - a * mo

    def _mse(z_old_, z_new_):
        return float(((z_old_ * a + b) - z_new_).pow(2).mean())

    fit_mse = _mse(zo, zn)
    heldout_mse = _mse(zo_h, zn_h)
    denom = float(zn_h.pow(2).mean().clamp_min(1e-8))
    rel = heldout_mse / denom
    no_align_mse = float((zo_h - zn_h).pow(2).mean())
    no_align_rel = no_align_mse / denom
    improvement = 1.0 - heldout_mse / max(no_align_mse, 1e-12)

    per_class, class_err = {}, float("nan")
    fit_class_err, n_fit_cls, n_held_cls = float("nan"), 0, 0
    no_align_class_err, class_improvement = float("nan"), float("nan")

    def _center_err(zo_, zn_, yy_, apply=True):
        errs, per = [], {}
        for c in torch.unique(yy_).tolist():
            m = yy_ == c
            if int(m.sum()) < 1:
                continue
            mo_c = zo_[m].mean(0)
            mn_c = zn_[m].mean(0)
            pred = mo_c * a + b if apply else mo_c
            e = float(((pred - mn_c).pow(2).mean()) /
                      mn_c.pow(2).mean().clamp_min(1e-8))
            per[str(c)] = e
            errs.append(e)
        return (float(np.mean(errs)) if errs else float("nan")), len(errs), per

    if y_all is not None:
        fit_class_err, n_fit_cls, _ = _center_err(zo, zn, y_all[fit_mask])
        class_err, n_held_cls, per_class = _center_err(
            zo_h, zn_h, y_all[~fit_mask])
        no_align_class_err, _, _ = _center_err(zo_h, zn_h, y_all[~fit_mask],
                                               apply=False)
        if np.isfinite(no_align_class_err) and no_align_class_err > 0:
            class_improvement = 1.0 - class_err / no_align_class_err

    return FeatureAlignment(
        method=method, scale=a.float(), bias=b.float(),
        version_from=int(version_from), version_to=int(version_to), dim=dim,
        n_anchors=n, n_fit=n_fit, n_heldout=n - n_fit,
        fit_mse=fit_mse, heldout_mse=heldout_mse, relative_err=rel,
        no_align_heldout_mse=no_align_mse,
        no_align_relative_err=no_align_rel,
        improvement_heldout=improvement,
        max_scale=float(max_scale), clamped_frac=n_clamped / max(1, dim),
        class_center_err=class_err, per_class_center_err=per_class,
        fit_class_center_err=fit_class_err,
        no_align_class_center_err=no_align_class_err,
        class_center_improvement=class_improvement,
        n_classes_center_fit=int(n_fit_cls),
        n_classes_center_heldout=int(n_held_cls),
        estimated=True)
