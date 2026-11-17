"""Shared-Stem Cross-Image Group-VAE

单图路径: stem -> tail -> mu (1024, 归一化)
组级路径: stem -> Set Attention/FiLM/Pooling -> tail -> 组级 mu (1024)

融合发生在 mu/logvar 头之前；不会先生成 K 个 1024 维潜变量。
K=1 时三条融合路径在初始化时与 encode_single 数值一致。
"""
import torch
import torch.nn as nn

from .group_fusion import AttentionPool, FiLM
from .set_attention import SetAttention
from .shared_stem import StemTail


class GroupVAE(nn.Module):
    def __init__(self, vae, num_classes, cfg):
        super().__init__()
        self.vae = vae
        self.num_classes = num_classes
        self.fusion = cfg.get("fusion", "film")
        self.stage = cfg.get("fusion_stage", "8x8")
        if self.fusion not in ("mean", "attention", "film"):
            raise ValueError(f"未知 fusion: {self.fusion}")
        self.st = StemTail(vae, self.stage)
        c = self.st.channels
        dim = int(cfg.get("attn_dim", 128))
        heads = int(cfg.get("attn_heads", 4))
        layers = int(cfg.get("attn_layers", 1))
        hidden = int(cfg.get("film_hidden", 128))
        if self.fusion != "mean":
            self.set_attn = SetAttention(c, dim, heads, layers)
            self.pool = AttentionPool(dim, hidden)
        else:
            self.set_attn = None
            self.pool = None
        self.film = FiLM(dim, c, hidden) if self.fusion == "film" else None
        self.group_head = nn.Linear(1024, num_classes)
        self.register_buffer("norm_mean", torch.zeros(1, 16, 1, 1))
        self.register_buffer("norm_std", torch.ones(1, 16, 1, 1))

    # ---------------- 特征归一化 ----------------
    def set_feature_norm(self, mean, std):
        self.norm_mean.copy_(mean.detach().cpu().float().view(1, -1, 1, 1))
        self.norm_std.copy_(std.detach().cpu().float().view(1, -1, 1, 1))

    def _normalize(self, mu):
        return ((mu - self.norm_mean) / self.norm_std.clamp_min(1e-6)).flatten(1)

    # ---------------- 共享 stem / tail ----------------
    def _stem_maps(self, images, mask):
        b, k = images.shape[:2]
        flat = images.reshape(b * k, *images.shape[2:])
        h = self.st.stem(flat)
        h = h.view(b, k, self.st.channels, self.st.spatial, self.st.spatial)
        return h * mask[..., None, None, None]

    def _fuse(self, h, mask):
        b, k, c, hh, ww = h.shape
        if self.fusion == "mean":
            w = mask.float() / mask.sum(1, keepdim=True).clamp_min(1)
            g = (w[..., None, None, None] * h).sum(1)
            return g, w.unsqueeze(-1), None
        tokens = (h.sum((-2, -1))
                  / mask.sum(1, keepdim=True).clamp_min(1).float().unsqueeze(-1))
        ctx = self.set_attn(tokens, mask)
        if self.film is not None:
            h = self.film(h, ctx, mask)
        g, w = self.pool(h, ctx, mask)
        return g, w, ctx

    # ---------------- 单图路径 ----------------
    def encode_single(self, x):
        mu, _ = self.st.tail(self.st.stem(x))
        return self._normalize(mu)

    # ---------------- 组级路径 ----------------
    def encode_group(self, images, mask=None, return_weights=False,
                     return_logvar=False):
        if mask is None:
            mask = torch.ones(images.shape[:2], dtype=torch.bool,
                              device=images.device)
        h = self._stem_maps(images, mask)
        g, w, _ = self._fuse(h, mask)
        mu, logvar = self.st.tail(g)
        feat = self._normalize(mu)
        outs = [feat]
        if return_logvar:
            outs.append(logvar)
        if return_weights:
            outs.append(w)
        return tuple(outs) if len(outs) > 1 else feat

    def forward_group(self, images, mask, with_single_ref=True):
        b, k = images.shape[:2]
        h = self._stem_maps(images, mask)
        g, w, _ = self._fuse(h, mask)
        mu_g, logvar_g = self.st.tail(g)
        feat_g = self._normalize(mu_g)
        out = {"group_mu": feat_g, "group_mu_raw": mu_g,
               "group_logvar": logvar_g,
               "group_logits": self.group_head(feat_g),
               "weights": w}
        if with_single_ref:
            with torch.no_grad():
                mu_s, _ = self.st.tail(h.detach().reshape(
                    b * k, self.st.channels, self.st.spatial, self.st.spatial))
                out["single_mu"] = self._normalize(mu_s).view(b, k, 1024)
        return out

    def forward_joint(self, images, mask):
        """单图 VAE 损失 + 组级路径一次前向（stem 共享）"""
        b, k = images.shape[:2]
        h = self._stem_maps(images, mask)
        flat_h = h.reshape(b * k, self.st.channels, self.st.spatial,
                           self.st.spatial)
        mu_s, logvar_s = self.st.tail(flat_h)
        z = mu_s + torch.randn_like(logvar_s) * torch.exp(0.5 * logvar_s)
        recon = self.vae.decode(z)
        logits_s = self.vae.classify(z)
        g, w, _ = self._fuse(h, mask)
        mu_g, logvar_g = self.st.tail(g)
        feat_g = self._normalize(mu_g)
        return {"recon": recon, "logits_single": logits_s,
                "mu_single": mu_s, "logvar_single": logvar_s,
                "single_mu": self._normalize(mu_s).view(b, k, 1024),
                "valid": mask.reshape(-1),
                "group_mu": feat_g, "group_mu_raw": mu_g,
                "group_logvar": logvar_g,
                "group_logits": self.group_head(feat_g),
                "weights": w}
