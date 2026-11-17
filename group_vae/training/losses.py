"""Group-VAE 损失：组级 DM / CE / KL，单图 VAE 损失"""
import torch
import torch.nn.functional as F


def group_kl(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def group_dm_loss(group_mu, single_mu, labels, mask=None, lambda_var=1.0):
    """group_mu [B,d]；single_mu [B,K,d]（参考特征应已 detach）；mask [B,K]"""
    d = group_mu.size(1)
    if mask is None:
        mask = torch.ones(single_mu.shape[:2], dtype=torch.bool,
                          device=single_mu.device)
    l_mean = group_mu.new_zeros(())
    l_var = group_mu.new_zeros(())
    n_cls = 0
    n_var = 0
    for c in torch.unique(labels).tolist():
        sel = labels == c
        gm = group_mu[sel]
        sm = single_mu[sel][mask[sel]]
        l_mean = l_mean + (gm.mean(0) - sm.mean(0)).pow(2).mean()
        if gm.size(0) > 1 and sm.size(0) > 1:
            l_var = l_var + (gm.var(0, unbiased=False)
                             - sm.var(0, unbiased=False)).pow(2).mean()
            n_var += 1
        n_cls += 1
    n_cls = max(1, n_cls)
    mean_term = l_mean / n_cls
    var_term = l_var / max(1, n_var)
    loss = mean_term + lambda_var * var_term
    return loss, {"dm_mean": float(mean_term.detach()),
                  "dm_var": float(var_term.detach()),
                  "dm_classes": n_cls, "dm_var_classes": n_var}


def variance_ratio(group_mu, single_mu, labels, mask=None):
    d = group_mu.size(1)
    if mask is None:
        mask = torch.ones(single_mu.shape[:2], dtype=torch.bool,
                          device=single_mu.device)
    ratios = []
    for c in torch.unique(labels).tolist():
        sel = labels == c
        if sel.sum() < 2:
            continue
        gv = group_mu[sel].var(0, unbiased=False).mean()
        sm = single_mu[sel][mask[sel]]
        if sm.size(0) < 2:
            continue
        sv = sm.var(0, unbiased=False).mean()
        ratios.append(float(gv / sv.clamp_min(1e-8)))
    return sum(ratios) / max(1, len(ratios))


def single_vae_loss(recon, images, logits, labels, mu, logvar, valid,
                    beta=1e-4, cls_lambda=0.1):
    x = images.reshape(-1, *images.shape[2:])
    m = valid.bool()
    rec = F.mse_loss(recon[m], x[m])
    kl = -0.5 * (1 + logvar[m] - mu[m].pow(2) - logvar[m].exp()).mean()
    cls = F.cross_entropy(logits[m], labels.repeat_interleave(
        images.size(1))[m])
    loss = rec + beta * kl + cls_lambda * cls
    return loss, {"rec": float(rec.detach()), "kl": float(kl.detach()),
                  "cls": float(cls.detach())}
