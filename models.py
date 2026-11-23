"""EdgeVAE 潜特征主干（冻结）"""

import torch
import torch.nn as nn


class VAEBackbone(nn.Module):
    """冻结的 EdgeVAE 编码器, 输出确定性 mu (16×8×8)"""

    def __init__(self, num_classes, device, ckpt_path):
        super().__init__()
        from vae import EdgeVAE
        self.ae = EdgeVAE(latent_dim=16, num_classes=num_classes).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.ae.load_state_dict(ck["model_state_dict"])
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False
        self.n_params = sum(p.numel() for p in self.ae.parameters())
        self.channels = (16,)

    def forward_features(self, x, t=None, layer=None):
        return self.ae.encode_mu(x)
