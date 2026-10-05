"""Encode cell crops with LoRA-adapted UNI2-h. Input: normalized RGB crops and 16x16 masks. Output: 1536-dimensional cell features."""
from __future__ import annotations


from voice import paths as _paths
try:
    _paths.hf_home()
except RuntimeError:
    pass

import torch
import torch.nn as nn
import timm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PREFIX = 9


class LoRALinear(nn.Module):
    """B is zero-initialised, so a freshly injected adapter is exactly the frozen model."""
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__(); self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.s = alpha / r
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.s


def build_uni2(dev):
    """Build frozen UNI2-h with 1536-dimensional features."""
    m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24,
                          num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                          no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                          reg_tokens=8, dynamic_img_size=True)
    for p in m.parameters():
        p.requires_grad = False
    return m.to(dev).eval()


def inject_lora(m, nblocks, r, alpha, dropout=0.0):
    """Adapt attention qkv and proj in the last nblocks transformer blocks."""
    n = len(m.blocks)
    for i in range(n - nblocks, n):
        m.blocks[i].attn.qkv = LoRALinear(m.blocks[i].attn.qkv, r, alpha, dropout)
        m.blocks[i].attn.proj = LoRALinear(m.blocks[i].attn.proj, r, alpha, dropout)


def pooled_feat(model, x, w16):
    t = model.forward_features(x)
    patch = t[:, PREFIX:]
    side = int(round(patch.shape[1] ** 0.5))
    grid = patch.reshape(patch.shape[0], side, side, -1)
    return (grid * w16.unsqueeze(-1)).sum((1, 2))


def load_lora(model, stage1_lora: dict, stage2_lora: dict | None = None):
    """Stage-1 covers every adapted block; Stage-2 overlays the blocks it actually trained."""
    msd = dict(model.named_parameters())
    for n, v in stage1_lora.items():
        msd[n].data.copy_(v.to(msd[n].device))
    if stage2_lora:
        for n, v in stage2_lora.items():
            msd[n].data.copy_(v.to(msd[n].device))
    return model
