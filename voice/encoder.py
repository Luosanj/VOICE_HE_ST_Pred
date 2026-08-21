"""The LoRA-adapted UNI2-h encoder and the cell-mask pooling that turns it into one vector per cell.

Everything needed to run a trained VOICE model forward lives here, and nothing else does: no dataset, no
corpus index, no scFoundation. `predict/` imports only this module, so inference has no training-data
dependencies at all.

The LoRA weights are stored as a flat name -> tensor dict keyed by `model.named_parameters()`, so loading is
`msd[name].data.copy_(v)` -- Stage-1 first (it covers every adapted block), then the Stage-2 overlay on top of
the blocks Stage 2 trained. Both are needed: Stage 2 only ships the tensors it updated.
"""
from __future__ import annotations

# HF_HOME must be in the environment BEFORE huggingface_hub is imported -- it freezes its cache paths at import
# time, so setting the variable afterwards is silently ignored and UNI2-h is fetched from the network instead of
# the local cache, which then fails on the model's access gate. timm pulls in huggingface_hub, hence the order.
from voice import paths as _paths
try:
    _paths.hf_home()
except RuntimeError:
    pass                       # HF_HOME already set in the environment, or configured some other way

import torch
import torch.nn as nn
import timm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PREFIX = 9          # UNI2-h prepends 1 CLS + 8 register tokens before the 256 patch tokens


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
    """UNI2-h (ViT-H/14, 1536-d) frozen. GATED on HuggingFace -- see configs/default.yaml for the download.

    """
    m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24,
                          num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                          no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                          reg_tokens=8, dynamic_img_size=True)
    for p in m.parameters():
        p.requires_grad = False
    return m.to(dev).eval()


def inject_lora(m, nblocks, r, alpha, dropout=0.0):
    """Adapt the attention qkv and proj of the LAST `nblocks` transformer blocks."""
    n = len(m.blocks)
    for i in range(n - nblocks, n):
        m.blocks[i].attn.qkv = LoRALinear(m.blocks[i].attn.qkv, r, alpha, dropout)
        m.blocks[i].attn.proj = LoRALinear(m.blocks[i].attn.proj, r, alpha, dropout)


def pooled_feat(model, x, w16):
    """x: [B,3,224,224] normalised; w16: [B,16,16] cell-mask weights summing to 1. Returns [B,1536].

    The mask is what makes the feature belong to ONE cell rather than to the crop: patch tokens are averaged
    with the cell's own area as the weight, so a neighbouring cell inside the same crop contributes nothing.
    """
    t = model.forward_features(x)                     # [B, PREFIX + 256, 1536]
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
