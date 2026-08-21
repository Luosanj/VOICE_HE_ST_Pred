"""[VOICE] SE(2)-equivariant decoder / negative-binomial head.

Contributed by Yicheng Tao as part of the VOICE work; vendored here so the repository is
self-contained. Behaviour is unchanged from the original.
"""
"""UNI (frozen) -> SE(2) Transformer -> [optional ref cross-attn] -> linear gene head. MSE loss."""

import math
from typing import Optional
import torch
import torch.nn as nn
import timm


class UNIEncoder(nn.Module):
    """
    UNI ViT-L/16 (frozen) + trainable projection head.

    forward(x):
        x is patches [N, 3, H, W]    -> run UNI -> proj
        x is features [N, 1024]      -> skip UNI, just proj (use precomputed features)
    """

    UNI_DIM = 1024

    def __init__(self, d_model: int, load_uni: bool = True, feat_dim: int = 1024):
        super().__init__()
        if load_uni:
            self.uni = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            for p in self.uni.parameters():
                p.requires_grad = False
        else:
            self.uni = None
        # feat_dim allows concatenated multi-backbone caches (e.g. UNI(+)Phikon = 2048).
        self.proj = nn.Sequential(nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            assert self.uni is not None, "UNI not loaded but raw patches passed"
            with torch.no_grad():
                x = self.uni(x)  # [N, 1024]
        return self.proj(x)  # [N, d_model]


class GaussianModule(nn.Module):
    def __init__(self, d_pair: int, n_gaussians: int = 128, max_dist: float = 500.0):
        super().__init__()
        self.max_dist = max_dist
        self.means = nn.Parameter(torch.linspace(0.0, 1.0, n_gaussians))
        self.log_stds = nn.Parameter(torch.zeros(n_gaussians))
        self.proj = nn.Linear(n_gaussians, d_pair)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        D_norm = (D / self.max_dist).clamp(0.0, 1.0)
        stds = self.log_stds.exp().clamp(min=1e-4)
        basis = torch.exp(-0.5 * ((D_norm.unsqueeze(-1) - self.means) / stds) ** 2)
        return self.proj(basis)


class SE2Layer(nn.Module):
    def __init__(self, d_model: int, d_pair: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pair_update = nn.Linear(d_model, n_heads)
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_pair + n_heads, d_pair), nn.GELU(), nn.Linear(d_pair, d_pair)
        )
        self.pair_to_bias = nn.Linear(d_pair, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_pair = nn.LayerNorm(d_pair)

    def forward(self, feat, Z):
        N = feat.shape[0]
        feat_n = self.norm1(feat)
        pair_update = self.pair_update(feat_n.unsqueeze(1) + feat_n.unsqueeze(0))
        Z = Z + self.pair_mlp(torch.cat([self.norm_pair(Z), pair_update], dim=-1))

        attn_bias = self.pair_to_bias(Z).permute(2, 0, 1)
        Q = self.q_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        K = self.k_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        V = self.v_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_head)
        scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, V).permute(1, 0, 2).reshape(N, -1)
        feat = feat + self.out_proj(out)
        feat = feat + self.ffn(self.norm2(feat))
        return feat, Z


class SE2Transformer(nn.Module):
    def __init__(self, d_model, d_pair, n_heads, n_layers, n_gaussians=128, max_dist=500.0):
        super().__init__()
        self.gaussian = GaussianModule(d_pair, n_gaussians, max_dist)
        self.layers = nn.ModuleList(
            [SE2Layer(d_model, d_pair, n_heads) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, pos):
        # x: [N, d_model], pos: [N, 2] (y, x in pixels)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        D = diff.norm(dim=-1)
        Z = self.gaussian(D)
        feat = x
        for layer in self.layers:
            feat, Z = layer(feat, Z)
        return self.norm_out(feat)


class SimpleHE2Cell(nn.Module):
    """
    Default pipeline:  UNI(frozen) -> SE(2) Transformer -> linear gene head.

    Optional reference cross-attention: pass `ref_module` to insert a layer
    between SE(2) output and the gene head. The module is expected to accept
    `(h, training)` and return either `h_refined` or `(h_refined, aux)`.
    Leave as None for the minimal UNI + SE(2) + MSE baseline.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        d_pair: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        n_gaussians: int = 128,
        max_dist: float = 500.0,
        ref_module: Optional[nn.Module] = None,
        load_uni: bool = True,
    ):
        super().__init__()
        self.encoder = UNIEncoder(d_model, load_uni=load_uni)
        self.se2 = SE2Transformer(d_model, d_pair, n_heads, n_layers, n_gaussians, max_dist)
        self.ref_module = ref_module
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_genes),
        )

    def forward(self, patches_or_features, pos):
        # patches: [N, 3, H, W] OR pre-cached UNI features [N, 1024]
        x = self.encoder(patches_or_features)
        h = self.se2(x, pos)
        if self.ref_module is not None:
            out = self.ref_module(h, training=self.training)
            h = out[0] if isinstance(out, tuple) else out
        return self.head(h)
