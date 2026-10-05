"""Predict gene expression. Input: cell representations. Output: log1p count means and negative-binomial parameters."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from voice._se2_arch import UNIEncoder, SE2Transformer


class MoEHead(nn.Module):
    """Gated mixture of linear gene heads over a shared trunk."""

    def __init__(self, d_model: int, n_genes: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_genes = n_genes
        self.trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.gate = nn.Linear(d_model, n_experts)

        self.expert_w = nn.Parameter(torch.empty(n_experts, d_model, n_genes))
        self.expert_b = nn.Parameter(torch.zeros(n_experts, n_genes))
        nn.init.xavier_uniform_(self.expert_w)

    def init_expert_bias(self, avg_log: torch.Tensor):
        """avg_log: [n_experts, n_genes] cell-type mean log1p expr -> expert bias prior."""
        with torch.no_grad():
            if avg_log.shape == self.expert_b.shape:
                self.expert_b.copy_(avg_log)

    def forward(self, h: torch.Tensor):
        t = self.trunk(h)
        gate_logits = self.gate(t)
        g = torch.softmax(gate_logits, dim=-1)

        eo = torch.einsum("nd,edg->neg", t, self.expert_w) + self.expert_b
        pred = (g.unsqueeze(-1) * eo).sum(1)
        return pred, {"gate_logits": gate_logits}


class LinearHead(nn.Module):
    def __init__(self, d_model: int, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_genes),
        )

    def forward(self, h):
        return self.net(h), {}


class NBHead(nn.Module):

    def __init__(self, d_model: int, n_genes: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.mu_lin = nn.Linear(d_model, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def forward(self, h):
        mu = F.softplus(self.mu_lin(self.trunk(h))) + 1e-4
        return torch.log1p(mu), {"mu": mu, "log_theta": self.log_theta}


class RAGFusion(nn.Module):
    """Learned gene-space gate that blends base prediction with retrieved expr."""

    def __init__(self, d_model: int, n_genes: int, per_gene_gate: bool = False):
        super().__init__()
        out = n_genes if per_gene_gate else 1
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, d_model), nn.GELU(), nn.Linear(d_model, out)
        )


        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 0.0)

    def forward(self, base_pred, h, retrieved, sim_stats):
        beta = torch.sigmoid(self.gate(torch.cat([h, sim_stats], dim=-1)))
        return base_pred + beta * (retrieved - base_pred), beta


class HE2CellPlus(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        d_pair: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        n_gaussians: int = 128,
        max_dist: float = 500.0,
        load_uni: bool = False,
        use_moe: bool = False,
        n_experts: int = 8,
        use_rag: bool = False,
        rag_k: int = 16,
        rag_temp: float = 0.03,
        per_gene_gate: bool = False,
        rag_mode: str = "gate",
        head_type: str = "linear",
        feat_dim: int = 1024,
    ):
        super().__init__()
        self.encoder = UNIEncoder(d_model, load_uni=load_uni, feat_dim=feat_dim)
        self.se2 = SE2Transformer(d_model, d_pair, n_heads, n_layers, n_gaussians, max_dist)
        self.use_moe = use_moe
        self.use_rag = use_rag
        self.rag_k = rag_k
        self.rag_temp = rag_temp
        self.rag_mode = rag_mode
        if use_moe:
            self.head = MoEHead(d_model, n_genes, n_experts)
        elif head_type == "nb":
            self.head = NBHead(d_model, n_genes)
        else:
            self.head = LinearHead(d_model, n_genes)
        self.rag = RAGFusion(d_model, n_genes, per_gene_gate) if (use_rag and rag_mode == "gate") else None
        if use_rag and rag_mode == "input":


            self.retr_in = nn.Linear(n_genes, d_model)
            nn.init.zeros_(self.retr_in.weight)
            nn.init.zeros_(self.retr_in.bias)


        self.register_buffer("bank_feats", torch.zeros(0), persistent=False)
        self.register_buffer("bank_expr", torch.zeros(0), persistent=False)
        self._id2pos = None

    @torch.no_grad()
    def set_bank(self, feats: torch.Tensor, expr_log: torch.Tensor, ids):
        """feats: [M,1024] raw UNI; expr_log: [M,G] log1p; ids: list/array of cell ids."""
        dev = next(self.parameters()).device
        self.bank_feats = F.normalize(feats.to(dev).float(), dim=1)
        self.bank_expr = expr_log.to(dev).float()
        self._id2pos = {int(c): i for i, c in enumerate(ids)}

    @torch.no_grad()
    def _retrieve(self, uni_feats: torch.Tensor, query_ids=None):
        q = F.normalize(uni_feats.float(), dim=1)
        sim = q @ self.bank_feats.T
        if self.training and query_ids is not None and self._id2pos is not None:

            rows, cols = [], []
            for r, cid in enumerate(query_ids):
                p = self._id2pos.get(int(cid), -1)
                if p >= 0:
                    rows.append(r); cols.append(p)
            if rows:
                sim[torch.tensor(rows, device=sim.device), torch.tensor(cols, device=sim.device)] = -1e4
        topv, topi = sim.topk(self.rag_k, dim=1)
        neigh = self.bank_expr[topi]
        w = torch.softmax(topv / self.rag_temp, dim=1).unsqueeze(-1)
        retrieved = (w * neigh).sum(1)
        sim_stats = torch.stack([topv.mean(1), topv[:, 0]], dim=1)
        return retrieved, sim_stats

    def forward(self, features, pos, query_ids=None):
        x = self.encoder(features)
        h = self.se2(x, pos)
        base_pred, head_aux = self.head(h)
        gate_logits = head_aux.get("gate_logits")
        beta = None
        if self.use_rag and self.bank_feats.numel() > 0:
            assert features.ndim == 2, "RAG needs cached UNI features as input"
            retrieved, sim_stats = self._retrieve(features, query_ids)
            if self.rag_mode == "input":
                h2 = h + self.retr_in(retrieved.detach())
                pred, _ = self.head(h2)
            else:
                pred, beta = self.rag(base_pred, h, retrieved, sim_stats)
        else:
            pred = base_pred
        return {"pred": pred, "base_pred": base_pred, "gate_logits": gate_logits, "beta": beta,
                "mu": head_aux.get("mu"), "log_theta": head_aux.get("log_theta")}
