"""The scoring the paper reports: per-gene Pearson across cells, and the HVG / SVG gene sets.

Correlation is computed for each gene ACROSS CELLS, then averaged over a gene set -- never per cell. A gene the
model cannot emit is scored 0 and still counts in the denominator (see `score_against_head`), because dropping
it would silently reward a model for having a smaller output space.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-8


def per_gene_pcc(P, Y, chunk: int = 512):
    """P, Y: [N, G] log1p. Returns [G]. Chunked so no [N, G] temporary is allocated for a 700k-cell slide."""
    G = P.shape[1]
    out = np.empty(G, np.float32)
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        a = np.asarray(P[:, s:e], np.float32).copy()
        b = np.asarray(Y[:, s:e], np.float32).copy()
        a -= a.mean(0, keepdims=True); b -= b.mean(0, keepdims=True)
        num = (a * b).sum(0); den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        out[s:e] = np.where(den > EPS, num / np.where(den > EPS, den, 1.0), 0.0)
    return np.nan_to_num(out)


def morans_i(coords, Y, k: int = 6, chunk: int = 512):
    """Spatial autocorrelation per gene on a symmetric k-NN graph with row-normalised weights."""
    from sklearn.neighbors import NearestNeighbors
    from scipy import sparse
    n = coords.shape[0]; k = min(k, n - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    W = sparse.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    rs = np.asarray(W.sum(1)).ravel(); rs[rs == 0] = 1
    Wn = sparse.diags(1 / rs) @ W
    out = np.empty(Y.shape[1], np.float64)
    for s in range(0, Y.shape[1], chunk):
        e = min(s + chunk, Y.shape[1])
        Z = np.asarray(Y[:, s:e], np.float64)
        Z = Z - Z.mean(0, keepdims=True)
        out[s:e] = np.asarray((Z * (Wn @ Z)).sum(0)) / np.clip((Z * Z).sum(0), 1e-9, None)
    return np.nan_to_num(out)


def rank_hvg_svg(Y, pos, k: int = 6):
    """Return (hvg_rank, svg_rank): rank 0 = most variable / most spatially structured."""
    var = np.asarray(Y).var(0)
    mor = morans_i(pos, Y, k)
    hvg = np.empty(len(var), int); hvg[np.argsort(-var)] = np.arange(len(var))
    svg = np.empty(len(mor), int); svg[np.argsort(-mor)] = np.arange(len(mor))
    return var, mor, hvg, svg


def summarize(pcc, hvg_order, svg_order, ks=(20, 50, 100)):
    """{'ALL': ..., 'HVG20': ..., 'SVG20': ..., ...} from a per-gene PCC vector and two orderings."""
    r = {"ALL": float(np.mean(pcc))}
    for k in ks:
        r[f"HVG{k}"] = float(np.mean(pcc[hvg_order[:k]])) if len(hvg_order) >= k else float("nan")
        r[f"SVG{k}"] = float(np.mean(pcc[svg_order[:k]])) if len(svg_order) >= k else float("nan")
    return r


def score_against_head(pred_head, Y, gid, pos, hvg_order=None, svg_order=None, ks=(20, 50, 100)):
    """Score a head-space prediction on a slide's own panel.

    pred_head : [N, n_head]  the model's output
    Y         : [N, G_panel] measured log1p, real genes only
    gid       : [G_panel]    global index of each panel gene, -1 if the head cannot emit it

    Genes with gid < 0 get a constant-zero prediction, hence PCC 0, and still count in the denominator.
    """
    P = np.zeros_like(Y, dtype=np.float32)
    cov = gid >= 0
    P[:, cov] = pred_head[:, gid[cov]]
    pcc = per_gene_pcc(P, Y)
    if hvg_order is None or svg_order is None:
        _, _, hr, sr = rank_hvg_svg(Y, pos)
        hvg_order = np.argsort(hr); svg_order = np.argsort(sr)
    out = summarize(pcc, hvg_order, svg_order, ks)
    out.update(n_genes=int(Y.shape[1]), n_covered=int(cov.sum()))
    return out, pcc
