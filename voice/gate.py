"""Fit and apply per-gene fusion weights. Input: direct/retrieval predictions and reference expression. Output: beta weights and fused predictions."""
from __future__ import annotations
import numpy as np

GRID = np.linspace(0, 1, 41).astype(np.float32)
EPS = 1e-8


def _centred(X):
    X = np.asarray(X, np.float32)
    return X - X.mean(0, keepdims=True)


def per_gene_pcc_blend(Ac, Rc, Yc, Yn, beta):
    """Pearson per gene of `beta*A + (1-beta)*R`, with A, R, Y already centred and |Y| precomputed."""
    b = beta if np.isscalar(beta) else np.asarray(beta, np.float32)[None, :]
    P = b * Ac + (1.0 - b) * Rc
    return (P * Yc).sum(0) / (np.sqrt((P ** 2).sum(0)) * Yn + EPS)


def fit_beta(A, R, Y, covered=None, grid=GRID):
    A = np.asarray(A, np.float32); R = np.asarray(R, np.float32); Y = np.asarray(Y, np.float32)
    if covered is None:
        covered = ~np.isnan(R).any(0)
    covered = np.asarray(covered, bool)
    G = A.shape[1]
    Rz = np.nan_to_num(R)
    Ac, Rc, Yc = _centred(A), _centred(Rz), _centred(Y)
    Yn = np.sqrt((Yc ** 2).sum(0))

    best = np.full(G, -np.inf, np.float32)
    beta = np.ones(G, np.float32)
    for b in grid:
        p = per_gene_pcc_blend(Ac, Rc, Yc, Yn, float(b))
        m = p > best
        best[m] = p[m]; beta[m] = b
    beta[~covered] = 1.0
    best[~covered] = per_gene_pcc_blend(Ac, Rc, Yc, Yn, 1.0)[~covered]
    return beta, np.nan_to_num(best).astype(np.float32)


def fit_beta_reference(references, grid=GRID):
    acc, cnt = {}, {}
    for A, R, Y, panel in references:
        beta, _p = fit_beta(A, R, Y)
        for g, b in zip(np.asarray(panel, np.int64), beta):
            if not np.isfinite(b):
                continue
            g = int(g)
            acc[g] = acc.get(g, 0.0) + float(b)
            cnt[g] = cnt.get(g, 0) + 1
    return {g: acc[g] / cnt[g] for g in acc}


def apply_gate(A, R, panel, beta_by_gene, default=1.0):
    A = np.asarray(A, np.float32); R = np.asarray(R, np.float32)
    panel = np.asarray(panel, np.int64)
    covered = ~np.isnan(R).any(0)
    beta = np.array([beta_by_gene.get(int(g), default) for g in panel], np.float32)


    beta[~np.isfinite(beta)] = 1.0
    beta[~covered] = 1.0
    Rz = np.nan_to_num(R)
    return (beta[None, :] * A + (1.0 - beta[None, :]) * Rz).astype(np.float32), beta


def oracle_beta(A, R, Y, covered=None):
    """Fit beta on target expression for an upper-bound evaluation."""
    return fit_beta(A, R, Y, covered)
