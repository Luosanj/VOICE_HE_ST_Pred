"""Compare grid fusion and MLP gating. Input: YAML slides with out-of-fold A/R/Y NPZ and gene-list TSV. Output: per-gene and seven-metric TSVs."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
GRID = np.linspace(0, 1, 41).astype(np.float32)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
METHODS = ['A_only', 'R_only', 'pergene_grid', 'mlp_gate']
MLP_PAIRS, MLP_EPOCHS, MLP_SEED = 4_000_000, 4, 0
def pcc_cols(P, Y):
    Pc = P - P.mean(0); Yc = Y - Y.mean(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    return np.where(den > 1e-8, (Pc * Yc).sum(0) / np.where(den > 1e-8, den, 1), 0.0)


def grid_pergene(A, R, Y):
    best = np.full(A.shape[1], -2.0); bw = np.ones(A.shape[1], np.float32)
    for w in GRID:
        r = pcc_cols(w * A + (1 - w) * R, Y); t = r > best; best[t] = r[t]; bw[t] = w
    return bw


class PairMLP(nn.Module):
    def __init__(self, n_genes, gate):
        super().__init__()
        self.gate = gate; self.emb = nn.Embedding(n_genes, 16)
        self.net = nn.Sequential(nn.Linear(4 + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, a, r, dcell, g):
        h = self.net(torch.cat([a[:, None], r[:, None], (a * r)[:, None], dcell[:, None], self.emb(g)], 1))[:, 0]
        return torch.sigmoid(h) * a + (1 - torch.sigmoid(h)) * r if self.gate else h


def zstats(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-6] = 1
    return mu, sd


def fit_mlp(zA, zR, zY, gate, n_pairs=4_000_000, epochs=4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    N, G = zA.shape
    dcell = np.abs(zA - zR).mean(1).astype(np.float32)
    m = PairMLP(G, gate).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    ci = rng.integers(0, N, n_pairs); gi = rng.integers(0, G, n_pairs)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
    a, r, y, dc, gg = t(zA[ci, gi]), t(zR[ci, gi]), t(zY[ci, gi]), t(dcell[ci]), t(gi.astype(np.int64))
    bs = 16384
    for _ in range(epochs):
        perm = torch.randperm(n_pairs, device=DEV)
        for s in range(0, n_pairs, bs):
            b = perm[s:s + bs]
            loss = ((m(a[b], r[b], dc[b], gg[b]) - y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def pred_mlp(m, zA, zR):
    N, G = zA.shape
    dcell = torch.from_numpy(np.abs(zA - zR).mean(1).astype(np.float32)).to(DEV)
    out = np.empty((N, G), np.float32); A = torch.from_numpy(zA).to(DEV); R = torch.from_numpy(zR).to(DEV)
    for g in range(G):
        gi = torch.full((N,), g, dtype=torch.long, device=DEV)
        out[:, g] = m(A[:, g], R[:, g], dcell, gi).cpu().numpy()
    return out


def metrics(pcc, genes, gl):
    """7 metrics; HVG/SVG from the canonical per-slide benchmark list (genes absent from pred -> skipped)."""
    s = pd.Series(pcc, index=genes)
    out = {"ALL": s.mean()}
    for k in (20, 50, 100):
        for tag, col in (("H", "hvg_rank"), ("S", "svg_rank")):
            top = gl.sort_values(col).gene.head(k)
            out[f"{tag}{k}"] = s.reindex(top).dropna().mean()
    return out


def run_inslide(key):
    d = load(key); A, R, Y, band = d["A"], d["R"], d["Y"], d["band"]
    oof = {m: np.zeros_like(Y) for m in METHODS}; log = {}
    for k in np.unique(band):
        te = band == k; tr = ~te
        P = fit_predict_all(A[tr], R[tr], Y[tr], A[te], R[te], log)
        for m in METHODS:
            oof[m][te] = P[m]
        print(f"  {key} band {k} done ({log['fit_seconds']:.0f}s)", flush=True)
    rows, pg = [], {}
    for m in METHODS:

        p = pcc_cols(oof[m], Y); pg[m] = p
        rows.append(dict(protocol="inslide", slide=key, method=m, **metrics(p, d["genes"], d["gl"])))
    pd.DataFrame(pg, index=d["genes"]).to_csv(OUT / f"pergene_inslide_{key}.tsv.gz", sep="\t")
    return rows


def run_transfer(src_keys, tgt_key):
    ds = [load(k) for k in src_keys]; dt = load(tgt_key)
    genes = [g for g in dt["genes"] if all(g in set(d["genes"]) for d in ds)]
    ix = lambda d: [list(d["genes"]).index(g) for g in genes]
    Atr = np.concatenate([d["A"][:, ix(d)] for d in ds]); Rtr = np.concatenate([d["R"][:, ix(d)] for d in ds])
    Ytr = np.concatenate([d["Y"][:, ix(d)] for d in ds]); it = ix(dt)
    log = {}
    P = fit_predict_all(Atr, Rtr, Ytr, dt["A"][:, it], dt["R"][:, it], log)
    rows = []
    for m in METHODS:
        p = pcc_cols(P[m], dt["Y"][:, it])
        rows.append(dict(protocol="transfer", slide=f"{'+'.join(src_keys)}->{tgt_key}", method=m,
                         **metrics(p, np.array(genes), dt["gl"])))
    print(f"  transfer {src_keys}->{tgt_key} done ({log['fit_seconds']:.0f}s, {len(genes)} genes)", flush=True)
    return rows


def load(key):
    entry = SLIDES[key]
    with np.load(entry['predictions'], allow_pickle=True) as z:
        d = dict(A=z['Apred'].astype(np.float32), R=z['Rr'].astype(np.float32),
                 Y=z['Ylog'].astype(np.float32), band=z['band'], genes=z['genes'].astype(str))
    d['gl'] = pd.read_csv(entry['gene_list'], sep='\t')
    if d['A'].shape != d['Y'].shape or d['R'].shape != d['Y'].shape:
        raise ValueError(f'{key}: matrix shape mismatch')
    if not all(np.isfinite(d[k]).all() for k in ['A', 'R', 'Y']):
        raise ValueError(f'{key}: missing branch predictions')
    return d


def fit_predict_all(Atr, Rtr, Ytr, Ate, Rte, log):
    t0 = time.time()
    w = grid_pergene(Atr, Rtr, Ytr)
    muA, sdA = zstats(Atr); muR, sdR = zstats(Rtr); muY, sdY = zstats(Ytr)
    zAtr, zRtr, zYtr = (Atr-muA)/sdA, (Rtr-muR)/sdR, (Ytr-muY)/sdY
    zAte, zRte = (Ate-muA)/sdA, (Rte-muR)/sdR
    m = fit_mlp(zAtr.astype(np.float32), zRtr.astype(np.float32), zYtr.astype(np.float32),
                True, n_pairs=MLP_PAIRS, epochs=MLP_EPOCHS, seed=MLP_SEED)
    log['fit_seconds'] = log.get('fit_seconds', 0) + time.time()-t0
    return dict(A_only=Ate, R_only=Rte, pergene_grid=w*Ate+(1-w)*Rte,
                mlp_gate=pred_mlp(m, zAte.astype(np.float32), zRte.astype(np.float32)))


def main():
    global SLIDES, OUT, DEV, MLP_PAIRS, MLP_EPOCHS, MLP_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', required=True, help='YAML: slides mapping; optional transfers list of {source: [keys], target: key}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default=str(DEV))
    ap.add_argument('--pairs', type=int, default=4_000_000)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.slides).read_text())
    SLIDES = cfg['slides']; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    DEV = torch.device(a.device); MLP_PAIRS, MLP_EPOCHS, MLP_SEED = a.pairs, a.epochs, a.seed
    rows = []
    for key in SLIDES:
        rows += run_inslide(key)
    for transfer in cfg.get('transfers', []):
        rows += run_transfer(transfer['source'], transfer['target'])
    pd.DataFrame(rows).to_csv(OUT/'metrics_all.tsv', sep='\t', index=False)
    (OUT/'run_meta.json').write_text(json.dumps(dict(device=str(DEV), pairs=a.pairs, epochs=a.epochs, seed=a.seed)))


if __name__ == '__main__':
    main()
