"""Run in-slide component rows 2–9. Input: matched cell features and projection/decoder weights. Output: five-fold predictions, PCC tables, and fitted decoders."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from experiments.ablation.alignment_projection import Tower
from voice.release import read_weights
from voice.scale_train import ScaleHE2Cell, nb_nll, per_gene_pcc, morans_i

METHODS = {
    2: 'ridge_retrieval', 3: 'alignment_projection_128', 4: 'frozen_retrieval_1536',
    5: 'stage1_retrieval', 6: 'stage2_retrieval', 7: 'frozen_direct',
    8: 'stage2_direct', 9: 'direct_retrieval_fusion',
}
MET = ['ALL', 'HVG20', 'HVG50', 'HVG100', 'SVG20', 'SVG50', 'SVG100']


def gene_orders(path, genes):
    table = pd.read_csv(path, sep='\t')
    indices = {g: i for i, g in enumerate(genes)}
    hvg = np.array([indices[g] for g in table.sort_values('hvg_rank').gene if g in indices], np.int64)
    svg = np.array([indices[g] for g in table.sort_values('svg_rank').gene if g in indices], np.int64)
    return hvg, svg


def bands5(pos):
    x = pos[:, 1]
    q = np.quantile(x, np.linspace(0, 1, 6))
    q[-1] += 1
    return np.clip(np.digitize(x, q[1:-1]), 0, 4)


def spatial_patches(pos, seed=0):
    rng = np.random.RandomState(seed)
    cy, cx = pos[:, 0], pos[:, 1]
    patches = []
    for y0 in np.arange(0, cy.max() + 256, 226):
        rm = (cy >= y0) & (cy < y0 + 256)
        if not rm.any():
            continue
        idxr, cxr = np.nonzero(rm)[0], cx[rm]
        for x0 in np.arange(0, cx.max() + 256, 226):
            cm = (cxr >= x0) & (cxr < x0 + 256)
            if cm.sum() < 2:
                continue
            cells = idxr[cm]
            if len(cells) > 200:
                cells = rng.choice(cells, 200, replace=False)
            patches.append(np.sort(cells))
    if not patches:
        raise ValueError('No spatial patches with at least two cells')
    return patches


@torch.no_grad()
def knn_R(q, b, bank_Y, K=200, tau=0.03, device='cuda', chunk=512):
    if len(b) == 0 or K < 1 or tau <= 0:
        raise ValueError('Require a nonempty reference band, K>=1 and tau>0')
    dev = torch.device(device)
    K = min(K, len(b))
    q = F.normalize(torch.from_numpy(np.ascontiguousarray(q, np.float32)), dim=1)
    b = F.normalize(torch.from_numpy(np.ascontiguousarray(b, np.float32)), dim=1)
    R = np.empty((len(q), bank_Y.shape[1]), np.float32)
    try:
        import faiss
    except ImportError:
        faiss = None
    index = None
    if faiss is not None:
        index = faiss.IndexFlatIP(b.shape[1])
        if dev.type == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, dev.index or 0, index)
        index.add(b.numpy())
    else:
        b = b.to(dev)
    for i in range(0, len(q), chunk):
        if index is not None:
            D, I = index.search(q[i:i + chunk].numpy(), K)
        else:
            qc = q[i:i + chunk].to(dev)
            ds, ids = [], []
            for s in range(0, len(b), 65536):
                sim = qc @ b[s:s + 65536].T
                d, ix = sim.topk(min(K, sim.shape[1]), dim=1)
                ds.append(d)
                ids.append(ix + s)
            dcat, icat = torch.cat(ds, 1), torch.cat(ids, 1)
            d, sel = dcat.topk(K, dim=1)
            D = d.cpu().numpy()
            I = torch.gather(icat, 1, sel).cpu().numpy()
        w = np.exp((D - D.max(1, keepdims=True)) / tau)
        w /= w.sum(1, keepdims=True)
        R[i:i + chunk] = np.einsum('nk,nkg->ng', w, bank_Y[I])
    return R


def ridge_R(Xb, Yb, Xh, device):
    dev = torch.device(device)
    X = torch.from_numpy(np.ascontiguousarray(Xb, np.float32)).to(dev)
    Y = torch.from_numpy(np.ascontiguousarray(Yb, np.float32)).to(dev)
    W = torch.linalg.solve(X.T @ X + 100.0 * torch.eye(X.shape[1], device=dev), X.T @ Y)
    zb = (X @ W).cpu().numpy()
    zh = (torch.from_numpy(np.ascontiguousarray(Xh, np.float32)).to(dev) @ W).cpu().numpy()
    return knn_R(zh, zb, Yb, K=20, tau=1.0, device=device)


@torch.no_grad()
def project_features(E, checkpoint, device):
    w = read_weights(checkpoint)
    if (int(w['in_dim']), int(w['hidden']), int(w['out_dim'])) != (1536, 512, 128):
        raise ValueError('Row 3 requires a 1536→512→128 frozen-feature tower')
    model = Tower(1536, 512, 128).to(device).eval()
    model.load_state_dict(w['state_dict'])
    mu, sd = torch.as_tensor(w['mu'], device=device), torch.as_tensor(w['sd'], device=device)
    if not bool(torch.isfinite(sd).all()) or bool((sd <= 0).any()):
        raise ValueError('Projection standard deviations must be finite and positive')
    return np.concatenate([model((torch.from_numpy(E[i:i + 8192]).to(device) - mu) / sd).cpu().numpy()
                           for i in range(0, len(E), 8192)])


def best_w(A, R, Y):
    best = np.full(A.shape[1], -2.0)
    beta = np.ones(A.shape[1], np.float32)
    for w in np.linspace(0, 1, 41):
        pcc = np.nan_to_num(per_gene_pcc(w * A + (1 - w) * R, Y))
        take = pcc > best
        best = np.where(take, pcc, best)
        beta = np.where(take, w, beta).astype(np.float32)
    return beta


def stack_cf(A, R, Y, seed=0):
    idx = np.random.default_rng(seed).permutation(len(A))
    f1, f2 = idx[:len(A) // 2], idx[len(A) // 2:]
    w1, w2 = best_w(A[f1], R[f1], Y[f1]), best_w(A[f2], R[f2], Y[f2])
    S = np.empty_like(A)
    S[f2] = w1 * A[f2] + (1 - w1) * R[f2]
    S[f1] = w2 * A[f1] + (1 - w2) * R[f1]
    half = np.empty(len(A), np.int8)
    half[f1], half[f2] = 0, 1
    return S, np.stack([w2, w1]), half


@torch.no_grad()
def predict_patches(model, ft, pt, patches, selected, panel, n_cells):
    model.eval()
    S = torch.zeros((n_cells, len(panel)), device=ft.device)
    C = torch.zeros(n_cells, device=ft.device)
    for j in selected:
        ci = patches[j]
        lm, _ = model(ft[ci], pt[ci])
        S[ci] += lm[:, panel].float()
        C[ci] += 1
    return (S / C.clamp_min(1)[:, None]).cpu().numpy()


def direct_cv(E, data, checkpoint, args, out):
    dev = torch.device(args.device)
    w = read_weights(checkpoint)
    state = w['se2'] if 'se2' in w else w['model']
    cfg = w if 'd_model' in w else w.get('args', {})
    dm = int(cfg.get('d_model', state['head.mu_lin.weight'].shape[1]))
    nl = int(cfg.get('n_layers', len({k.split('.')[2] for k in state if k.startswith('se2.layers.')})))
    n_global = state['head.mu_lin.weight'].shape[0]
    panel, pos, Ylog = data['panel'], data['pos'], data['Ylog']
    if panel.min() < 0 or panel.max() >= n_global:
        raise ValueError('Feature panel indices must match the decoder checkpoint gene order')
    patches = spatial_patches(pos)
    ft, pt = torch.from_numpy(E).to(dev), torch.from_numpy(pos).to(dev)
    Yraw = data.get('Yraw', np.expm1(Ylog))
    Yt, panel_t = torch.from_numpy(Yraw).to(dev), torch.from_numpy(panel).to(dev)
    patch_ci = [torch.from_numpy(c).to(dev) for c in patches]
    band = bands5(pos)
    xs = pos[:, 1]
    q = np.quantile(xs, np.linspace(0, 1, 6))
    q[-1] += 1
    span = xs.max() - xs.min()
    pred = np.zeros_like(Ylog)
    log = []
    torch.manual_seed(0)
    for fold in range(5):
        test = band == fold
        lo, hi = q[fold + 1], q[fold + 1] + args.inner_frac * span
        inner = (((xs >= lo) & (xs < hi)) if hi <= xs.max()
                 else ((xs >= lo) | (xs < xs.min() + args.inner_frac * span))) & ~test
        train = ~test & ~inner
        if not test.any() or inner.sum() < 2 or train.sum() < 2:
            raise ValueError(f'Fold {fold} needs test cells and at least two training/validation cells')
        train_t = torch.from_numpy(train).to(dev)
        tr = [j for j, c in enumerate(patches) if train[c].any()]
        va = [j for j, c in enumerate(patches) if (test | inner)[c].any()]
        model = ScaleHE2Cell(n_global, feat_dim=1536, d_model=dm, n_layers=nl).to(dev)
        model.load_state_dict(state)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best, no_imp, best_pred = -np.inf, 0, None
        for ep in range(1, args.epochs + 1):
            model.train()
            for j in np.random.RandomState(1000 * fold + ep).permutation(len(tr)):
                ci = patch_ci[tr[j]]
                mask = train_t[ci]
                lm, aux = model(ft[ci], pt[ci])
                loss = F.mse_loss(lm[:, panel_t][mask], torch.log1p(Yt[ci][mask]))
                if args.nb_weight > 0:
                    loss += args.nb_weight * nb_nll(Yt[ci][mask], aux['mu'][:, panel_t][mask],
                                                   aux['log_theta'][panel_t].exp())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            P = predict_patches(model, ft, pt, patch_ci, va, panel_t, len(E))
            score = float(np.nanmean(per_gene_pcc(P[inner], Ylog[inner])))
            if not np.isfinite(score):
                raise ValueError(f'Fold {fold}: validation PCC is undefined')
            improved = score > best
            if improved:
                best, no_imp, best_pred = score, 0, P.copy()
                torch.save(dict(se2={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                d_model=dm, n_layers=nl, fold=fold, epoch=ep, validation_pcc=score,
                                panel=panel, nb_weight=args.nb_weight), out / f'fold{fold}.pt')
            else:
                no_imp += 1
            log.append(dict(fold=fold, epoch=ep, validation_pcc=score, selected=improved))
            print(f'{out.name}: fold={fold} epoch={ep} validation={score:.4f}', flush=True)
            if no_imp >= args.patience:
                break
        pred[test] = best_pred[test]
    pd.DataFrame(log).to_csv(out / 'training.tsv', sep='\t', index=False)
    return pred


def load_features(slide, variants):
    data, features = None, {}
    for variant in sorted(variants):
        with np.load(slide['features'][variant], allow_pickle=True) as z:
            item = {k: z[k] for k in ('Ylog', 'pos', 'panel', 'genes')}
            for key in ('expr_rows', 'Yraw'):
                if key in z:
                    item[key] = z[key]
            item['Ylog'] = np.asarray(item['Ylog'], np.float32)
            item['pos'] = np.asarray(item['pos'], np.float32)
            item['panel'] = np.asarray(item['panel'], np.int64)
            item['genes'] = item['genes'].astype(str)
            E = np.asarray(z['E1536'], np.float32)
        n, g = item['Ylog'].shape
        if (E.shape != (n, 1536) or item['pos'].shape != (n, 2)
                or item['panel'].shape != (g,) or item['genes'].shape != (g,) or n < 10 or g == 0):
            raise ValueError(f'{variant}: inconsistent feature, cell or gene dimensions')
        if (not np.isfinite(E).all() or not np.isfinite(item['Ylog']).all()
                or not np.isfinite(item['pos']).all() or (item['Ylog'] < 0).any()):
            raise ValueError(f'{variant}: features, counts and positions must be finite; Ylog must be nonnegative')
        if len(np.unique(item['genes'])) != g or len(np.unique(item['panel'])) != g or (item['panel'] < 0).any():
            raise ValueError('Genes and panel indices must be unique, with nonnegative panel indices')
        if 'expr_rows' in item and (item['expr_rows'].shape != (n,) or len(np.unique(item['expr_rows'])) != n):
            raise ValueError('expr_rows must identify every feature cell once')
        if 'Yraw' in item:
            raw = item['Yraw'] = np.asarray(item['Yraw'], np.float32)
            if raw.shape != (n, g) or (raw < 0).any() or not np.isfinite(raw).all() or not np.allclose(np.log1p(raw), item['Ylog'], atol=1e-5):
                raise ValueError('Yraw must be raw counts matching Ylog')
        if data is not None:
            for key in ('Ylog', 'pos', 'panel', 'genes', 'expr_rows'):
                if (key in data) != (key in item) or (key in data and not np.array_equal(data[key], item[key])):
                    raise ValueError(f'{variant}: {key} differs between encoder variants')
        else:
            data = item
        features[variant] = E
    if set(np.unique(bands5(data['pos']))) != set(range(5)):
        raise ValueError('Coordinates must yield five nonempty vertical bands')
    return data, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML: projection, heads, and slides with matched feature paths')
    ap.add_argument('--protocol', choices=['7m', '23m'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', default='2,3,4,5,6,7,8,9')
    ap.add_argument('--gene_lists', help='Optional directory of canonical slide-name.tsv gene rankings')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--inner_frac', type=float, default=0.1)
    ap.add_argument('--nb_weight', type=float, default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    try:
        requested = set(int(r) for r in args.rows.split(','))
    except ValueError:
        ap.error('--rows must be comma-separated row numbers')
    if not requested or not requested <= set(METHODS):
        ap.error('--rows must select rows 2–9')
    if args.nb_weight is None:
        args.nb_weight = 0.0 if args.protocol == '7m' else 0.5
    if args.epochs < 1 or args.patience < 1 or args.lr <= 0 or not 0 < args.inner_frac < 1 or args.nb_weight < 0:
        ap.error('Require positive epochs, patience and lr; 0<inner_frac<1; nb_weight>=0')
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = yaml.safe_load(Path(args.config).read_text())
    needed = requested | ({6, 8} if 9 in requested else set())
    variants = set()
    for r in needed:
        variants.add('frozen' if r in (2, 3, 4, 7) else ('stage1' if r == 5 else 'stage2'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    names = [sl['name'] for sl in spec['slides']]
    if not names or len(set(names)) != len(names) or any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Supply unique slide names without path separators')
    for slide in spec['slides']:
        data, E = load_features(slide, variants)
        Y, pos, genes = data['Ylog'], data['pos'], data['genes']
        band = bands5(pos)
        if args.gene_lists:
            hvg, svg = gene_orders(Path(args.gene_lists) / f"{slide['name']}.tsv", genes)
            if len(hvg) != len(genes) or len(svg) != len(genes) or len(np.unique(hvg)) != len(genes) or len(np.unique(svg)) != len(genes):
                raise ValueError('Canonical rankings must cover every evaluated gene once')
        else:
            hvg = np.argsort(-Y.var(0))
            svg = np.argsort(-np.nan_to_num(morans_i(pos, Y)))
        preds, extra = {}, {}
        for r in sorted(needed - {9}):
            row_out = out / slide['name'] / f'row{r}'
            row_out.mkdir(parents=True, exist_ok=True)
            if r in (7, 8):
                variant = 'frozen' if r == 7 else 'stage2'
                preds[r] = direct_cv(E[variant], data, spec['heads'][variant], args, row_out)
            else:
                X = E['frozen' if r in (2, 3, 4) else ('stage1' if r == 5 else 'stage2')]
                if r == 3:
                    X = project_features(X, spec['projection'], args.device)
                P = np.zeros_like(Y)
                for f in range(5):
                    te, tr = band == f, band != f
                    P[te] = (ridge_R(X[tr], Y[tr], X[te], args.device) if r == 2
                             else knn_R(X[te], X[tr], Y[tr], K=20 if r == 3 else 200,
                                        tau=1.0 if r == 3 else 0.03, device=args.device))
                preds[r] = P
            print(f"{slide['name']}: row {r} complete", flush=True)
        if 9 in requested:
            preds[9], beta, half = stack_cf(preds[8], preds[6], Y)
            extra[9] = dict(Apred=preds[8], Rr=preds[6], Spred=preds[9], beta=beta, gate_half=half)
        for r in sorted(needed):
            pc = per_gene_pcc(preds[r], Y)
            if r == 9:
                pc = np.nan_to_num(pc)
            scores = {'ALL': float(np.nanmean(pc))}
            for k in (20, 50, 100):
                scores[f'HVG{k}'] = float(np.nanmean(pc[hvg[:k]]))
                scores[f'SVG{k}'] = float(np.nanmean(pc[svg[:k]]))
            results.append(dict(slide=slide['name'], row=r, method=METHODS[r], n_cells=len(Y), n_genes=len(genes), **scores))
            row_out = out / slide['name'] / f'row{r}'
            row_out.mkdir(parents=True, exist_ok=True)
            saved = {k: v for k, v in data.items() if k != 'Yraw'}
            np.savez_compressed(row_out / 'predictions.npz', **saved, pred=preds[r], band=band, **extra.get(r, {}))
            pd.DataFrame(dict(gene=genes, pcc=pc)).to_csv(row_out / 'pergene.tsv', sep='\t', index=False)
        del preds, E
    df = pd.DataFrame(results)
    df.to_csv(out / 'results.tsv', sep='\t', index=False)
    df.groupby(['row', 'method'])[MET].mean().reset_index().to_csv(out / 'macro.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
