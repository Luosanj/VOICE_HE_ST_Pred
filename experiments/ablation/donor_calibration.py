"""Donor-held-out calibration of the fusion gate. Input: donor split, reference A/R/Y caches and target predictions for a
Stage-2 model trained on all slides and one trained without the held-out donors. Output: PCC tables per gate source,
gate statistics, reference-slide diagnostics, and a summary."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

GRID = np.linspace(0, 1, 41).astype(np.float32)
KS = (20, 50, 100)
MET = ['ALL'] + [f'HVG{k}' for k in KS] + [f'SVG{k}' for k in KS]
ARMS = [('(a)', 'all_slides', 'largest4'), ('(a, all refs)', 'all_slides', 'all'),
        ('(b)', 'held_out', 'train'), ('(c)', 'held_out', 'val')]


def ctr(X):
    return X - X.mean(0, keepdims=True)


def pcc_cols(P, Y):
    Pc, Yc = ctr(P.astype(np.float64)), ctr(Y.astype(np.float64)); num = (Pc * Yc).sum(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    return np.where(den > 1e-8, num / np.where(den > 1e-8, den, 1.0), 0.0)


def best_w(A, R, Y):
    """First grid point reaching the strict maximum of PCC(w*A+(1-w)*R, Y) (inputs centered)."""
    best = np.full(A.shape[1], -2.0); bw = np.ones(A.shape[1], np.float32)
    for w in GRID:
        r = pcc_cols(w * A + (1 - w) * R, Y); t = r > best
        best = np.where(t, r, best); bw = np.where(t, w, bw)
    return bw.astype(np.float32), best


def moments(A, R, Y, chunk=256):
    """Per-gene centered second moments (float64), so every beta mix is scored exactly without forming it."""
    out = {k: np.zeros(A.shape[1]) for k in ('aa', 'rr', 'ar', 'ay', 'ry', 'yy')}
    for j in range(0, A.shape[1], chunk):
        a = ctr(A[:, j:j + chunk].astype(np.float64)); y = ctr(Y[:, j:j + chunk].astype(np.float64))
        r = ctr(np.nan_to_num(R[:, j:j + chunk].astype(np.float64)))
        for k, v in (('aa', a * a), ('rr', r * r), ('ar', a * r), ('ay', a * y), ('ry', r * y), ('yy', y * y)):
            out[k][j:j + chunk] = v.sum(0)
    return out


def pcc_mix(M, b):
    num = b * M['ay'] + (1 - b) * M['ry']
    den = np.sqrt(np.clip(b * b * M['aa'] + (1 - b) ** 2 * M['rr'] + 2 * b * (1 - b) * M['ar'], 0, None) * M['yy'])
    return np.where(den > 1e-8, num / np.where(den > 1e-8, den, 1.0), 0.0)


def fit_references(cache_dir, tissue, test_ids):
    """Per reference cache (cache_references.py): grid w on covered test-panel genes + branch PCC diagnostics."""
    out = {}
    for f in sorted(Path(cache_dir).glob(f'{tissue}__*.npz')):
        z = np.load(f, allow_pickle=True); panel = z['panel'].astype(np.int64)
        cov = np.array([j for j in z['covered'].astype(int) if int(panel[j]) in test_ids], np.int64)
        if not len(cov): continue
        A, R, Y = z['A'][:, cov], np.nan_to_num(z['R'][:, cov]), z['Y'][:, cov]
        w, ps = best_w(ctr(A), ctr(R), ctr(Y))
        out[str(z['slide'])] = dict(gid=panel[cov], w=w, n_cells=int(z['n_cells']), pccA=pcc_cols(A, Y).mean(),
                                    pccR=pcc_cols(R, Y).mean(), pccS=ps.mean())
    return out


def gate(fits, refs):
    acc = {}
    for r in refs:
        if r in fits:
            for g, w in zip(fits[r]['gid'], fits[r]['w']): acc.setdefault(int(g), []).append(float(w))
    return {g: (float(np.mean(v)), len(v), float(np.std(v))) for g, v in acc.items()}


def target_arrays(prediction, directory, gene_list):
    """A (direct), R (retrieval, NaN where uncovered) and measured Y over the canonical real genes.
    prediction: predict.py --bank output (.h5ad: X = direct, layers['R']) or NPZ with A, R, gid, Ylog."""
    gl = pd.read_csv(gene_list, sep='\t'); G = len(gl)
    gid = np.where(gl.in_head.values == 1, gl.global_id.values, -1).astype(np.int64)
    if str(prediction).endswith('.npz'):
        z = np.load(prediction, allow_pickle=True)
        if not np.array_equal(z['gid'], gid): raise ValueError(f'{prediction}: gid differs from {gene_list}')
        return z['A'].astype(np.float32), z['R'].astype(np.float32), z['Ylog'].astype(np.float32), gl, gid
    import anndata as ad
    from predict.inputs import PreparedSlide
    h = ad.read_h5ad(prediction, backed='r'); col = {g: i for i, g in enumerate(h.var_names)}
    src = PreparedSlide(directory); Yr, genes = src.expression()
    if h.n_obs != len(src) or not np.array_equal(h.obs_names.astype(str), src.ids):
        raise ValueError(f'{prediction}: cells differ from {directory}')
    yc = {g: i for i, g in enumerate(genes)}
    miss = [g for g in gl.gene if g not in yc]
    if miss: raise ValueError(f'{directory}: {len(miss)} gene-list genes absent from the expression: {miss[:5]}')
    A = np.zeros((h.n_obs, G), np.float32); R = np.full((h.n_obs, G), np.nan, np.float32)
    for j, (g, ok) in enumerate(zip(gl.gene, gl.in_head.values == 1)):
        if ok and g in col:
            A[:, j] = np.asarray(h.X[:, col[g]]).ravel(); R[:, j] = np.asarray(h.layers['R'][:, col[g]]).ravel()
    return A, R, Yr[:, [yc[g] for g in gl.gene]], gl, gid


def scores(p, gl):
    r = {'ALL': float(p.mean())}
    for k in KS:
        r[f'HVG{k}'] = float(p[gl.hvg_rank.values < k].mean()); r[f'SVG{k}'] = float(p[gl.svg_rank.values < k].mean())
    return r


def table(L, rows, cols, fmt='{:.4f}'):
    L += ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    L += ['| ' + ' | '.join(x if isinstance(x, str) else fmt.format(x) for x in r) + ' |' for r in rows]
    L.append('')


def summarize(df, ref, split, out):
    T = list(split['tissues']); L = ['# Donor-held-out fusion-gate calibration\n']
    g = lambda m, v: df[(df.model == m) & (df.variant == v)].set_index('tissue').reindex(T)
    for bank in ('slide_out', 'donor_out'):
        L.append(f'## ALL, reference bank: {bank}\n'); rows = []
        for lab, m, rs in ARMS:
            x = g(m, f'gate_{rs}_{bank}')
            if x.ALL.notna().any():
                rows.append([lab] + list(x.ALL) + [x.ALL.mean(), x.HVG50.mean(), x.SVG50.mean(), x.beta_mean.mean()])
        for m in ('all_slides', 'held_out'):
            for v in ('direct', 'oracle'):
                x = g(m, v)
                if x.ALL.notna().any():
                    rows.append([f'{m}, {v}'] + list(x.ALL) + [x.ALL.mean(), x.HVG50.mean(), x.SVG50.mean(), x.beta_mean.mean()])
        table(L, rows, ['arm'] + T + ['Macro', 'Macro HVG50', 'Macro SVG50', 'mean beta'])
        rows = []
        for lab, (m1, v1), (m2, v2) in [('(c) - (b)', ('held_out', f'gate_val_{bank}'), ('held_out', f'gate_train_{bank}')),
                                        ('(a) - (c)', ('all_slides', f'gate_largest4_{bank}'), ('held_out', f'gate_val_{bank}')),
                                        ('(a) - (b)', ('all_slides', f'gate_largest4_{bank}'), ('held_out', f'gate_train_{bank}')),
                                        ('(c) - held_out direct', ('held_out', f'gate_val_{bank}'), ('held_out', 'direct'))]:
            a, b = g(m1, v1), g(m2, v2)
            if a.ALL.isna().all() or b.ALL.isna().all(): continue
            d = a[MET].values - b[MET].values
            rows.append([lab] + [f'{v:+.4f}' for v in d[:, 0]] + [f'{d[:, 0].mean():+.4f}', f'{int((d[:, 0] > 0).sum())}/{len(T)}',
                                                                  f'{int((d.mean(0) > 0).sum())}/7'])
        table(L, rows, ['difference'] + T + ['Macro', 'slides > 0', 'metrics > 0'])
    L.append('## Reference slides (test-panel genes)\n')
    rows = [[t, m, b, role, str(len(x)), x.pccA.mean(), x.pccR.mean(), x.pccS.mean(), x.w_mean.mean()]
            for (t, m, b, role), x in ref.groupby(['tissue', 'model', 'bank', 'role'])]
    table(L, rows, ['tissue', 'model', 'bank', 'role', 'n', 'PCC A', 'PCC R', 'best mix', 'mean w'])
    Path(out, 'summary.md').write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML: split, targets {tissue: {dir, gene_list}}, models '
                    '{all_slides|held_out: {predictions: {tissue: path}, references: {slide_out|donor_out: dir}}}')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); cfg = yaml.safe_load(Path(a.config).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(cfg['split']).read_text()); rows, pergene, refrows = [], [], []
    for t, sp in split['tissues'].items():
        tgt = cfg['targets'][t]; pool = sp['train'] + sp['val']
        for m, mc in cfg['models'].items():
            A, R, Y, gl, gid = target_arrays(mc['predictions'][t], tgt['dir'], tgt['gene_list'])
            G = len(gl); cov = ~np.isnan(R).all(0) & (gid >= 0); M = moments(A, R, Y); test_ids = set(gid[gid >= 0].tolist())
            pA = pcc_mix(M, np.ones(G)); pR = np.where(cov, pcc_mix(M, np.zeros(G)), pA)
            curve = np.stack([pcc_mix(M, np.full(G, w)) for w in GRID]); bO = np.where(cov, GRID[curve.argmax(0)], 1.0)
            variants = {'direct': (np.ones(G), None), 'retrieval': (np.where(cov, 0.0, 1.0), None),
                        'fixed_0.5': (np.where(cov, 0.5, 1.0), None), 'oracle': (bO, None)}
            for bank, cdir in mc['references'].items():
                fits = fit_references(cdir, t, test_ids)
                size = {r: fits[r]['n_cells'] for r in fits}
                refsets = {'largest4': sorted(pool, key=lambda r: -size.get(r, 0))[:4], 'all': pool,
                           'train': sp['train'], 'val': sp['val']}
                for rs, refs in refsets.items():
                    gt = gate(fits, refs); has = np.array([cov[j] and int(x) in gt for j, x in enumerate(gid)])
                    b = np.array([gt[int(x)][0] if h else 1.0 for x, h in zip(gid, has)])
                    variants[f'gate_{rs}_{bank}'] = (b, dict(nref=np.array([gt[int(x)][1] if h else 0 for x, h in zip(gid, has)]),
                                                         sd=np.array([gt[int(x)][2] if h else np.nan for x, h in zip(gid, has)]), has=has))
                for r, f in fits.items():
                    refrows.append(dict(model=m, tissue=t, bank=bank, slide=r, role='val' if r in sp['val'] else 'train',
                                        n_genes=len(f['gid']), pccA=f['pccA'], pccR=f['pccR'], pccS=f['pccS'], w_mean=float(f['w'].mean())))
            for v, (b, ex) in variants.items():
                p = pA if v == 'direct' else (pR if v == 'retrieval' else np.where(cov, pcc_mix(M, b), pA))
                gated = ex['has'] if ex else cov & (b < 1)
                rows.append(dict(model=m, variant=v, tissue=t, n_genes=G, n_covered=int(cov.sum()), n_gated=int(gated.sum()),
                                 beta_mean=float(b[gated].mean()) if gated.any() else np.nan,
                                 refs_per_gene=float(ex['nref'][gated].mean()) if ex and gated.any() else np.nan,
                                 beta_sd_across_refs=float(np.nanmean(ex['sd'][gated])) if ex and gated.any() else np.nan,
                                 pcc_covered=float(p[cov].mean()) if cov.any() else np.nan, **scores(p, gl)))
                pergene.append(pd.DataFrame(dict(model=m, variant=v, tissue=t, gene=gl.gene, global_id=gid,
                                                 covered=cov.astype(int), beta=b, pcc=p)))
            print(f'{t} {m}: ' + ' '.join(f"{r['variant']}={r['ALL']:.4f}" for r in rows if r['model'] == m and r['tissue'] == t), flush=True)
    df = pd.DataFrame(rows); ref = pd.DataFrame(refrows)
    df.to_csv(out / 'results.tsv', sep='\t', index=False); ref.to_csv(out / 'references.tsv', sep='\t', index=False)
    pd.concat(pergene).to_csv(out / 'pergene.tsv.gz', sep='\t', index=False, float_format='%.6g')
    summarize(df, ref, split, out)
    print(f'-> {out}/results.tsv, references.tsv, pergene.tsv.gz, summary.md', flush=True)


if __name__ == '__main__':
    main()
