"""Compare measured and predicted subtype differences. Input: aligned expression/predictions and cell annotations. Output: pair, gene, and broad-type CSVs."""
from itertools import combinations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from experiments.common import pair_arguments, load_pair
EPS=1e-10
GROUPS = {
    'T cell': ['CD4+_T_Cells', 'CD8+_T_Cells'],
    'Macrophage': ['Macrophages_1', 'Macrophages_2'],
    'Myoepithelial': ['Myoepi_ACTA2+', 'Myoepi_KRT15+'],
    'Dendritic cell': ['IRF7+_DCs', 'LAMP3+_DCs'],
    'Tumor epithelial': ['Invasive_Tumor', 'Prolif_Invasive_Tumor', 'DCIS_1', 'DCIS_2'],
}

def pcc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x, y = x[mask], y[mask]
    x, y = x-x.mean(), y-y.mean()
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den > 1e-12 else np.nan


def moment_effect(n, sy, sp, sy2, sp2, ia, ib):
    na, nb = n[ia], n[ib]
    if min(na, nb) < 2:
        return None
    mya, myb = sy[ia]/na, sy[ib]/nb
    mpa, mpb = sp[ia]/na, sp[ib]/nb
    vy = np.maximum(0, sy2[ia]-sy[ia]**2/na + sy2[ib]-sy[ib]**2/nb)/(na+nb-2)
    vp = np.maximum(0, sp2[ia]-sp[ia]**2/na + sp2[ib]-sp[ib]**2/nb)/(na+nb-2)
    sy_pool, sp_pool = np.sqrt(vy), np.sqrt(vp)
    dy = np.divide(mya-myb, sy_pool, out=np.zeros_like(mya), where=sy_pool>EPS)
    dp = np.divide(mpa-mpb, sp_pool, out=np.zeros_like(mpa), where=sp_pool>EPS)
    return dy, dp, mya-myb, mpa-mpb, sy_pool, sp_pool


def main():
    ap=argparse.ArgumentParser(); pair_arguments(ap)
    ap.add_argument('--groups', help='JSON mapping broad cell types to subtype labels')
    a=ap.parse_args(); Y,P,genes,pos,cells=load_pair(a)
    if cells is None: raise ValueError('--cells is required for subtype labels')
    groups=json.loads(Path(a.groups).read_text()) if a.groups else GROUPS
    labels=[x for v in groups.values() for x in v]
    keep=cells.cell_type.isin(labels).to_numpy()
    y=Y[keep].astype(np.float64); p=P[keep].astype(np.float64)
    codes=pd.Categorical(cells.loc[keep,'cell_type'],categories=labels).codes
    ns,ng=len(labels),len(genes)
    n=np.bincount(codes,minlength=ns).astype(float)
    sy=np.zeros((ns,ng)); sp=sy.copy(); sy2=sy.copy(); sp2=sy.copy(); detection=sy.copy()
    for mat,out in [(y,sy),(p,sp),(y*y,sy2),(p*p,sp2),((y>0).astype(float),detection)]: np.add.at(out,codes,mat)
    rows=[]; per_gene=[]
    for typ,subtypes in groups.items():
        for sa,sb in combinations(subtypes,2):
            ia,ib=labels.index(sa),labels.index(sb)
            effect=moment_effect(n,sy,sp,sy2,sp2,ia,ib)
            if effect is None: continue
            dy,dp,delta_y,delta_p,sd_y,sd_p=effect
            detected=detection[ia]+detection[ib]
            mask=(detected>=20)&(detected>=np.ceil(.01*(n[ia]+n[ib])))&(sd_y>EPS)
            r=pcc(dy[mask],dp[mask])
            rows.append(dict(main_type=typ,contrast=sa+' vs '+sb,subtype_a=sa,subtype_b=sb,n_a=int(n[ia]),n_b=int(n[ib]),
                             n_eligible_genes=int(mask.sum()),effect_pearson=r))
            per_gene.append(pd.DataFrame(dict(main_type=typ,contrast=sa+' vs '+sb,gene=genes,eligible=mask,
                measured_mean_difference=delta_y,predicted_mean_difference=delta_p,
                measured_pooled_sd=sd_y,predicted_pooled_sd=sd_p,
                measured_standardized_difference=dy,predicted_standardized_difference=dp)))
    out=Path(a.out); pairs=pd.DataFrame(rows)
    if pairs.empty: raise ValueError('No subtype pairs with at least two cells per subtype')
    pairs.to_csv(out/'pair_summary.csv',index=False)
    pd.concat(per_gene).to_csv(out/'pair_per_gene.csv',index=False)
    broad=pairs.groupby('main_type').effect_pearson.agg(['mean','count']).reset_index()
    broad.to_csv(out/'broad_type_summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps(dict(equal_type_mean_pcc=float(broad['mean'].mean()),n_types=len(broad))))


if __name__=='__main__': main()
