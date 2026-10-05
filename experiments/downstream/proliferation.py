"""Evaluate the panel-covered G2/M score. Input: aligned predictions, raw expression, and tumor subtype labels. Output: AUROC, paired-score PCC, and per-cell scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
from experiments.common import pair_arguments, load_pair
GENES=['CCND1','CENPF','HMGA1','MKI67','SQLE','TOP2A']
TUMOR=['Invasive_Tumor','Prolif_Invasive_Tumor','DCIS_1','DCIS_2']
PAIR=['Invasive_Tumor','Prolif_Invasive_Tumor']
def corr(x,y,w=None):
 if w is None:return float(stats.pearsonr(x,y).statistic)
 sw=w.sum();xc=x-np.dot(w,x)/sw;yc=y-np.dot(w,y)/sw
 return float(np.dot(w,xc*yc)/np.sqrt(np.dot(w,xc*xc)*np.dot(w,yc*yc)))


def score(a,mu,sd):return ((a-mu)/sd).sum(axis=1)/np.sqrt(a.shape[1])


def params(a,w=None):
 if w is None:return a.mean(0),a.std(0,ddof=1)
 n=w.sum();mu=np.sum(a*w[:,None],axis=0)/n
 sd=np.sqrt(np.sum((a-mu)**2*w[:,None],axis=0)/(n-1))
 return mu,sd


def main():
    ap=argparse.ArgumentParser(); pair_arguments(ap); a=ap.parse_args()
    Y,P,genes,pos,cells=load_pair(a)
    if cells is None: raise ValueError('--cells is required for tumor labels')
    ix=[list(genes).index(g) for g in GENES]
    Y,P=Y[:,ix].astype(np.float64),P[:,ix].astype(np.float64)
    tumor=cells.cell_type.isin(TUMOR).to_numpy(); pair=cells.cell_type.isin(PAIR).to_numpy()
    binary=cells.loc[pair,'cell_type'].eq('Prolif_Invasive_Tumor').astype(int).to_numpy()
    my,sy=params(Y[tumor]); mp,sp=params(P[tumor])
    if not ((sy>0).all() and (sp>0).all()): raise ValueError('Constant score genes')
    measured,predicted=score(Y,my,sy),score(P,mp,sp)
    out=Path(a.out)
    summary=dict(genes=GENES,n_reference_cells=int(tumor.sum()),n_evaluation_cells=int(pair.sum()),
                 predicted_auc=float(roc_auc_score(binary,predicted[pair])),
                 measured_auc=float(roc_auc_score(binary,measured[pair])),score_pair_pearson=corr(measured[pair],predicted[pair]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(dict(gene=GENES,measured_mean=my,measured_sd=sy,predicted_mean=mp,predicted_sd=sp)).to_csv(out/'normalization_parameters.csv',index=False)
    result=cells.copy(); result['measured_g2m_score']=measured; result['predicted_g2m_score']=predicted
    result.to_csv(out/'per_cell_scores.csv.gz',index=False)


if __name__=='__main__': main()
