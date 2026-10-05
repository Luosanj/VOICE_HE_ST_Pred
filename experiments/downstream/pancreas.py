"""Recover pancreatic domains, DEGs, and pathway scores. Input: paired expression/predictions, frozen PCA/centroids, spatial neighbors, and Reactome GMT. Output: domain labels, shared DEGs, block and pathway correlations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats, sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from experiments.common import pair_arguments, load_pair
MIN_BLOCKS=10
def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        out[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return out


def from_stats(mean, sd, n):
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    p = np.full(len(mean), np.nan); lo=p.copy(); hi=p.copy(); t=p.copy()
    if n >= MIN_BLOCKS:
        se=sd/np.sqrt(n)
        good=np.isfinite(mean)&np.isfinite(sd)&(sd>0)
        t[good]=mean[good]/se[good]
        p[good]=2*stats.t.sf(np.abs(t[good]),n-1)
        zero=(sd==0)&(mean==0);p[zero]=1;t[zero]=0
        crit=stats.t.ppf(.975,n-1)
        lo[good]=mean[good]-crit*se[good];hi[good]=mean[good]+crit*se[good]
        lo[zero]=0;hi[zero]=0
    return t,p,lo,hi
def differences(Y,labels,xy,size=500,offset=0):
    grid,block=np.unique(np.floor((xy-offset)/size).astype(np.int32),axis=0,return_inverse=True)
    n=len(grid);N=len(labels);ops=[];counts=[]
    for mask in [labels,~labels]:
        ix=np.flatnonzero(mask)
        op=csr_matrix((np.ones(len(ix)),(block[ix],ix)),shape=(n,N))
        counts.append(np.asarray(op.sum(1)).ravel());ops.append(op)
    good=(counts[0]>=5)&(counts[1]>=5)
    means=[np.asarray(op@Y)/np.maximum(ct,1)[:,None] for op,ct in zip(ops,counts)]
    return (means[0]-means[1])[good],good,grid
def internal_block_means(Y, region, block, n_blocks):
    ix = np.flatnonzero(region)
    counts = np.bincount(block[ix], minlength=n_blocks)
    op = csr_matrix((1 / counts[block[ix]], (block[ix], ix)), shape=(n_blocks, len(region)))
    means = np.asarray(op @ Y)
    return means, counts


def column_corr(a, b):
    aa, bb = a - a.mean(0), b - b.mean(0)
    denom = np.sqrt((aa * aa).sum(0) * (bb * bb).sum(0))
    return np.divide((aa * bb).sum(0), denom,
                     out=np.full(a.shape[1], np.nan), where=denom > 0)
def knn_mean_operator(idx: np.ndarray, k: int, include_self: bool = False) -> sparse.csr_matrix:
    """Row-stochastic sparse matrix averaging over the first k neighbours (optionally + self)."""
    n = idx.shape[0]
    cols = idx[:, :k]
    if include_self:
        cols = np.concatenate([np.arange(n)[:, None], cols], 1)
    m = cols.shape[1]
    rows = np.repeat(np.arange(n), m)
    data = np.full(rows.shape[0], 1.0 / m, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols.ravel())), shape=(n, n))


def main():
    ap=argparse.ArgumentParser();pair_arguments(ap)
    ap.add_argument('--frozen_model',help='NPZ: mean, sd, pca_mean, pca_components, centers, knn14, genes')
    ap.add_argument('--knn',help='NPZ spatial-neighbor indices (idx), needed when fitting the published domains')
    ap.add_argument('--reactome',required=True,help='MSigDB Reactome GMT')
    ap.add_argument('--domain',type=int,default=2)
    ap.add_argument('--clusters',type=int,default=6)
    a=ap.parse_args();Y,P,genes,pos,cells=load_pair(a);out=Path(a.out)
    keep=P.std(0)>0;Y,P,genes=Y[:,keep],P[:,keep],genes[keep]
    xy=np.stack([pos[:,1]*a.mpp,pos[:,0]*a.mpp],1).astype(np.float32)
    if a.frozen_model:
        f=np.load(a.frozen_model)
        if not np.array_equal(f['genes'],genes): raise ValueError('Frozen model gene order differs')
        mu,sd,components,pcmean,centers,knn=(f[k] for k in ['mean','sd','pca_components','pca_mean','centers','knn14'])
        pt=((Y-mu)/sd-pcmean)@components.T;pp=((P-mu)/sd-pcmean)@components.T
    else:
        mu,sd=Y.mean(0),Y.std(0)+1e-6
        pca=PCA(30,svd_solver='randomized',random_state=0).fit((Y-mu)/sd)
        pt=pca.transform((Y-mu)/sd);pp=pca.transform((P-mu)/sd)
        components,pcmean=pca.components_,pca.mean_
        knn=np.load(a.knn)['idx'][:,:14] if a.knn else NearestNeighbors(n_neighbors=15,algorithm='kd_tree').fit(xy).kneighbors(xy,return_distance=False)[:,1:]
    W=knn_mean_operator(knn,14,True);pt=np.asarray(W@pt,np.float32);pp=np.asarray(W@pp,np.float32)
    if not a.frozen_model: centers=KMeans(a.clusters,n_init=10,random_state=0).fit(pt).cluster_centers_
    dt=pairwise_distances_argmin(pt,centers).astype(np.int8);dp=pairwise_distances_argmin(pp,centers).astype(np.int8)
    tm,pm=dt==a.domain,dp==a.domain
    np.savez_compressed(out/'frozen_model.npz',genes=genes,mean=mu,sd=sd,pca_mean=pcmean,pca_components=components,centers=centers,knn14=knn)
    np.savez_compressed(out/'cell_assignments.npz',genes=genes,xy_um=xy,truth_domain=dt,predicted_domain=dp)
    at,vt,grid=differences(Y,tm,xy);apred,vp,_=differences(P,pm,xy);common=vt&vp
    at,apred=at[common[vt]],apred[common[vp]]
    if len(at)<MIN_BLOCKS: raise ValueError('Fewer than ten eligible paired blocks')
    tq=bh(from_stats(at.mean(0),at.std(0,ddof=1),len(at))[1]);pq=bh(from_stats(apred.mean(0),apred.std(0,ddof=1),len(apred))[1])
    shared=(tq<.05)&(pq<.05)
    table=pd.DataFrame(dict(gene=genes,truth_effect=at.mean(0),prediction_effect=apred.mean(0),truth_q=tq,prediction_q=pq,shared=shared))
    table.to_csv(out/'all_degs.tsv',sep='\t',index=False);table[shared].to_csv(out/'shared_degs.tsv',sep='\t',index=False)
    blockgrid,blocks=np.unique(np.floor(xy/500).astype(np.int32),axis=0,return_inverse=True)
    yt,nt=internal_block_means(Y,tm,blocks,len(blockgrid));yp,npred=internal_block_means(P,pm,blocks,len(blockgrid))
    good=(nt>=5)&(npred>=5)
    gene_pcc=column_corr(yt[good],yp[good])
    pd.DataFrame(dict(gene=genes,block_expression_pearson=gene_pcc)).to_csv(out/'gene_block_correlations.tsv',sep='\t',index=False)
    mean=Y.mean(0,dtype=np.float64);score_sd=np.maximum(Y.std(0,dtype=np.float64),.1)
    lookup={g:i for i,g in enumerate(genes)};pathways=[];weights=[]
    for line in Path(a.reactome).read_text().splitlines():
        fields=line.split('\t');members=set(fields[2:]);present=sorted(members&set(genes))
        if len(present)<5: continue
        w=np.zeros(len(genes));w[[lookup[g] for g in present]]=1/len(present)
        weights.append(w);pathways.append(dict(pathway_id=fields[0],n_measured=len(present),measured_genes=';'.join(present)))
    if not weights: raise ValueError('No Reactome pathways with at least five panel genes')
    weights=np.stack(weights,axis=1)
    st=((yt[good]-mean)/score_sd)@weights;sp=((yp[good]-mean)/score_sd)@weights
    pathways=pd.DataFrame(pathways);pathways['internal_score_pearson']=column_corr(st,sp)
    pathways.to_csv(out/'pathway_correlations.tsv',sep='\t',index=False)
    np.savez_compressed(out/'paired_block_expression.npz',genes=genes,block_grid=blockgrid[good],truth=yt[good],prediction=yp[good])
    summary=dict(domain_agreement=float(np.mean(dt==dp)),domain_ari=float(adjusted_rand_score(dt,dp)),
        n_shared_degs=int(shared.sum()),n_deg_blocks=len(at),n_internal_blocks=int(good.sum()),domain=a.domain)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
