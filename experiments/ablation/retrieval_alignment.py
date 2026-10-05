"""Compare retrieval with frozen and aligned features. Input: per-variant query NPZs and same-tissue banks. Output: covered-gene PCC and paired per-gene tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from voice.retrieval import crossR, list_bank
from voice.metrics_bench import per_gene_pcc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--slides',required=True,help='YAML: slides list of name, variants mapping {label: {query, bank}}, optional exclude')
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--knn',type=int,default=200);ap.add_argument('--tau',type=float,default=.03)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sl in yaml.safe_load(Path(a.slides).read_text())['slides']:
        common_banks=None;common_genes=None;common_pos=None;common_Y=None;pcs={}
        for label,entry in sl['variants'].items():
            with np.load(entry['query'],allow_pickle=True) as q:
                E,Y,gid,genes,pos=(q[k] for k in ['E1536','Ylog','panel','genes','pos'])
                excluded=set(sl.get('exclude',[]))|{sl['name'],str(q['slide'])}
                banks=sorted(set(list_bank(entry['bank']))-excluded)
                if common_banks is not None and banks!=common_banks:raise ValueError('Reference slides differ between variants')
                if common_genes is not None and (not np.array_equal(genes,common_genes) or not np.array_equal(pos,common_pos) or not np.array_equal(Y,common_Y)):
                    raise ValueError('Query cells, genes, or measured expression differ between variants')
                common_banks,common_genes,common_pos,common_Y=banks,genes,pos,Y
                R,cov=crossR(E,gid,entry['bank'],names=banks,exclude=excluded,K=a.knn,tau=a.tau,
                             weights_id=str(q['weights_id']),device=a.device)
            pc=np.full(len(genes),np.nan);pc[cov]=per_gene_pcc(R[:,cov],Y[:,cov]);pcs[label]=pc
        shared=np.logical_and.reduce([np.isfinite(pc) for pc in pcs.values()])
        if not shared.any():raise ValueError('No genes covered in every variant')
        for label,pc in pcs.items():rows.append(dict(slide=sl['name'],variant=label,n_covered_genes=int(shared.sum()),covered_gene_pcc=float(pc[shared].mean())))
        pd.DataFrame({'gene':common_genes,'shared_coverage':shared,**pcs}).to_csv(out/f"{sl['name']}.tsv",sep='\t',index=False)
    pd.DataFrame(rows).to_csv(out/'summary.tsv',sep='\t',index=False)
    pd.DataFrame(rows).groupby('variant').covered_gene_pcc.mean().to_csv(out/'macro.tsv',sep='\t')


if __name__=='__main__':main()
