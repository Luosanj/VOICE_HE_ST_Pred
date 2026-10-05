"""Cache reference A/R/Y for beta analysis. Input: same-tissue prepared slides, model weights, and bank. Output: per-reference NPZ arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--slides',required=True,help='YAML: slides list with name, tissue, dir, bank, optional exclude')
    ap.add_argument('--release');ap.add_argument('--stage1');ap.add_argument('--stage2')
    ap.add_argument('--global_genes',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--cells',type=int,default=30000);ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    from voice.release import load_release
    from voice.panel import sym2glob_map
    from voice.retrieval import crossR
    from predict.inputs import PreparedSlide
    from predict.geometry import tile
    from predict.predict import run
    from predict.build_bank import weights_id
    model,head,cfg=load_release(a.release,device=a.device,stage1=a.stage1,stage2=a.stage2)
    mapping,_=sym2glob_map(a.global_genes);wid=weights_id(a.release,a.stage1,a.stage2)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for entry in yaml.safe_load(Path(a.slides).read_text())['slides']:
        src=PreparedSlide(entry['dir']);Y,genes=src.expression()
        if Y is None: raise ValueError('Reference expression is required')
        panel=np.array([mapping.get(g,-1) for g in genes],np.int64)
        valid=(panel>=0)&(panel<cfg['n_genes']);panel=panel[valid];Y=Y[:,valid]
        A,E=run(model,head,src,tile(src.pos,256),cfg['n_genes'],a.workers,a.device,True,True)
        sel=np.random.default_rng(a.seed).choice(len(src),a.cells,replace=False) if a.cells and len(src)>a.cells else np.arange(len(src))
        R,cov=crossR(E[sel],panel,entry['bank'],exclude=set(entry.get('exclude',[]))|{entry['name'],Path(entry['dir']).name},weights_id=wid,device=a.device)
        np.savez_compressed(out/f"{entry['tissue']}__{entry['name']}.npz",slide=entry['name'],tissue=entry['tissue'],
            A=A[sel][:,panel],R=R,Y=Y[sel],panel=panel,sel=sel,covered=np.asarray(cov),n_cells=len(src))
        print(entry['name'],len(sel),flush=True)


if __name__=='__main__': main()
