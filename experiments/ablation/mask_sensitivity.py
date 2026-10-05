"""Recompute predictions under boundary perturbations. Input: prepared slide, weights, canonical gene list, bank, and gate. Output: mask statistics, predictions, and PCC tables."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader
CONDITIONS=[('original','original',0.,0,0)]
for um in (1.,2.):
    CONDITIONS += [(f'{op}_{um:g}um',op,um,0,0) for op in ('erosion','dilation')]
    CONDITIONS += [(f'translation_{name}_{um:g}um','translation',um,dy,dx)
        for name,dy,dx in [('up',-1,0),('down',1,0),('left',0,-1),('right',0,1)]]
def tiles(pos):
    """Same tile membership and cell ordering as se2_lora_v2_eval.tile."""
    key = np.floor(pos / 256).astype(np.int64)
    order = np.lexsort((np.arange(len(pos)), key[:, 1], key[:, 0]))
    splits = np.flatnonzero(np.any(np.diff(key[order], axis=0), axis=1)) + 1
    return np.split(order, splits)


def translate(mask, dy, dx):
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = mask[
        max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


def masks_and_weights(mask, umpp):
    inside = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    outside = distance_transform_edt(~mask) if mask.any() else None
    weights, stats = [], []
    for _, op, um, dy, dx in CONDITIONS:
        radius = um / umpp
        if op == 'original':
            m = mask
        elif op == 'erosion':
            m = mask & (inside > radius)
        elif op == 'dilation':
            m = (outside <= radius) if outside is not None else mask
        else:
            shift = int(round(radius))
            m = translate(mask, dy * shift, dx * shift)
        area = int(m.sum())
        union = int((m | mask).sum())
        iou = float((m & mask).sum() / union) if union else 1.
        w = m.astype(np.float32).reshape(16, 14, 16, 14).mean((1, 3))
        fallback = not bool(area)
        if fallback:
            w[8, 8] = 1.
        weights.append(w / w.sum())
        stats.append((area, iou, int(fallback)))
    return np.stack(weights), np.array(stats, dtype=np.float32)


class CropDS(Dataset):
    def __init__(self, source, boundary, rows, umpp):
        man = pd.read_csv(source / 'manifest.csv.gz', usecols=['expr_row', 'patch_path'])
        man = man.drop_duplicates('expr_row').set_index('expr_row').patch_path
        er = boundary['expr_rows'].astype(np.int64)
        self.paths = [str(source / str(man.loc[int(er[r])])) for r in rows]
        self.rows = rows
        self.ip, self.vx, self.vy = (boundary[k] for k in ('indptr', 'vertex_x_patch', 'vertex_y_patch'))
        self.umpp = umpp

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        r = self.rows[j]
        with Image.open(self.paths[j]) as im:
            image = np.asarray(im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
        a, b = self.ip[r:r + 2]
        mask = Image.new('L', (224, 224), 0)
        if b - a >= 3:
            ImageDraw.Draw(mask).polygon(list(zip(self.vx[a:b].tolist(), self.vy[a:b].tolist())), fill=1)
        weights, stats = masks_and_weights(np.asarray(mask).astype(bool), self.umpp)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), weights, stats


def pcc(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a*a).sum(0) * (b*b).sum(0))
    return np.divide((a*b).sum(0), den, out=np.zeros_like(den), where=den > 1e-8)


def correlation_columns(a, b, chunk=64):
    return np.concatenate([pcc(a[:, j:j+chunk], b[:, j:j+chunk]) for j in range(0, a.shape[1], chunk)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',required=True)
    ap.add_argument('--release')
    ap.add_argument('--stage1'); ap.add_argument('--stage2')
    ap.add_argument('--global_genes',required=True)
    ap.add_argument('--gene_list',required=True)
    ap.add_argument('--bank',required=True)
    ap.add_argument('--gate',required=True)
    ap.add_argument('--exclude',nargs='*',default=[])
    ap.add_argument('--cells',type=int,default=30000)
    ap.add_argument('--seed',type=int,default=20260929)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--batch_size',type=int,default=64)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed)
    from voice.release import load_release
    from voice.encoder import MEAN, STD
    from voice.retrieval import crossR
    from voice.gate import apply_gate
    from predict.build_bank import weights_id
    from voice.panel import sym2glob_map, read_genes
    source=Path(a.prepared)
    boundary=np.load(source/'patch_cell_boundaries.npz',allow_pickle=True)
    if int(boundary['output_size'])!=224: raise ValueError('Expected 224-pixel crop boundaries')
    umpp=float(json.loads((source/'dataset.json').read_text())['um_per_pixel_output'])
    if abs(umpp*224-55)>=1: raise ValueError('Expected 55-um prepared crops')
    pos=np.stack([boundary['y_pixel'],boundary['x_pixel']],1).astype(np.float32)
    alltiles=tiles(pos);perm=np.random.default_rng(a.seed).permutation(len(alltiles))
    if a.cells:
        end=int(np.searchsorted(np.cumsum([len(alltiles[i]) for i in perm]),a.cells))+1
        ids=np.sort(perm[:end])
    else: ids=np.arange(len(alltiles))
    selected=[alltiles[i] for i in ids];rows=np.sort(np.concatenate(selected))
    local=np.full(len(pos),-1,np.int64);local[rows]=np.arange(len(rows))
    groups=[local[t] for t in selected]
    np.save(out/'rows.npy',rows);np.save(out/'pos_yx.npy',pos[rows])
    pg=pd.read_csv(a.gene_list,sep='\t');genes=pg.gene.astype(str).to_numpy()
    symbols,_=sym2glob_map(a.global_genes);gid=np.array([symbols.get(g,-1) for g in genes])
    rawgenes=read_genes(source/'genes.tsv');rawcols=[rawgenes.index(g) for g in genes]
    er=boundary['expr_rows'].astype(np.int64)
    Y=np.log1p(sparse.load_npz(source/'expression.npz').tocsr()[er[rows]][:,rawcols].toarray().astype(np.float32))
    model,head,cfg=load_release(a.release,device=a.device,stage1=a.stage1,stage2=a.stage2)
    if np.any(gid<0) or np.any(gid>=cfg['n_genes']): raise ValueError('Gene list must use model-head genes')
    ds=CropDS(source,boundary,rows,umpp)
    dl=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers,pin_memory=True)
    features=[np.lib.format.open_memmap(out/f'{c[0]}.E.npy',mode='w+',dtype='float32',shape=(len(rows),1536)) for c in CONDITIONS]
    statistics=np.empty((len(rows),len(CONDITIONS),3),np.float32)
    device=torch.device(a.device);mean,std=MEAN.to(device),STD.to(device)
    amp=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    offset=0
    with torch.inference_mode():
        for image,w,st in dl:
            n=len(image);x=(image.to(device).float()/255.-mean)/std;w=w.to(device)
            with torch.autocast(device.type,dtype=amp,enabled=device.type=='cuda'):
                grid=model.forward_features(x)[:,9:].reshape(n,16,16,1536)
                for ci in range(len(CONDITIONS)):
                    f=(grid*w[:,ci].unsqueeze(-1)).sum((1,2))
                    features[ci][offset:offset+n]=f.float().cpu().numpy()
            statistics[offset:offset+n]=st.numpy();offset+=n
    for f in features: f.flush()
    np.save(out/'mask_statistics.npy',statistics)
    del model,features;gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    beta={int(k):float(v) for k,v in json.loads(Path(a.gate).read_text()).items()}
    wid=weights_id(a.release,a.stage1,a.stage2)
    summary=[]
    for ci,(name,op,um,dy,dx) in enumerate(CONDITIONS):
        E=np.load(out/f'{name}.E.npy',mmap_mode='r')
        A=np.empty_like(Y)
        with torch.inference_mode():
            for group in groups:

                pred,_=head(torch.from_numpy(np.array(E[group])).to(device),torch.from_numpy(pos[rows[group]]).to(device))
                A[group]=pred[:,gid].float().cpu().numpy()
        R,cov=crossR(E,gid,a.bank,exclude=set(a.exclude)|{source.name},weights_id=wid,device=a.device)
        S,used=apply_gate(A,R,gid,beta)
        pa=correlation_columns(A,Y);pr=np.zeros(len(genes),np.float64)
        pr[cov]=correlation_columns(R[:,cov],Y[:,cov]);ps=correlation_columns(S,Y)
        for branch,pc in [('direct',pa),('retrieval',pr),('fused',ps)]:
            baseline=statistics[:,0,0];valid=baseline>0
            entry=dict(condition=name,operation=op,requested_um=um,branch=branch,n_cells=len(rows),ALL=float(pc.mean()),
                mean_mask_iou=float(statistics[:,ci,1].mean()),empty_mask_fraction=float(statistics[:,ci,2].mean()),
                mean_area_ratio=float((statistics[valid,ci,0]/baseline[valid]).mean()))
            for k in (20,50,100):
                for tag,rank in [('HVG','hvg_rank'),('SVG','svg_rank')]:
                    selected_genes=np.flatnonzero(np.isfinite(pg[rank])&(pg[rank]>0)&(pg[rank]<=k))
                    entry[f'{tag}{k}']=float(pc[selected_genes].mean()) if len(selected_genes) else np.nan
            summary.append(entry)
        np.savez_compressed(out/f'{name}.npz',A=A,R=R,pred=S,Ylog=Y,genes=genes,gid=gid,pos=pos[rows],beta=used)
        pd.DataFrame(dict(gene=genes,pcc_A=pa,pcc_R=pr,pcc_S=ps)).to_csv(out/f'{name}.pergene.tsv',sep='\t',index=False)
        print(name,float(ps.mean()),flush=True)
    d=pd.DataFrame(summary);baseline=d[d.condition.eq('original')].set_index('branch').ALL
    d['delta_ALL']=d.ALL-d.branch.map(baseline);d.to_csv(out/'summary.tsv',sep='\t',index=False)
    (out/'run_meta.json').write_text(json.dumps(dict(seed=a.seed,stage1=a.stage1,stage2=a.stage2,release=a.release,
        bank=a.bank,gate=a.gate,n_cells=len(rows),um_per_output_pixel=umpp),indent=2))


if __name__=='__main__': main()
