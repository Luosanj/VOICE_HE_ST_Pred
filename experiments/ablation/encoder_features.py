"""Extract frozen or aligned cell features. Input: prepared slide, encoder checkpoints, and global genes. Output: reference bank file or query NPZ."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',required=True)
    ap.add_argument('--mode',choices=['frozen','stage1','stage2'],required=True)
    ap.add_argument('--stage1');ap.add_argument('--stage2');ap.add_argument('--release')
    ap.add_argument('--global_genes',required=True)
    ap.add_argument('--bank_dir',help='write a reference .bank.npz instead of a query')
    ap.add_argument('--out',help='query NPZ')
    ap.add_argument('--name');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    if bool(a.bank_dir)==bool(a.out):ap.error('Supply --bank_dir or --out')
    from voice.encoder import build_uni2, inject_lora, load_lora, pooled_feat, MEAN, STD
    from voice.release import read_weights, load_release
    from voice.retrieval import save_bank_slide
    from voice.panel import sym2glob_map
    from predict.inputs import PreparedSlide
    from predict.predict import TileDS
    from predict.geometry import tile
    from predict.build_bank import weights_id
    dev=torch.device(a.device)
    if a.mode=='stage2':model,_head,_cfg=load_release(a.release,device=dev,stage1=a.stage1,stage2=a.stage2)
    else:
        model=build_uni2(dev)
        if a.mode=='stage1':
            if not a.stage1:ap.error('--stage1 is required for stage1 mode')
            w=read_weights(a.stage1);inject_lora(model,int(w.get('nblocks',12)),int(w.get('r',16)),int(w.get('alpha',32)),float(w.get('dropout',.05)))
            model.to(dev);load_lora(model,w['lora']);model.eval()
    source=PreparedSlide(a.prepared);Y,genes=source.expression()
    if Y is None:raise ValueError('Measured expression is required')
    mapping,_=sym2glob_map(a.global_genes);panel=np.array([mapping.get(g,-1) for g in genes]);keep=panel>=0
    panel,Y,genes=panel[keep],Y[:,keep],np.asarray(genes)[keep]
    E=np.empty((len(source),1536),np.float32);mean,std=MEAN.to(dev),STD.to(dev)
    dl=DataLoader(TileDS(source,tile(source.pos,256)),batch_size=1,num_workers=a.workers,collate_fn=lambda b:b[0])
    amp=torch.bfloat16 if dev.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for images,masks,pos,indices in dl:
            x=(images.to(dev).float()/255.-mean)/std
            with torch.autocast(dev.type,dtype=amp,enabled=dev.type=='cuda'):
                f=pooled_feat(model,x,masks.to(dev))
            E[indices.numpy()]=f.float().cpu().numpy()
    identity='UNI2-h-frozen' if a.mode=='frozen' else ('stage1-'+Path(a.stage1).name if a.mode=='stage1' else weights_id(a.release,a.stage1,a.stage2))
    if a.bank_dir:save_bank_slide(a.bank_dir,a.name or Path(a.prepared).name,E,Y,source.pos,panel,identity)
    else:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.out,E1536=E,Ylog=Y,pos=source.pos,panel=panel,genes=genes,weights_id=identity,slide=a.name or Path(a.prepared).name)


if __name__=='__main__':main()
