#!/usr/bin/env python
"""Predict single-cell gene expression, and per-cell embeddings, from H&E.

Two ways to give it a slide.

**A whole-slide image you just scanned.** Segment, then predict:

    python predict/segment.py  --image slide.svs --out cells.npz --mpp 0.25
    python predict/predict.py  --image slide.svs --cells cells.npz --mpp 0.25 \\
                               --release weights/voice-23m --out pred.h5ad

**A slide already prepared in the corpus layout** (`patch_cell_boundaries.npz` + `manifest.csv.gz` +
`patches/`), which is what the training and benchmark data look like:

    python predict/predict.py --prepared /data/slides/my_slide \\
                              --release weights/voice-23m --out pred.h5ad

The prepared path needs no `--mpp`: crops were cut at the right physical size when the slide was built, and the
polygons are already in crop coordinates. It is also exactly reproducible, since the same PNG bytes reach the
encoder every run. See `predict/inputs.py` for both layouts.

**Embeddings.** `--save_embeddings` stores the 1536-d per-cell feature the gene head reads from, in
`obsm["X_voice"]`. That vector is the model's representation of the cell's morphology, and it is what to use for
downstream tasks — cell-type classification, clustering, integration — rather than the 6029 predicted genes,
which are a lossy view of it. With `--embeddings_only` the gene head is skipped entirely.

Cells are grouped into non-overlapping 256-px tiles so each is predicted once with its neighbours as spatial
context, matching the evaluation protocol.

This runs the direct branch (Stages 1-2). The retrieval branch and the Stage-3 gate need a reference bank of
millions of embedded cells from the same tissue, which cannot ship with the code; `benchmark/` shows how to build
one from your own reference slides.
"""
from __future__ import annotations
import os, sys, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from voice import paths as _p
_p.hf_home()                      # HF_HOME / *_OFFLINE before timm imports
import torch
from torch.utils.data import Dataset, DataLoader

from voice.encoder import pooled_feat, MEAN, STD
from predict.geometry import tile, FOV_UM
from predict.inputs import open_slide


class TileDS(Dataset):
    """One item = one tile of cells: their crops, masks, positions and indices.

    Crops are produced in the worker processes, so image decode overlaps the GPU forward.
    """
    def __init__(self, source, tiles):
        self.src = source; self.tiles = tiles

    def __len__(self): return len(self.tiles)

    def __getitem__(self, k):
        cells = self.tiles[k]
        imgs, W = self.src.batch(cells)
        return (torch.from_numpy(imgs), torch.from_numpy(W),
                torch.from_numpy(self.src.pos[cells]), torch.from_numpy(cells.astype(np.int64)))


@torch.no_grad()
def run(model, se2, source, tiles, n_genes, workers, dev, want_genes=True, want_emb=False):
    """Returns (pred [N,n_genes] or None, feats [N,1536] float32 or None)."""
    dl = DataLoader(TileDS(source, tiles), batch_size=1, num_workers=workers,
                    collate_fn=lambda b: b[0], pin_memory=True,
                    prefetch_factor=4 if workers else None)
    n = len(source)
    pred = np.zeros((n, n_genes), np.float32) if want_genes else None
    emb = np.zeros((n, 1536), np.float32) if want_emb else None
    mean = MEAN.to(dev); std = STD.to(dev)
    done = 0; t0 = time.time()
    for imgs, w, p, cells in dl:
        x = imgs.to(dev, non_blocking=True).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = pooled_feat(model, x, w.to(dev, non_blocking=True))
            if want_genes:
                lm, _ = se2(feats.float(), p.to(dev, non_blocking=True))
        idx = cells.numpy()
        if want_genes:
            pred[idx] = lm.float().cpu().numpy()
        if want_emb:
            emb[idx] = feats.float().cpu().numpy()
        done += len(idx)
        if done % 20000 < len(idx):
            print(f"    {done:>8,}/{n:,} cells  ({done/max(time.time()-t0,1e-9):.0f} cell/s)", flush=True)
    return pred, emb


def main():
    ap = argparse.ArgumentParser(description="Predict single-cell expression and embeddings from H&E.")
    src = ap.add_argument_group("input (either --prepared, or --image with --cells)")
    src.add_argument("--image", help="H&E slide (SVS/NDPI/TIFF/PNG)")
    src.add_argument("--cells", help="cells.npz from predict/segment.py, or your own")
    src.add_argument("--prepared", help="a prepared slide directory (see predict/inputs.py)")
    src.add_argument("--mpp", type=float, default=None,
                     help=f"microns per pixel of --image; the crop is rescaled to the ~{FOV_UM:.1f} um the model "
                          f"was trained on. Read from slide metadata when omitted; ignored for --prepared.")

    w = ap.add_argument_group("weights")
    w.add_argument("--release", default=None,
                   help="a released model directory (stage1/stage2.safetensors + config.json + genes.tsv)")
    w.add_argument("--stage1", default=None, help="Stage-1 weights, .safetensors or .pt")
    w.add_argument("--stage2", default=None, help="Stage-2 weights, .safetensors or .pt")
    w.add_argument("--genes", default=None, help="TSV of the head's gene symbols; taken from --release if absent")

    o = ap.add_argument_group("output")
    o.add_argument("--out", required=True, help="output .h5ad")
    o.add_argument("--save_embeddings", action="store_true",
                   help="also store the 1536-d per-cell feature in obsm['X_voice'] (use this for downstream tasks)")
    o.add_argument("--embeddings_only", action="store_true", help="skip the gene head entirely")
    o.add_argument("--tile", type=int, default=256, help="neighbourhood tile in slide pixels")
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--device", default="cuda")
    a = ap.parse_args()

    want_emb = a.save_embeddings or a.embeddings_only
    want_genes = not a.embeddings_only

    if a.release or (a.stage1 and a.stage2):
        s1, s2 = a.stage1, a.stage2
    else:
        ck_dir = _p.ckpt_dir()
        s1 = os.path.join(ck_dir, "clip_lora_v2_final.pt")
        s2 = os.path.join(ck_dir, "se2_lora_p2v2_noinslide_epoch0.pt")
        for f in (s1, s2):
            if not os.path.exists(f):
                raise SystemExit(f"weights not found: {f}\n  Pass --release <dir> for a released model.")

    source = open_slide(a.image, a.cells, a.prepared, a.mpp)
    print(f"[slide] {source.describe()}", flush=True)
    if source.kind == "wsi" and source.mpp is None:
        print("[warn]  no resolution found. If the slide is not at ~0.2125 um/px, pass --mpp -- the crops would "
              "otherwise cover the wrong amount of tissue and predictions degrade silently.", flush=True)

    dev = torch.device(a.device)
    from voice.release import load_release
    model, se2, rcfg = load_release(a.release, device=dev, stage1=s1, stage2=s2)
    n_genes = int(rcfg["n_genes"])
    name = rcfg.get("_class_name", "VOICE") if a.release else f"{os.path.basename(s1)} + {os.path.basename(s2)}"
    print(f"[model] {name} | SE2 d{rcfg.get('d_model')}/L{rcfg.get('n_layers')} | {n_genes} genes"
          + ("  (gene head skipped)" if a.embeddings_only else ""), flush=True)

    tiles = tile(source.pos, a.tile)
    print(f"[run]   {len(source):,} cells in {len(tiles):,} tiles of {a.tile} px", flush=True)
    t0 = time.time()
    pred, emb = run(model, se2, source, tiles, n_genes, a.workers, dev, want_genes, want_emb)
    print(f"[run]   done in {time.time()-t0:.0f}s", flush=True)

    import anndata as ad, pandas as pd
    from voice.release import gene_names
    if want_genes:
        names = gene_names(a.release, n_genes) if a.release else None
        gf = a.genes
        if names is None and gf and os.path.exists(gf):
            g = pd.read_csv(gf, sep="\t")
            col = "gene_symbol" if "gene_symbol" in g.columns else g.columns[-1]
            if "global_gene_index" in g.columns:
                g = g.sort_values("global_gene_index")
            names = g[col].astype(str).tolist()[:n_genes]
        if names is None:
            names = [f"gene_{i}" for i in range(n_genes)]
            print("[warn]  no gene table: var names are positional indices into the model head.", flush=True)
        X, var = pred, pd.DataFrame(index=names)
    else:
        X, var = np.zeros((len(source), 0), np.float32), pd.DataFrame(index=[])

    obs = pd.DataFrame(dict(y_pixel=source.pos[:, 0], x_pixel=source.pos[:, 1]),
                       index=pd.Index(source.ids, name="cell_id"))
    A = ad.AnnData(X=X, obs=obs, var=var)
    A.obsm["spatial"] = source.pos[:, ::-1].copy()          # (x, y), the scanpy convention
    if want_emb:
        A.obsm["X_voice"] = emb
    A.uns["voice"] = dict(model=name, source=source.kind, crop_px=float(source.crop_px),
                          mpp=float(source.mpp) if source.mpp else None, tile=int(a.tile),
                          layer="log1p(mu), direct branch only (no retrieval, no Stage-3 gate)",
                          embedding="obsm['X_voice'] = 1536-d cell feature" if want_emb else None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    A.write_h5ad(a.out)
    parts = [f"X = log1p(mu) [{A.n_obs:,} x {A.n_vars:,}]"] if want_genes else [f"{A.n_obs:,} cells"]
    if want_emb:
        parts.append(f"obsm['X_voice'] [{A.n_obs:,} x 1536]")
    print(f"[out]   {a.out}  " + " · ".join(parts), flush=True)


if __name__ == "__main__":
    main()
