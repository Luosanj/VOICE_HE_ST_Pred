"""Loading a trained VOICE model and running it over a benchmark slide.

Shared by both evaluations so they cannot drift apart, and identical to what `predict/predict.py` does -- the
numbers a user reproduces come from the same code path as the numbers they would get on their own slide.
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from voice import paths as _p
_p.hf_home()
import torch
from torch.utils.data import DataLoader

from voice.encoder import build_uni2, inject_lora, load_lora, pooled_feat, MEAN, STD
from voice.scale_train import ScaleHE2Cell
from predict.geometry import tile, crop_px_for
from predict.predict import TileDS
from predict.inputs import WSISlide


def load_model(stage1=None, stage2=None, device="cuda"):
    """Returns (uni2_with_lora, se2_head, n_genes, meta)."""
    ck_dir = _p.ckpt_dir() if not (stage1 and stage2) else None
    s1 = stage1 or os.path.join(ck_dir, "clip_lora_v2_final.pt")
    s2 = stage2 or os.path.join(ck_dir, "se2_lora_p2v2_noinslide_epoch0.pt")
    for f in (s1, s2):
        if not os.path.exists(f):
            raise SystemExit(f"checkpoint not found: {f}")
    dev = torch.device(device)
    ck = torch.load(s2, map_location=dev, weights_only=False)
    model = build_uni2(dev)
    inject_lora(model, ck["nblocks"], ck["r"], ck["alpha"], ck["dropout"]); model.to(dev)
    load_lora(model, torch.load(s1, map_location=dev, weights_only=False)["lora"], ck["lora"])
    model.eval()
    n_genes = int(ck["se2"]["head.mu_lin.weight"].shape[0])
    se2 = ScaleHE2Cell(n_genes, feat_dim=1536, d_model=ck["d_model"], n_layers=ck["n_layers"]).to(dev)
    se2.load_state_dict(ck["se2"]); se2.eval()
    meta = dict(stage1=os.path.basename(s1), stage2=os.path.basename(s2),
                d_model=int(ck["d_model"]), n_layers=int(ck["n_layers"]), n_genes=n_genes)
    return model, se2, n_genes, meta


@torch.no_grad()
def predict_slide(model, se2, slide, pos, poly, n_genes, mpp=None, tile_px=256, workers=8,
                  device="cuda", return_features=False):
    """[n_cells, n_genes] log1p(mu), plus the 1536-d pooled features when asked (needed for retrieval)."""
    dev = torch.device(device)
    src = WSISlide.__new__(WSISlide)                 # the benchmark already loaded cells; reuse them as-is
    src.image = slide.image; src.pos = pos; src.poly = poly
    src.mpp = mpp; src.crop_px = crop_px_for(mpp); src._rd = None
    tiles = tile(pos, tile_px)
    dl = DataLoader(TileDS(src, tiles), batch_size=1, num_workers=workers, collate_fn=lambda b: b[0],
                    pin_memory=True, prefetch_factor=4 if workers else None)
    pred = np.zeros((len(pos), n_genes), np.float32)
    feats_out = np.zeros((len(pos), 1536), np.float16) if return_features else None
    mean = MEAN.to(dev); std = STD.to(dev)
    for imgs, w, p, cells in dl:
        x = imgs.to(dev, non_blocking=True).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = pooled_feat(model, x, w.to(dev, non_blocking=True))
            lm, _ = se2(f.float(), p.to(dev, non_blocking=True))
        idx = cells.numpy()
        pred[idx] = lm.float().cpu().numpy()
        if return_features:
            feats_out[idx] = f.float().cpu().numpy().astype(np.float16)
    return (pred, feats_out) if return_features else pred


def gene_orders(gene_list_tsv, genes):
    """HVG/SVG orderings as positions into `genes`, from a canonical list built by benchmark/gene_lists.py."""
    import pandas as pd
    gl = pd.read_csv(gene_list_tsv, sep="\t")
    pos_of = {g: i for i, g in enumerate(genes)}
    hvg = np.array([pos_of[g] for g in gl.sort_values("hvg_rank").gene if g in pos_of], np.int64)
    svg = np.array([pos_of[g] for g in gl.sort_values("svg_rank").gene if g in pos_of], np.int64)
    return hvg, svg
