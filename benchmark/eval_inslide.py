#!/usr/bin/env python
"""In-slide evaluation: five-fold cross-validation within a single slide.

    python benchmark/eval_inslide.py --slides slides.yaml --global_genes genes.tsv --out results.csv

This is the protocol most prior work reports, so it is here for comparability -- but read what it measures. The
slide is cut into five contiguous vertical bands. For each fold, the gene head is fine-tuned on four bands and
predicts the fifth; the five held-out band predictions are pooled and per-gene correlation is computed once over
the whole slide. Bands, not random cells: neighbouring cells share tissue and a random split would put the same
structure on both sides.

The LoRA encoder is NOT trained here. The 1536-d features are extracted once per slide and only the SE(2)
decoder and NB head are fitted per fold, which is what makes five folds affordable.

**In-slide numbers are not zero-shot.** Part of the target slide is in training for every fold, so they are
systematically higher than the cross-slide numbers and the two must never be pooled. If you only report one,
report cross-slide.
"""
from __future__ import annotations
import os, sys, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from voice import paths as _p
_p.hf_home()
import torch
from torch.utils.data import DataLoader

from benchmark.dataset import load_slides
from benchmark.runner import load_model, gene_orders
from voice.encoder import pooled_feat, MEAN, STD
from voice.scale_train import ScaleHE2Cell, nb_nll
from voice.metrics_bench import score_against_head, rank_hvg_svg
from voice.panel import sym2glob_map
from predict.geometry import crop_px_for, tile
from predict.predict import TileDS

MET = ["ALL", "HVG20", "HVG50", "HVG100", "SVG20", "SVG50", "SVG100"]


@torch.no_grad()
def extract_features(model, slide, pos, poly, mpp, tile_px, workers, dev):
    """[N,1536] pooled features, extracted once. The encoder is frozen for the whole CV."""
    tiles = tile(pos, tile_px)
    from predict.inputs import WSISlide
    src = WSISlide.__new__(WSISlide)
    src.image = slide.image; src.pos = pos; src.poly = poly
    src.mpp = mpp; src.crop_px = crop_px_for(mpp); src._rd = None
    dl = DataLoader(TileDS(src, tiles), batch_size=1,
                    num_workers=workers, collate_fn=lambda b: b[0], pin_memory=True,
                    prefetch_factor=4 if workers else None)
    out = np.zeros((len(pos), 1536), np.float32)
    mean = MEAN.to(dev); std = STD.to(dev)
    for imgs, w, _p_, cells in dl:
        x = imgs.to(dev, non_blocking=True).float().div_(255.0); x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = pooled_feat(model, x, w.to(dev, non_blocking=True))
        out[cells.numpy()] = f.float().cpu().numpy()
    return out


def bands(pos, n_folds):
    """Contiguous vertical bands by x-quantile: fold i is the i-th band."""
    x = pos[:, 1]
    edges = np.quantile(x, np.linspace(0, 1, n_folds + 1))
    edges[0] -= 1; edges[-1] += 1
    return [np.where((x >= edges[i]) & (x < edges[i + 1]))[0] for i in range(n_folds)]


def fit_fold(feats, pos, Ytrue, gid, train_idx, n_genes, d_model, n_layers, init_sd,
             epochs, lr, batch, nb_weight, dev, tile_px=256):
    """Fine-tune the decoder + head on `train_idx`, warm-started from the released weights."""
    head = ScaleHE2Cell(n_genes, feat_dim=1536, d_model=d_model, n_layers=n_layers).to(dev)
    head.load_state_dict(init_sd)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    cov = gid >= 0
    cols = torch.from_numpy(gid[cov].astype(np.int64)).to(dev)
    Yt = torch.from_numpy(Ytrue[:, cov]).to(dev)
    Yraw = torch.expm1(Yt).clamp_min(0)
    tiles = [t for t in tile(pos, tile_px) if np.isin(t, train_idx).all()]
    F = torch.from_numpy(feats).to(dev); P = torch.from_numpy(pos).to(dev)
    head.train()
    for ep in range(epochs):
        order = np.random.RandomState(ep).permutation(len(tiles))
        for k in range(0, len(order), batch):
            loss = 0.0
            for t in order[k:k + batch]:
                c = torch.from_numpy(tiles[t]).to(dev)
                lm, nb = head(F[c], P[c])
                pred = lm[:, cols]
                loss = loss + torch.nn.functional.mse_loss(pred, Yt[c])
                if nb_weight and nb is not None:
                    loss = loss + nb_weight * nb_nll(Yraw[c], nb["mu"][:, cols], nb["log_theta"].exp()[:, cols])
            if isinstance(loss, float):
                continue
            loss = loss / max(1, len(order[k:k + batch]))
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True)
    ap.add_argument("--global_genes", required=True)
    ap.add_argument("--gene_lists", default=None)
    ap.add_argument("--out", default="results_inslide.csv")
    ap.add_argument("--stage1", default=None); ap.add_argument("--stage2", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8, help="tiles per optimiser step")
    ap.add_argument("--nb_weight", type=float, default=0.5, help="0 disables the NB term (MSE only)")
    ap.add_argument("--tile", type=int, default=256); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    sym2glob, _n = sym2glob_map(a.global_genes)
    model, se2, n_genes, meta = load_model(a.stage1, a.stage2, a.device)
    init_sd = {k: v.clone() for k, v in se2.state_dict().items()}
    print(f"[model] {meta['stage1']} + {meta['stage2']} | {a.folds}-fold, encoder frozen", flush=True)

    rows = []
    for sl in load_slides(a.slides):
        t0 = time.time()
        pos, poly, ids, Y, genes, n_prot = sl.load()
        gid = np.array([sym2glob.get(g, -1) for g in genes], np.int64)
        feats = extract_features(model, sl, pos, poly, sl.mpp, a.tile, a.workers, dev)
        print(f"  [{sl.name[:34]:34s}] features {feats.shape} ({time.time()-t0:.0f}s)", flush=True)
        pred = np.zeros((len(pos), n_genes), np.float32)
        for f, held in enumerate(bands(pos, a.folds)):
            train_idx = np.setdiff1d(np.arange(len(pos)), held)
            h = fit_fold(feats, pos, Y, gid, train_idx, n_genes, meta["d_model"], meta["n_layers"], init_sd,
                         a.epochs, a.lr, a.batch, a.nb_weight, dev, a.tile)
            with torch.no_grad():
                for t in tile(pos, a.tile):
                    t = t[np.isin(t, held)]
                    if not len(t):
                        continue
                    c = torch.from_numpy(t).to(dev)
                    lm, _ = h(torch.from_numpy(feats[t]).to(dev), torch.from_numpy(pos[t]).to(dev))
                    pred[t] = lm.float().cpu().numpy()
            print(f"      fold {f}: held {len(held):,} cells ({time.time()-t0:.0f}s)", flush=True)
            del h; torch.cuda.empty_cache()

        if a.gene_lists:
            hvg, svg = gene_orders(os.path.join(a.gene_lists, f"{sl.name}.tsv"), genes)
        else:
            _v, _m, hr, sr = rank_hvg_svg(Y, pos); hvg, svg = np.argsort(hr), np.argsort(sr)
        r, _pcc = score_against_head(pred, Y, gid, pos, hvg, svg)
        r.update(slide=sl.name, n_cells=int(len(pos)), n_protein_dropped=n_prot, folds=a.folds)
        rows.append(r)
        print(f"  [{sl.name[:34]:34s}] " + " ".join(f"{m}={r[m]:.4f}" for m in ("ALL", "HVG50", "SVG50"))
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)[["slide", "n_cells", "n_genes", "n_covered", "folds"] + MET]
    df.to_csv(a.out, index=False)
    print(f"\n  {'slide':34s}" + "".join(f"{m:>9s}" for m in MET))
    for _, r in df.iterrows():
        print(f"  {r['slide'][:34]:34s}" + "".join(f"{r[m]:9.4f}" for m in MET))
    if len(df) > 1:
        print(f"  {'MACRO':34s}" + "".join(f"{df[m].mean():9.4f}" for m in MET))
    print(f"\n[done] -> {a.out}\n[note] in-slide is NOT zero-shot; do not pool with cross-slide numbers.", flush=True)


if __name__ == "__main__":
    main()
