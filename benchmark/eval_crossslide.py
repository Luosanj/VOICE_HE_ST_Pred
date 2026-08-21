#!/usr/bin/env python
"""Cross-slide evaluation: apply a trained model to slides it has never seen, and score it.

    python benchmark/eval_crossslide.py --slides slides.yaml --gene_lists gene_lists/ --out results.csv

Nothing on the test slide is fitted -- no fold, no refit, no calibration. Each slide's cells are grouped into
non-overlapping 256-px tiles so every cell is predicted exactly once with its neighbours as context, and scored
against its own measured expression.

A panel gene the model's head cannot emit takes a correlation of 0 and stays in the denominator, so the number
is comparable between methods with different output spaces.

`--gene_lists` is optional. With it, HVG/SVG use the canonical model-independent lists from
benchmark/gene_lists.py -- required if you are comparing several methods. Without it the rankings are derived
from the slide's own measured expression, which is fine for a single model.
"""
from __future__ import annotations
import os, sys, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd

from benchmark.dataset import load_slides
from benchmark.runner import load_model, predict_slide, gene_orders
from voice.metrics_bench import score_against_head, rank_hvg_svg
from voice.panel import sym2glob_map

MET = ["ALL", "HVG20", "HVG50", "HVG100", "SVG20", "SVG50", "SVG100"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True)
    ap.add_argument("--global_genes", required=True,
                    help="TSV (gene_symbol, global_gene_index) for the head -- released with the weights")
    ap.add_argument("--gene_lists", default=None, help="dir from benchmark/gene_lists.py")
    ap.add_argument("--out", default="results_crossslide.csv")
    ap.add_argument("--stage1", default=None); ap.add_argument("--stage2", default=None)
    ap.add_argument("--save_preds", default=None, help="dir to write per-slide predictions for figures / retrieval")
    ap.add_argument("--tile", type=int, default=256); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sym2glob, _n = sym2glob_map(a.global_genes)
    model, se2, n_genes, meta = load_model(a.stage1, a.stage2, a.device)
    print(f"[model] {meta['stage1']} + {meta['stage2']} | SE2 d{meta['d_model']}/L{meta['n_layers']} "
          f"| {n_genes} genes", flush=True)

    rows = []
    for sl in load_slides(a.slides):
        t0 = time.time()
        pos, poly, ids, Y, genes, n_prot = sl.load()
        gid = np.array([sym2glob.get(g, -1) for g in genes], np.int64)
        pred, feats = predict_slide(model, se2, sl, pos, poly, n_genes, sl.mpp, a.tile, a.workers,
                                    a.device, return_features=bool(a.save_preds))
        if a.gene_lists:
            f = os.path.join(a.gene_lists, f"{sl.name}.tsv")
            if not os.path.exists(f):
                sys.exit(f"{sl.name}: no canonical gene list at {f}. Run benchmark/gene_lists.py first, or drop "
                         f"--gene_lists to rank from this slide's own expression.")
            hvg, svg = gene_orders(f, genes)
        else:
            _v, _m, hr, sr = rank_hvg_svg(Y, pos)
            hvg, svg = np.argsort(hr), np.argsort(sr)
        r, pcc = score_against_head(pred, Y, gid, pos, hvg, svg)
        r.update(slide=sl.name, n_cells=int(len(pos)), n_protein_dropped=n_prot,
                 stage1=meta["stage1"], stage2=meta["stage2"])
        rows.append(r)
        print(f"  [{sl.name[:34]:34s}] N={len(pos):>7,} real={r['n_genes']:>5} cov={r['n_covered']:>5} "
              + " ".join(f"{m}={r[m]:.4f}" for m in ("ALL", "HVG50", "SVG50"))
              + f" ({time.time()-t0:.0f}s)", flush=True)
        if a.save_preds:
            os.makedirs(a.save_preds, exist_ok=True)
            np.savez_compressed(os.path.join(a.save_preds, f"{sl.name}.npz"),
                                pred=pred.astype(np.float16), Ylog=Y.astype(np.float16), pos=pos, gid=gid,
                                genes=np.array(genes, dtype=object), pcc=pcc, cell_id=ids)
            np.save(os.path.join(a.save_preds, f"{sl.name}_features.npy"), feats)

    df = pd.DataFrame(rows)[["slide", "n_cells", "n_genes", "n_covered", "n_protein_dropped"] + MET +
                            ["stage1", "stage2"]]
    df.to_csv(a.out, index=False)
    print(f"\n  {'slide':34s}" + "".join(f"{m:>9s}" for m in MET))
    for _, r in df.iterrows():
        print(f"  {r['slide'][:34]:34s}" + "".join(f"{r[m]:9.4f}" for m in MET))
    if len(df) > 1:
        print(f"  {'MACRO':34s}" + "".join(f"{df[m].mean():9.4f}" for m in MET))
    print(f"\n[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
