#!/usr/bin/env python
"""Spatial heat maps: measured versus predicted expression of one gene, cell by cell.

    python benchmark/figures/spatial_heatmap.py --preds preds/my_slide.npz --gene ACTA2 --out fig.png

Reads a prediction file written by `benchmark/eval_crossslide.py --save_preds`.

Expression is shown as a **within-panel z-score**: each cell's value for the gene is standardised against that
gene's distribution over the slide, so measured counts and predicted log1p(mu) -- which live on different scales
-- can share a colour bar and be compared by eye. Colour limits are the same for both panels, taken from a
robust percentile range of the two together, so a difference in the picture is a difference in the prediction
and not in the scaling.

Cells are drawn as points at their centroids. `--zoom x0,y0,x1,y1` adds an inset over a region, matching the
enlarged panels in the paper's figures.
"""
from __future__ import annotations
import os, sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import numpy as np


def zscore(v):
    v = np.asarray(v, np.float64)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="npz from eval_crossslide.py --save_preds")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=None, help="optional H&E to show as a third panel")
    ap.add_argument("--zoom", default=None, help="x0,y0,x1,y1 in slide pixels -> inset row")
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=2.0, help="colour limits at +/- this many z units")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.preds, allow_pickle=True)
    genes = [str(g) for g in z["genes"]]
    if a.gene not in genes:
        sys.exit(f"{a.gene} is not in this panel. First few: {genes[:12]}")
    j = genes.index(a.gene)
    gid = z["gid"].astype(np.int64)
    if gid[j] < 0:
        sys.exit(f"{a.gene} is outside the model's gene head on this slide -- there is nothing to plot.")
    pos = z["pos"]; y, x = pos[:, 0], pos[:, 1]
    truth = zscore(z["Ylog"][:, j].astype(np.float32))
    pred = zscore(z["pred"][:, gid[j]].astype(np.float32))
    pcc = float(z["pcc"][j]) if "pcc" in z.files else float(np.corrcoef(truth, pred)[0, 1])

    panels = [("Measured", truth), ("VOICE", pred)]
    nrow = 2 if a.zoom else 1
    ncol = len(panels) + (1 if a.image else 0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    vlim = a.clip

    def draw(ax, title, vals, xlim=None, ylim=None):
        sc = ax.scatter(x, y, c=vals, s=a.point_size, cmap=a.cmap, vmin=-vlim, vmax=vlim,
                        linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        if xlim: ax.set_xlim(*xlim); ax.set_ylim(ylim[1], ylim[0])
        for s in ax.spines.values(): s.set_linewidth(0.5)
        return sc

    col = 0
    if a.image:
        from predict.slide_io import SlideReader
        rd = SlideReader(a.image)
        thumb = rd.region(int(x.min()), int(y.min()), int(max(x.max() - x.min(), y.max() - y.min())))
        rd.close()
        axes[0][0].imshow(thumb); axes[0][0].set_title("H&E", fontsize=11)
        axes[0][0].set_xticks([]); axes[0][0].set_yticks([])
        col = 1
    sc = None
    for i, (t, v) in enumerate(panels):
        sc = draw(axes[0][col + i], f"{t} — {a.gene}" + (f"  (PCC {pcc:.3f})" if i else ""), v)

    if a.zoom:
        x0, y0, x1, y1 = [float(v) for v in a.zoom.split(",")]
        if a.image: axes[1][0].axis("off")
        for i, (t, v) in enumerate(panels):
            draw(axes[1][col + i], f"{t} (zoom)", v, (x0, x1), (y0, y1))
        for ax in axes[0][col:]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.2))

    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("within-panel z-score", fontsize=9)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[out] {a.out}  {a.gene}: PCC {pcc:.4f} over {len(x):,} cells", flush=True)


if __name__ == "__main__":
    main()
