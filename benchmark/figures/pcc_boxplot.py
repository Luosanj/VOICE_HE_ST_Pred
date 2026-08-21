#!/usr/bin/env python
"""Per-gene PCC distributions, as box plots.

    python benchmark/figures/pcc_boxplot.py --preds VOICE=preds/slide.npz Baseline=other/slide.npz --out fig.png

One box per method, over the per-gene correlations on a slide. The distribution is the point: a mean hides
whether a method is uniformly mediocre or excellent on some genes and useless on others, which is exactly the
difference that matters for a gene panel.

With several `--preds` entries the genes are intersected first, so every method is summarised over the same set.
Only genes each method can actually emit are eligible; how many were dropped is printed, because a method with a
smaller output space would otherwise look better simply by being scored on fewer genes.

`--gene_set` restricts to the top-N HVG or SVG from a canonical list (benchmark/gene_lists.py).
"""
from __future__ import annotations
import os, sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import numpy as np


def load(path):
    z = np.load(path, allow_pickle=True)
    genes = [str(g) for g in z["genes"]]
    if "pcc" in z.files:
        pcc = np.asarray(z["pcc"], np.float64)
    else:
        from voice.metrics_bench import per_gene_pcc
        gid = z["gid"].astype(np.int64); Y = z["Ylog"]
        P = np.zeros_like(Y, np.float32); cov = gid >= 0
        P[:, cov] = z["pred"][:, gid[cov]]
        pcc = per_gene_pcc(P, Y)
    cov = z["gid"].astype(np.int64) >= 0
    return dict(zip(genes, pcc)), set(np.array(genes)[cov])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="LABEL=path.npz ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gene_list", default=None, help="canonical <slide>.tsv, needed for --gene_set")
    ap.add_argument("--gene_set", default="all", choices=["all", "hvg20", "hvg50", "hvg100",
                                                          "svg20", "svg50", "svg100"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, covered = {}, []
    for item in a.preds:
        if "=" not in item:
            sys.exit(f"--preds entries must be LABEL=path.npz, got {item}")
        lab, path = item.split("=", 1)
        d, cov = load(path)
        series[lab] = d; covered.append(cov)

    keep = set.intersection(*covered)
    if a.gene_set != "all":
        if not a.gene_list:
            sys.exit("--gene_set needs --gene_list (a <slide>.tsv from benchmark/gene_lists.py)")
        import pandas as pd
        gl = pd.read_csv(a.gene_list, sep="\t")
        col = "hvg_rank" if a.gene_set.startswith("hvg") else "svg_rank"
        n = int(a.gene_set[3:])
        ordered = [g for g in gl.sort_values(col).gene if g in keep]
        keep = set(ordered[:n])
    genes = sorted(keep)
    dropped = {lab: len(d) - len(genes) for lab, d in series.items()}
    print(f"[genes] scoring {len(genes)} genes shared by all methods "
          f"(dropped per method: {dropped})", flush=True)

    labels = list(series)
    data = [[series[l][g] for g in genes] for l in labels]
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(labels), 4.2))
    kw = dict(showfliers=False, widths=0.6, patch_artist=True, medianprops=dict(color="black", lw=1.4))
    try:                                     # renamed in Matplotlib 3.9
        bp = ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, **kw)
    cmap = plt.get_cmap("tab10")
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(cmap(i % 10)); b.set_alpha(0.65); b.set_linewidth(0.8)
    for i, vals in enumerate(data):                       # jittered points, so n is visible
        xj = np.random.RandomState(0).normal(i + 1, 0.055, len(vals))
        ax.plot(xj, vals, ".", ms=1.6, color="0.25", alpha=0.35, zorder=3)
    ax.set_ylabel("per-gene PCC", fontsize=11)
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_title(a.title or f"{a.gene_set.upper()} — {len(genes)} genes", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[out] {a.out}")
    for l, v in zip(labels, data):
        v = np.asarray(v)
        print(f"      {l:16s} median {np.median(v):.4f}  mean {v.mean():.4f}  n {len(v)}")


if __name__ == "__main__":
    main()
