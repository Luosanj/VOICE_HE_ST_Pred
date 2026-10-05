#!/usr/bin/env python
"""Build canonical evaluation panels. Input: measured expression, cell coordinates, and optional vendor feature types. Output: HVG/SVG-ranked gene TSVs."""
from __future__ import annotations
import os, sys, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd

from experiments.benchmark.dataset import load_slides
from voice.metrics_bench import rank_hvg_svg
from voice.panel import sym2glob_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True, help="YAML describing your slides (see experiments/benchmark/dataset.py)")
    ap.add_argument("--out", required=True, help="output directory (must not already contain these files)")
    ap.add_argument("--global_genes", default=None,
                    help="TSV with columns gene_symbol, global_gene_index -- the model head's gene table. "
                         "Optional: without it the lists are still built, but `in_head` is left unset.")
    ap.add_argument("--knn", type=int, default=6, help="neighbours for Moran's I")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sym2glob = None
    if a.global_genes:
        sym2glob, n_global = sym2glob_map(a.global_genes)
        print(f"[head] {n_global} genes from {a.global_genes}", flush=True)

    rows = []
    for sl in load_slides(a.slides):
        f = os.path.join(a.out, f"{sl.name}.tsv")
        if os.path.exists(f):
            sys.exit(f"REFUSE overwrite: {f}")
        t0 = time.time()
        pos, _poly, _ids, Y, genes, n_prot = sl.load()
        var, mor, hvg_rank, svg_rank = rank_hvg_svg(Y, pos, a.knn)
        gid = (np.array([sym2glob.get(g, -1) for g in genes], np.int64) if sym2glob
               else np.full(len(genes), -1, np.int64))
        in_head = (gid >= 0).astype(int) if sym2glob else np.full(len(genes), -1, int)
        df = pd.DataFrame(dict(gene=genes, global_id=gid, in_head=in_head, variance=var,
                               hvg_rank=hvg_rank, moranI=mor, svg_rank=svg_rank))
        df.sort_values("hvg_rank").to_csv(f, sep="\t", index=False, float_format="%.6g")

        def cov(rank, n): return int(in_head[rank < n].sum()) if sym2glob else -1
        r = dict(slide=sl.name, n_cells=int(Y.shape[0]), n_real=len(genes), n_protein_dropped=n_prot,
                 n_in_head=int((gid >= 0).sum()) if sym2glob else -1,
                 HVG20_cov=cov(hvg_rank, 20), HVG50_cov=cov(hvg_rank, 50),
                 SVG20_cov=cov(svg_rank, 20), SVG50_cov=cov(svg_rank, 50))
        rows.append(r)
        print(f"  [{sl.name[:40]:40s}] N={r['n_cells']:>7,} real={r['n_real']:>5} "
              f"protein_dropped={n_prot:>3} in_head={r['n_in_head']:>5} ({time.time()-t0:.0f}s)", flush=True)
        if n_prot:
            print(f"      dropped {n_prot} antibody channels using {os.path.basename(sl.h5)}", flush=True)
        elif sl.h5 is None:
            print("      no features.h5: control probes removed by name only. If this panel has antibody "
                  "channels they are still in the list -- see voice/panel.py.", flush=True)

    smf = os.path.join(a.out, "SUMMARY.tsv")
    if os.path.exists(smf):
        sys.exit(f"REFUSE overwrite: {smf}")
    sm = pd.DataFrame(rows); sm.to_csv(smf, sep="\t", index=False)
    print(f"\n[done] {len(rows)} slides -> {a.out}/\n", flush=True)
    print(sm.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
