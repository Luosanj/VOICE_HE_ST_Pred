"""Load paired expression and predictions. Input: NPZ or aligned NPY/CSR/CSV files. Output: Y, P, genes, coordinates, cells."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse


def pair_arguments(parser):
    parser.add_argument('--data', help='NPZ: Ylog, pred (or Spred), genes, pos; positions in source pixels')
    parser.add_argument('--predictions', help='NPY prediction matrix; already log1p or fused scores')
    parser.add_argument('--expression', help='scipy sparse NPZ raw expression')
    parser.add_argument('--gene_alignment', help='CSV: gene, prediction_column, truth_column')
    parser.add_argument('--cells', help='CSV: expr_row, cell_type, x_pixel, y_pixel; prediction row order')
    parser.add_argument('--positions', help='optional NPY (y,x) positions for row verification')
    parser.add_argument('--gene_list', help='canonical TSV with gene, hvg_rank, svg_rank')
    parser.add_argument('--mpp', type=float, required=True)
    parser.add_argument('--out', required=True)


def load_pair(args):
    cells = pd.read_csv(args.cells) if args.cells else None
    if args.data:
        with np.load(args.data, allow_pickle=True) as z:
            Y = np.asarray(z['Ylog'], np.float32)
            P = np.asarray(z['pred'] if 'pred' in z else z['Spred'], np.float32)
            if args.predictions:
                P = np.asarray(np.load(args.predictions, mmap_mode='r'), np.float32)
            genes = z['genes'].astype(str)
            pos = np.asarray(z['pos'], np.float32)
            if 'expr_row' in z and cells is not None:
                if not np.array_equal(z['expr_row'], cells.expr_row):
                    raise ValueError('Cell row order differs')
    else:
        if not all([args.predictions, args.expression, args.gene_alignment, args.cells]):
            raise ValueError('Supply --data, or --predictions/--expression/--gene_alignment/--cells')
        g = pd.read_csv(args.gene_alignment)
        raw = sparse.load_npz(args.expression).tocsr()[cells.expr_row.to_numpy()]
        Y = np.log1p(raw[:, g.truth_column.to_numpy()].toarray().astype(np.float64))
        prediction = np.load(args.predictions, mmap_mode='r')
        P = np.asarray(prediction[:, g.prediction_column.to_numpy()], np.float32)
        genes = g.gene.astype(str).to_numpy()
        pos = cells[['y_pixel', 'x_pixel']].to_numpy(np.float32)
    if P.shape != Y.shape or P.shape != (len(pos), len(genes)):
        raise ValueError('Expression, predictions, genes, and cells must be aligned')
    if len(set(genes)) != len(genes) or not np.isfinite(Y).all() or not np.isfinite(P).all():
        raise ValueError('Expected unique genes and finite matrices')
    if cells is not None and (len(cells) != len(pos) or not np.allclose(pos, cells[['y_pixel','x_pixel']], atol=.002, rtol=0)):
        raise ValueError('Annotation and prediction positions differ')
    if args.positions and not np.allclose(np.load(args.positions), pos, atol=.002, rtol=0):
        raise ValueError('Prediction positions differ')
    if args.gene_list:
        canonical = pd.read_csv(args.gene_list, sep='\t').gene.astype(str)
        keep = np.isin(genes, canonical)
        Y, P, genes = Y[:, keep], P[:, keep], genes[keep]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return Y, P, genes, pos, cells
