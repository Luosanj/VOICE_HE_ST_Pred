"""Select RNA genes and map panels to model outputs. Input: gene names, global-gene TSV, and optional vendor HDF5. Output: gene masks and indices."""
from __future__ import annotations
import numpy as np

CONTROL_PREFIXES = ("negcontrol", "blank", "antisense", "deprecatedcodeword",
                    "unassignedcodeword", "intergenic", "genomic", "codeword")
_HDR = {"gene", "gene_symbol", "symbol", "name", "feature", "feature_name"}


def is_control(g) -> bool:
    return str(g).lower().startswith(CONTROL_PREFIXES)


def read_genes(path):
    """One gene symbol per line (first tab-separated field); a header line is detected and skipped."""
    lines = [l.strip().split("\t")[0] for l in open(path) if l.strip()]
    return lines[1:] if lines and lines[0].lower() in _HDR else lines


def protein_names(h5_path) -> set:
    """Names carrying at least one TXP (antibody) feature in a 10x cell_feature_matrix.h5. Empty if no file."""
    if not h5_path:
        return set()
    import os
    if not os.path.exists(h5_path):
        return set()
    import h5py
    with h5py.File(h5_path, "r") as f:
        ff = f["matrix/features"]
        ids = [x.decode() for x in ff["id"][:]]
        nms = [x.decode() for x in ff["name"][:]]
    return {n for i, n in zip(ids, nms) if i.startswith("TXP")}


def real_gene_mask(genes, h5_path=None):
    """Boolean mask over `genes`: True for a real RNA gene, False for a control probe or a protein channel."""
    prot = protein_names(h5_path)
    by_name = np.array([not is_control(g) for g in genes])
    mask = by_name & np.array([g not in prot for g in genes])
    if not prot:
        assert (mask == by_name).all(), "no h5 given: the mask must equal the name-only one"
    return mask, int((by_name & ~mask).sum())


def sym2glob_map(global_genes_tsv):
    """gene symbol -> global gene index, from the head's gene table (columns gene_symbol, global_gene_index)."""
    import pandas as pd
    gg = pd.read_csv(global_genes_tsv, sep="\t")
    return dict(zip(gg["gene_symbol"].astype(str), gg["global_gene_index"].astype(int))), len(gg)
