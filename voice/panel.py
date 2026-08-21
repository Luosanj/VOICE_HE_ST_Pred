"""What counts as a gene, and how a slide's panel maps onto the model's output head.

A Xenium/CosMx panel file lists more than genes. Two kinds of entry must be excluded before anything is scored,
and they need different evidence:

  * **Control probes** -- negative controls, blanks, antisense, unassigned and deprecated codewords. These are
    recognisable from the NAME, which is why `is_control` is a prefix test.

  * **Protein (antibody) channels** -- only present on protein add-on slides. These are NOT recognisable from
    the name: an anti-CD3E antibody channel is called "CD3E", exactly like the RNA. They can only be identified
    from the feature id in the vendor's `cell_feature_matrix.h5`, where RNA is `ENSG*`/`ENST*` and protein is
    `TXP*`. Leaving them in is not a rounding error: antibody staining has a far wider dynamic range than RNA
    counts, so protein channels dominate a variance ranking and take over the HVG list.

    A further trap: some names appear TWICE in the h5, once as RNA and once as protein (CD3E, CD4, CD8A, CD68,
    CD163, PCNA, PTEN, CD45RA, CD45RO). Preprocessing that deduplicates by name merges the two measurements into
    one column that can no longer be separated, so `protein_names` returns every name carrying a TXP feature and
    all of them are dropped -- including the ambiguous ones.
"""
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
