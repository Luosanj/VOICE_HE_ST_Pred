"""What a benchmark slide is, and how to point the evaluation at your own.

To score a prediction you need measured expression as well as images, so a benchmark slide is an H&E image
paired with a spatial transcriptomics assay on the same tissue. One slide is a directory:

    <slide>/
        image.(svs|ndpi|tif|png)     H&E
        cells.npz                    centroids + polygons, in the SAME pixel frame as the image
        expression.npz               scipy sparse CSR [n_cells, n_genes], raw counts, rows aligned to cells.npz
        genes.tsv                    one gene symbol per line, one per column of expression.npz
        features.h5                  optional: the vendor's cell_feature_matrix.h5, ONLY needed if the panel has
                                     antibody channels -- it is the only way to tell a protein channel from the
                                     RNA of the same name (see voice/panel.py)

`cells.npz` is exactly what `predict/segment.py` writes, so if your assay ships boundaries, convert them once and
both prediction and evaluation read the same file. Row i of `expression.npz` must be the same cell as row i of
`cells.npz` -- that alignment is asserted, not assumed.

Describe your slides in a small YAML and pass it with `--slides`:

    slides:
      - name: my_breast_slide
        dir:  /data/benchmark/my_breast_slide
        mpp:  0.25
      - name: my_lung_slide
        dir:  /data/benchmark/my_lung_slide
"""
from __future__ import annotations
import os
import numpy as np


IMAGE_EXT = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".png", ".jpg", ".jpeg")


class BenchmarkSlide:
    def __init__(self, name, directory, mpp=None):
        self.name = name; self.dir = str(directory); self.mpp = mpp
        self.image = self._find_image()
        self.cells = os.path.join(self.dir, "cells.npz")
        self.expression = os.path.join(self.dir, "expression.npz")
        self.genes_tsv = os.path.join(self.dir, "genes.tsv")
        h5 = os.path.join(self.dir, "features.h5")
        self.h5 = h5 if os.path.exists(h5) else None
        for f in (self.cells, self.expression, self.genes_tsv):
            if not os.path.exists(f):
                raise FileNotFoundError(f"{name}: missing {os.path.basename(f)} in {self.dir}\n"
                                        f"  See benchmark/dataset.py for the expected layout.")

    def _find_image(self):
        for f in sorted(os.listdir(self.dir)):
            if f.lower().endswith(IMAGE_EXT) and not f.startswith("."):
                return os.path.join(self.dir, f)
        raise FileNotFoundError(f"{self.name}: no H&E image in {self.dir} (looked for {IMAGE_EXT})")

    def load(self):
        """(pos [N,2] (y,x), poly, ids, Y_log1p [N,G_real], genes_real, n_protein_dropped)."""
        from scipy import sparse
        from voice.panel import read_genes, real_gene_mask
        z = np.load(self.cells, allow_pickle=True)
        pos = np.stack([z["y_pixel"], z["x_pixel"]], 1).astype(np.float32)
        poly = ((z["indptr"].astype(np.int64), z["vertex_x"].astype(np.float32), z["vertex_y"].astype(np.float32))
                if "indptr" in z.files and "vertex_x" in z.files else None)
        ids = z["cell_id"].astype(str) if "cell_id" in z.files else np.array([f"cell_{i}" for i in range(len(pos))])
        genes = read_genes(self.genes_tsv)
        X = sparse.load_npz(self.expression).tocsr()
        if X.shape[0] != len(pos):
            raise ValueError(f"{self.name}: expression has {X.shape[0]} rows but cells.npz has {len(pos)}. "
                             f"Row i of each must be the same cell.")
        if X.shape[1] != len(genes):
            raise ValueError(f"{self.name}: expression has {X.shape[1]} columns but genes.tsv lists {len(genes)}.")
        mask, n_prot = real_gene_mask(genes, self.h5)
        cols = np.where(mask)[0]
        Y = np.log1p(np.asarray(X[:, cols].todense(), np.float32))
        return pos, poly, ids, Y, [genes[i] for i in cols], n_prot


def load_slides(spec_path):
    """Read the --slides YAML into BenchmarkSlide objects."""
    import yaml
    spec = yaml.safe_load(open(spec_path))
    out = []
    for s in spec["slides"]:
        out.append(BenchmarkSlide(s["name"], s["dir"], s.get("mpp")))
    return out
