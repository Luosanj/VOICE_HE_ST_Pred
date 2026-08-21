"""Where the model's input crops come from. Two sources, one interface.

VOICE needs three things per cell: a crop centred on it, a mask saying which pixels of that crop are the cell,
and the cell's slide coordinates. Those can be produced two ways, and downstream code should not care which:

**A whole-slide image plus segmentations** (`WSISlide`). Crops are cut from the image on the fly. This is the
path for a slide you just scanned: run `predict/segment.py`, then predict. The crop side is derived from your
resolution so the field of view matches training.

**An already-prepared slide** (`PreparedSlide`). Crops were written to disk ahead of time, one PNG per cell,
alongside the polygons in crop-local coordinates. This is the layout the training and benchmark corpora use:

    <slide>/
        patch_cell_boundaries.npz    expr_rows, indptr, vertex_x_patch, vertex_y_patch, x_pixel, y_pixel,
                                     output_size, crop_px
        manifest.csv.gz              expr_row -> patch_path
        patches/                     the crops
        expression.npz, genes.tsv    optional; only needed to SCORE a prediction

Prepared slides need no resolution argument: the crop was already cut at the right physical size, and the
polygon coordinates are already in the crop's frame. That makes this path exactly reproducible -- the same
bytes go into the encoder every time -- which is why the benchmark uses it.

Both classes expose the same three methods, so `predict/predict.py` and `benchmark/` treat them identically:

    len(src)                 number of cells
    src.pos                  [N,2] float32, (y, x) in slide pixels
    src.ids                  [N] str
    src.batch(idx)           (uint8 [n,3,224,224], float32 [n,16,16]) for the cells at `idx`
"""
from __future__ import annotations
import os
import numpy as np

from predict.geometry import polygon_mask, crop_px_for, OUTPUT_SIZE, TOKEN_GRID


class WSISlide:
    """Crops cut on demand from a whole-slide image, using segmentations from a cells.npz."""

    kind = "wsi"

    def __init__(self, image, cells, mpp=None):
        from predict.slide_io import SlideReader
        self.image = str(image)
        z = np.load(cells, allow_pickle=True)
        self.pos = np.stack([z["y_pixel"], z["x_pixel"]], 1).astype(np.float32)
        self.poly = ((z["indptr"].astype(np.int64), z["vertex_x"].astype(np.float32),
                      z["vertex_y"].astype(np.float32))
                     if "indptr" in z.files and "vertex_x" in z.files else None)
        self.ids = (z["cell_id"].astype(str) if "cell_id" in z.files
                    else np.array([f"cell_{i}" for i in range(len(self.pos))]))
        rd = SlideReader(self.image)
        self.mpp = mpp if mpp is not None else rd.mpp
        self.width, self.height, self.backend = rd.width, rd.height, rd.backend
        rd.close()
        self.crop_px = crop_px_for(self.mpp)
        self._rd = None

    def __len__(self): return len(self.pos)

    def describe(self):
        fov = self.crop_px * (self.mpp or 0.2125)
        return (f"{os.path.basename(self.image)} {self.width}x{self.height} via {self.backend} | "
                f"mpp={self.mpp if self.mpp else 'unknown'} -> crop {self.crop_px:.1f} px = {fov:.1f} um | "
                f"polygons: {'yes' if self.poly else 'NO (centre-token mask)'}")

    def _reader(self):
        if self._rd is None:                      # opened per worker: slide handles are not fork-safe
            from predict.slide_io import SlideReader
            self._rd = SlideReader(self.image)
        return self._rd

    def batch(self, idx):
        idx = np.asarray(idx)
        cy = self.pos[idx, 0]; cx = self.pos[idx, 1]
        imgs = self._reader().crops(cx, cy, self.crop_px, OUTPUT_SIZE)
        W = np.empty((len(idx), TOKEN_GRID, TOKEN_GRID), np.float32)
        half = self.crop_px / 2.0
        s = OUTPUT_SIZE / self.crop_px            # slide px -> crop frame
        for j, c in enumerate(idx):
            if self.poly is None:
                W[j] = polygon_mask(np.empty(0), np.empty(0))
            else:
                ip, vx, vy = self.poly
                a, b = int(ip[c]), int(ip[c + 1])
                W[j] = polygon_mask((vx[a:b] - (cx[j] - half)) * s, (vy[a:b] - (cy[j] - half)) * s)
        return imgs, W


class PreparedSlide:
    """Crops read from a pre-built slide directory (the training / benchmark layout)."""

    kind = "prepared"

    def __init__(self, directory, mpp=None):
        self.dir = str(directory)
        bz = np.load(os.path.join(self.dir, "patch_cell_boundaries.npz"), allow_pickle=True)
        self.er = bz["expr_rows"].astype(np.int64)
        self.ip = bz["indptr"].astype(np.int64)
        self.vx = bz["vertex_x_patch"].astype(np.float32)
        self.vy = bz["vertex_y_patch"].astype(np.float32)
        self.output_size = int(bz["output_size"])
        self.crop_px = float(bz["crop_px"]) if "crop_px" in bz.files else float(self.output_size)
        self.pos = np.stack([bz["y_pixel"], bz["x_pixel"]], 1).astype(np.float32)
        self.mpp = mpp

        import pandas as pd
        man = (pd.read_csv(os.path.join(self.dir, "manifest.csv.gz"), usecols=["expr_row", "patch_path"])
                 .drop_duplicates("expr_row").set_index("expr_row")["patch_path"])
        self.paths = [os.path.join(self.dir, str(man.loc[int(e)])) for e in self.er]
        self.ids = np.array([str(e) for e in self.er])
        if len(self.paths) != len(self.pos):
            raise ValueError(f"{self.dir}: manifest has {len(self.paths)} crops but boundaries have {len(self.pos)} cells")

    def __len__(self): return len(self.pos)

    def describe(self):
        return (f"{os.path.basename(self.dir)} | {len(self)} pre-cut crops at {self.output_size} px "
                f"(source crop {self.crop_px:.0f} px) | polygons: yes")

    def batch(self, idx):
        from PIL import Image
        idx = np.asarray(idx)
        imgs = np.empty((len(idx), 3, OUTPUT_SIZE, OUTPUT_SIZE), np.uint8)
        W = np.empty((len(idx), TOKEN_GRID, TOKEN_GRID), np.float32)
        for j, c in enumerate(idx):
            im = Image.open(self.paths[c]).convert("RGB")
            if im.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                im = im.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
            imgs[j] = np.ascontiguousarray(np.asarray(im, np.uint8).transpose(2, 0, 1))
            a, b = int(self.ip[c]), int(self.ip[c + 1])
            # vertices are already in the crop frame; rescale only if the stored crop is not 224 px
            sx = OUTPUT_SIZE / self.output_size
            W[j] = polygon_mask(self.vx[a:b] * sx, self.vy[a:b] * sx)
        return imgs, W

    def expression(self):
        """(Y_log1p [N, n_real], gene symbols) for scoring. None if the slide has no expression."""
        e = os.path.join(self.dir, "expression.npz")
        g = os.path.join(self.dir, "genes.tsv")
        if not (os.path.exists(e) and os.path.exists(g)):
            return None, None
        from scipy import sparse
        from voice.panel import read_genes, real_gene_mask
        genes = read_genes(g)
        X = sparse.load_npz(e).tocsr()
        h5 = os.path.join(self.dir, "features.h5")
        mask, _n = real_gene_mask(genes, h5 if os.path.exists(h5) else None)
        cols = np.where(mask)[0]
        return np.log1p(np.asarray(X[:, cols].todense(), np.float32)), [genes[i] for i in cols]


def open_slide(image=None, cells=None, prepared=None, mpp=None):
    """Pick the source from whichever arguments were given."""
    if prepared:
        return PreparedSlide(prepared, mpp)
    if image and cells:
        return WSISlide(image, cells, mpp)
    raise ValueError("give either --prepared <dir>, or both --image and --cells")
