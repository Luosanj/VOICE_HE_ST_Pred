"""Spatial-patch dataset over the global Xenium cache.

Per slide the cache already has central-mask FM features (xenium_fm_database/uni2/.../features.npz:
feats[N,1536], x_pixel, y_pixel, expr_rows) and raw counts (cell_expression_global/.../expr_csr.npz,
global 11251-gene space). We tile cells into 256px spatial patches by (x,y) — same as his dataset —
and yield per-patch {features, pos, y_counts (raw, panel genes), panel global-ids}. Held-out-slide
regime: whole slides held out (retrieval/eval exclude them).

Metric protocol matches his eval_utils: y = log1p(RAW counts) (no library normalization).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import Dataset

from voice.genes import GeneSpace          # reuse panel<->global gene mapping only (data plumbing)


class GlobalSpatialDataset(Dataset):
    def __init__(self, cfg, slides, gene_space: GeneSpace, mode="train",
                 patch_size=256, overlap=30, max_cells=200, min_cells=2, seed=0):
        self.gs = gene_space
        self.patch_size = patch_size
        self.slides = slides                  # list of (tissue, slide)
        self.slide_feats = []                 # per-slide feats [n,1536]
        self.slide_pos = []                   # per-slide [n,2] (y,x) pixels
        self.slide_ycsr = []                  # per-slide raw-count CSR over panel genes [n,Gp]
        self.slide_panel = []                 # per-slide panel global ids [Gp]
        self.patches = []                     # (slide_idx, cell_idx array)
        he = cfg.paths.he_emb_dir; ex = cfg.paths.expr_dir
        rng = np.random.RandomState(seed)
        for si, (t, s) in enumerate(slides):
            z = np.load(os.path.join(he, t, s, "features.npz"), allow_pickle=True)
            feats = z["feats"].astype(np.float32)
            pos = np.stack([z["y_pixel"], z["x_pixel"]], 1).astype(np.float32)   # (y,x) like his
            er = z["expr_rows"].astype(np.int64)
            csr = sparse.load_npz(os.path.join(ex, t, s, "expr_csr.npz")).tocsr()[er]   # raw counts, global
            panel_id = pd.read_parquet(os.path.join(ex, t, s, "cells.parquet"),
                                       columns=["gene_panel_id"])["gene_panel_id"].iloc[0]
            panel = gene_space.panel_global_ids(panel_id)                         # global ids w/ gene_emb
            y_panel = csr[:, panel].tocsr().astype(np.float32)                    # [n, Gp] raw counts
            self.slide_feats.append(feats); self.slide_pos.append(pos)
            self.slide_ycsr.append(y_panel); self.slide_panel.append(panel)
            # tile by (y,x)
            cy, cx = pos[:, 0], pos[:, 1]
            H = cy.max() + patch_size; W = cx.max() + patch_size
            ys = np.arange(0, H, patch_size - overlap); xs = np.arange(0, W, patch_size - overlap)
            for y0 in ys:
                rm = (cy >= y0) & (cy < y0 + patch_size)
                if not rm.any():
                    continue
                idxr = np.nonzero(rm)[0]; cxr = cx[rm]
                for x0 in xs:
                    cm = (cxr >= x0) & (cxr < x0 + patch_size)
                    if cm.sum() < min_cells:
                        continue
                    cells = idxr[cm]
                    if len(cells) > max_cells:
                        cells = rng.choice(cells, max_cells, replace=False)
                    self.patches.append((si, np.sort(cells)))
        self.mode = mode
        print(f"  [{mode}] {len(slides)} slides, {sum(f.shape[0] for f in self.slide_feats)} cells, "
              f"{len(self.patches)} patches")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        si, cells = self.patches[i]
        feats = torch.from_numpy(self.slide_feats[si][cells])               # [n,1536]
        pos = torch.from_numpy(self.slide_pos[si][cells])                   # [n,2]
        y = torch.from_numpy(np.asarray(self.slide_ycsr[si][cells].todense(), dtype=np.float32))  # [n,Gp] raw
        panel = torch.from_numpy(self.slide_panel[si])                      # [Gp] global ids
        return {"features": feats, "pos": pos, "y_counts": y, "panel": panel,
                "slide": si, "cell_idx": torch.from_numpy(cells)}
