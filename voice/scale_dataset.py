"""Group cached cells into spatial patches. Input: cached features, coordinates, counts, and panels. Output: patch tensors for decoder training."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import Dataset

from voice.genes import GeneSpace


class GlobalSpatialDataset(Dataset):
    def __init__(self, cfg, slides, gene_space: GeneSpace, mode="train",
                 patch_size=256, overlap=30, max_cells=200, min_cells=2, seed=0):
        self.gs = gene_space
        self.patch_size = patch_size
        self.slides = slides
        self.slide_feats = []
        self.slide_pos = []
        self.slide_ycsr = []
        self.slide_panel = []
        self.patches = []
        he = cfg.paths.he_emb_dir; ex = cfg.paths.expr_dir
        rng = np.random.RandomState(seed)
        for si, (t, s) in enumerate(slides):
            z = np.load(os.path.join(he, t, s, "features.npz"), allow_pickle=True)
            feats = z["feats"].astype(np.float32)
            pos = np.stack([z["y_pixel"], z["x_pixel"]], 1).astype(np.float32)
            er = z["expr_rows"].astype(np.int64)
            csr = sparse.load_npz(os.path.join(ex, t, s, "expr_csr.npz")).tocsr()[er]
            panel_id = pd.read_parquet(os.path.join(ex, t, s, "cells.parquet"),
                                       columns=["gene_panel_id"])["gene_panel_id"].iloc[0]
            panel = gene_space.panel_global_ids(panel_id)
            y_panel = csr[:, panel].tocsr().astype(np.float32)
            self.slide_feats.append(feats); self.slide_pos.append(pos)
            self.slide_ycsr.append(y_panel); self.slide_panel.append(panel)

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
        feats = torch.from_numpy(self.slide_feats[si][cells])
        pos = torch.from_numpy(self.slide_pos[si][cells])
        y = torch.from_numpy(np.asarray(self.slide_ycsr[si][cells].todense(), dtype=np.float32))
        panel = torch.from_numpy(self.slide_panel[si])
        return {"features": feats, "pos": pos, "y_counts": y, "panel": panel,
                "slide": si, "cell_idx": torch.from_numpy(cells)}
