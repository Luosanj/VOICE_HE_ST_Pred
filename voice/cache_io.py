"""IO for (a) the source cache of embedded slides and (b) the precompute shards.

Source cache per slide:
  he_emb : <he_emb_dir>/<tissue>/<slide>/features.npz   key 'feats' [N,1536], 'expr_rows', 'entity_ids'
  expr   : <expr_dir>/<tissue>/<slide>/expr_csr.npz      CSR [N,11251] raw counts (global gene space)
           <expr_dir>/<tissue>/<slide>/cells.parquet     has gene_panel_id
Alignment: feats row i corresponds to expr CSR row expr_rows[i]  (we align everything to feats order).

Precompute shard = one slide -> a directory of per-field .npy (memmap-friendly) + meta.json + DONE.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy import sparse


# ----------------------------- source cache -----------------------------
class SourceCache:
    def __init__(self, cfg):
        self.he_dir = cfg.paths.he_emb_dir
        self.expr_dir = cfg.paths.expr_dir
        self.manifest = pd.read_csv(cfg.paths.expr_manifest)

    def list_slides(self):
        """[(tissue, slide, panel_id, n_cells), ...] for slides present in BOTH modalities."""
        out = []
        for _, r in self.manifest.iterrows():
            t, s = r["dataset_name"], r["sample_id"]
            if os.path.exists(os.path.join(self.he_dir, t, s, "features.npz")):
                out.append((t, s, r["gene_panel_id"], int(r["n_cells"])))
        return out

    def _normalize_log1p_csr(self, csr: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
        csr = csr.tocsr().astype(np.float32)
        rowsum = np.asarray(csr.sum(1)).ravel()
        per_nnz = np.repeat(rowsum, np.diff(csr.indptr))
        y = csr.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            y.data = np.log1p(csr.data / np.clip(per_nnz, 1e-12, None) * target_sum).astype(np.float32)
        return y

    def load_slide_aligned(self, tissue, slide, gene_space, target_sum, max_cells=None, seed=0):
        """Returns dict aligned to feats order:
           he_emb [n,Dhe] f32, y_global [n,11251] CSR (normalize-then-log1p), entity_ids [n],
           panel_id, panel_global_ids [Gp]."""
        z = np.load(os.path.join(self.he_dir, tissue, slide, "features.npz"), allow_pickle=True)
        feats = z["feats"].astype(np.float32)                       # [N, Dhe], feats order
        expr_rows = z["expr_rows"].astype(np.int64)
        entity = np.asarray(z["entity_ids"]).astype(str)
        xy = np.stack([z["x_pixel"], z["y_pixel"]], 1).astype(np.float32)   # feats order

        ex_dir = os.path.join(self.expr_dir, tissue, slide)
        csr = sparse.load_npz(os.path.join(ex_dir, "expr_csr.npz")).tocsr()
        cells = pd.read_parquet(os.path.join(ex_dir, "cells.parquet"), columns=["entity_id", "gene_panel_id"])
        panel_id = str(cells["gene_panel_id"].iloc[0])

        # align expr -> feats order, then (optionally) subsample
        csr = csr[expr_rows]
        n = feats.shape[0]
        if max_cells is not None and n > max_cells:
            rng = np.random.RandomState(seed)
            sel = np.sort(rng.choice(n, size=max_cells, replace=False))
            feats, csr, entity, xy = feats[sel], csr[sel], entity[sel], xy[sel]
        y_global = self._normalize_log1p_csr(csr, target_sum)
        return {
            "he_emb": feats,
            "y_global": y_global.tocsr(),
            "entity_ids": entity,
            "xy": xy,
            "panel_id": panel_id,
            "panel_global_ids": gene_space.panel_global_ids(panel_id),
        }

    def load_coords(self, tissue, slide, entity_ids):
        """Spatial x/y for the given cells (aligned to `entity_ids` order). Used for SVG (Moran's I)
        at eval time, read straight from the source features.npz so the precompute cache needn't store it."""
        z = np.load(os.path.join(self.he_dir, tissue, slide, "features.npz"), allow_pickle=True)
        pos = {e: i for i, e in enumerate(np.asarray(z["entity_ids"]).astype(str))}
        idx = np.array([pos[str(e)] for e in entity_ids], dtype=np.int64)
        return np.stack([z["x_pixel"][idx], z["y_pixel"][idx]], 1).astype(np.float32)


# ----------------------------- precompute shards -----------------------------
SHARD_FIELDS = ["entity_ids", "he_emb", "y", "R", "disp", "ref_mask",
                "sim_stats", "neighbor_stats", "meta", "s_own", "n_neighbors"]


def shard_dir(precompute_dir, tissue, slide):
    return os.path.join(precompute_dir, tissue, slide)


def shard_done(precompute_dir, tissue, slide) -> bool:
    return os.path.exists(os.path.join(shard_dir(precompute_dir, tissue, slide), "DONE"))


def write_shard(precompute_dir, tissue, slide, fields: dict, meta: dict):
    d = shard_dir(precompute_dir, tissue, slide)
    os.makedirs(d, exist_ok=True)
    for k in SHARD_FIELDS:
        np.save(os.path.join(d, f"{k}.npy"), fields[k])
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(d, "DONE"), "w").close()


class ShardReader:
    """Memmaps one precompute shard. Ragged gene dim (Gp) lives in meta -> batching is within-shard."""

    def __init__(self, d):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.panel_global_ids = np.array(self.meta["panel_global_ids"], dtype=np.int64)
        self.scf_ids = np.array(self.meta["scf_ids"], dtype=np.int64)
        self.tissue = self.meta["tissue"]; self.slide = self.meta["slide"]
        self.n = int(self.meta["n_cells"]); self.Gp = int(self.meta["Gp"])
        self._mm = {}

    def field(self, name):
        if name not in self._mm:
            self._mm[name] = np.load(os.path.join(self.dir, f"{name}.npy"), mmap_mode="r")
        return self._mm[name]

    def rows(self, name, idx):
        return np.asarray(self.field(name)[idx])
