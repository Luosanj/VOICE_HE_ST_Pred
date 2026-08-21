"""Gene-space bookkeeping.

Three gene namespaces exist and must be kept aligned:
  - global : the 11251-gene union over all panels (expr CSR columns).        index = global_gene_index
  - scF    : scFoundation's 19264-gene vocabulary.                           index = scf_id
  - panel  : the genes a given Xenium panel actually measures (a subset).

Spec invariant §8.1: everything indexed by gene flows through gene_emb. A gene is "modelable"
only if it has a gene_emb, i.e. it exists in the foundation model's vocab. With
`universe='scf_vocab'` we restrict every gene set to global genes that map into scF; genes
without an embedding are dropped (and counted) rather than silently given a free parameter.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd


class GeneSpace:
    def __init__(self, global_genes_tsv, scf_gene_index_tsv, panels_dir, universe="scf_vocab"):
        g = pd.read_csv(global_genes_tsv, sep="\t").sort_values("global_gene_index")
        self.global_symbols = g["gene_symbol"].tolist()                 # global_idx -> symbol
        self.n_global = len(self.global_symbols)
        self.symbol_to_global = {s: i for i, s in enumerate(self.global_symbols)}

        s = pd.read_csv(scf_gene_index_tsv, sep="\t")                    # columns: gene_name, index
        self.scf_symbol_to_id = {sym: int(idx) for sym, idx in zip(s["gene_name"], s["index"])}
        self.n_scf = len(self.scf_symbol_to_id)

        # global -> scf id, or -1 when the symbol is absent from the foundation vocab
        self.global_to_scf = np.full(self.n_global, -1, dtype=np.int64)
        for i, sym in enumerate(self.global_symbols):
            self.global_to_scf[i] = self.scf_symbol_to_id.get(sym, -1)
        self.has_emb = self.global_to_scf >= 0                          # bool[n_global]

        self.universe = universe
        self.panels_dir = panels_dir
        self._panel_cache: dict[str, np.ndarray] = {}

    # ---- panel handling ----
    def panel_global_ids(self, panel_id) -> np.ndarray:
        """Global indices measured by `panel_id`, restricted to the modelable universe, sorted."""
        if panel_id in self._panel_cache:
            return self._panel_cache[panel_id]
        df = pd.read_csv(os.path.join(self.panels_dir, f"{panel_id}.tsv"), sep="\t")
        gids = [self.symbol_to_global[sym] for sym in df["gene_symbol"] if sym in self.symbol_to_global]
        gids = np.array(sorted(set(gids)), dtype=np.int64)
        if self.universe == "scf_vocab":
            gids = gids[self.has_emb[gids]]
        self._panel_cache[panel_id] = gids
        return gids

    def panel_coverage(self, panel_id) -> dict:
        """Diagnostics: how many panel genes survive the universe filter."""
        df = pd.read_csv(os.path.join(self.panels_dir, f"{panel_id}.tsv"), sep="\t")
        n_panel = df["gene_symbol"].nunique()
        in_global = sum(s in self.symbol_to_global for s in df["gene_symbol"].unique())
        modelable = len(self.panel_global_ids(panel_id))
        return {"panel_id": panel_id, "n_panel": int(n_panel), "in_global": int(in_global),
                "modelable": int(modelable), "dropped_no_emb": int(in_global - modelable)}

    # ---- mapping into the foundation vocab ----
    def scf_ids_for_globals(self, global_ids) -> np.ndarray:
        """scF vocab ids for the given global indices. All >=0 when universe='scf_vocab'."""
        return self.global_to_scf[np.asarray(global_ids)]

    def panel_bool_over_global(self, panel_id) -> np.ndarray:
        """bool[n_global]: True where the (universe-filtered) panel measures that global gene."""
        m = np.zeros(self.n_global, dtype=bool)
        m[self.panel_global_ids(panel_id)] = True
        return m
