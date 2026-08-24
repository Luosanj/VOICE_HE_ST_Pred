#!/usr/bin/env python
"""Build a retrieval bank from reference slides that have measured expression.

    python predict/build_bank.py --release weights/voice-23m --bank_dir bank/lung \\
        --prepared /data/ref/lung_a /data/ref/lung_b

A bank slide is one reference slide's cells: their embeddings, their measured log1p expression, their
coordinates, and the global ids of the genes the slide's panel measures. Retrieval then finds, for a query
cell, the bank cells that look like it, and averages what they were actually expressing (`voice/retrieval.py`).

**Reference slides must be the same tissue as the query and must not include the query slide.** A bank
containing the target is self-retrieval: it finds the cell itself and reports its own label, which looks
excellent and means nothing.

**The bank must be embedded with the weights the query will use.** Different weights put query and bank in
different spaces, and the neighbours stop corresponding to anything; the encoder identity is stored in each
bank file and checked at retrieval time.

Both input layouts work, the same as `predict/predict.py`:

    --prepared DIR [DIR ...]      prepared slides, each needing an expression.npz + genes.tsv alongside
    --image IMG --cells NPZ --expression NPZ --genes TSV      one slide from a whole-slide image

Size: one slide of N cells costs about `N x 1536 x 4` bytes of embeddings plus its expression, so a
half-million-cell slide is roughly 3 GB compressed. That is why the bank is written per slide and read back one
signature group at a time.
"""
from __future__ import annotations
import os, sys, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from voice import paths as _p
import torch

from voice.retrieval import save_bank_slide, list_bank, bank_path
from voice.panel import read_genes, real_gene_mask, sym2glob_map
from predict.geometry import tile
from predict.inputs import open_slide


def weights_id(release, stage1, stage2):
    """A short identifier for the encoder that produced an embedding, stored with every bank slide."""
    if release:
        import json
        p = os.path.join(release, "config.json")
        if os.path.exists(p):
            c = json.load(open(p))
            return f"{c.get('_class_name','VOICE')}-{c.get('_voice_version','?')}-g{c.get('n_genes','?')}"
        return os.path.basename(os.path.abspath(release))
    return f"{os.path.basename(stage1 or '?')}+{os.path.basename(stage2 or '?')}"


def slide_expression(directory, sym2glob):
    """(Y_log1p [N, G_real], panel global ids [G_real]) from expression.npz + genes.tsv in a slide directory."""
    e = os.path.join(directory, "expression.npz")
    g = os.path.join(directory, "genes.tsv")
    if not (os.path.exists(e) and os.path.exists(g)):
        raise SystemExit(f"{directory}: a bank slide needs measured expression — expected expression.npz and "
                         f"genes.tsv here. Reference slides without expression cannot be retrieved from.")
    from scipy import sparse
    genes = read_genes(g)
    X = sparse.load_npz(e).tocsr()
    h5 = os.path.join(directory, "features.h5")
    mask, n_prot = real_gene_mask(genes, h5 if os.path.exists(h5) else None)
    cols = np.where(mask)[0]
    names = [genes[i] for i in cols]
    gid = np.array([sym2glob.get(n, -1) for n in names], np.int64)
    keep = gid >= 0                                   # a gene outside the head cannot be a retrieval target
    Y = np.log1p(np.asarray(X[:, cols[keep]].todense(), np.float32))
    return Y, gid[keep], int(n_prot), int((~keep).sum())


def main():
    ap = argparse.ArgumentParser(description="Embed reference slides into a retrieval bank.")
    ap.add_argument("--bank_dir", required=True, help="output directory; one .bank.npz per reference slide")
    ap.add_argument("--prepared", nargs="*", default=[], help="prepared slide directories")
    ap.add_argument("--image", help="a single whole-slide image instead")
    ap.add_argument("--cells", help="its cells.npz")
    ap.add_argument("--expression", help="its expression.npz")
    ap.add_argument("--genes", help="its genes.tsv")
    ap.add_argument("--mpp", type=float, default=None)
    ap.add_argument("--release", default=None)
    ap.add_argument("--stage1", default=None); ap.add_argument("--stage2", default=None)
    ap.add_argument("--global_genes", required=True,
                    help="TSV (gene_symbol, global_gene_index) — the head's gene table, from the release")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="re-embed slides already in the bank")
    a = ap.parse_args()

    _p.hf_home()
    from voice.encoder import pooled_feat, MEAN, STD
    from voice.release import load_release
    from predict.predict import TileDS
    from torch.utils.data import DataLoader

    sym2glob, n_global = sym2glob_map(a.global_genes)
    wid = weights_id(a.release, a.stage1, a.stage2)
    dev = torch.device(a.device)
    model, _se2, rcfg = load_release(a.release, device=dev, stage1=a.stage1, stage2=a.stage2)
    print(f"[model] {wid} | head {n_global} genes", flush=True)

    jobs = [("prepared", d) for d in a.prepared]
    if a.image:
        jobs.append(("wsi", None))
    if not jobs:
        raise SystemExit("nothing to embed: pass --prepared DIR ... or --image/--cells")
    os.makedirs(a.bank_dir, exist_ok=True)

    mean = MEAN.to(dev); std = STD.to(dev)
    for kind, d in jobs:
        name = os.path.basename(os.path.abspath(d)) if kind == "prepared" else \
            os.path.splitext(os.path.basename(a.image))[0]
        if os.path.exists(bank_path(a.bank_dir, name)) and not a.force:
            print(f"  [{name}] already in the bank — skip (--force to redo)", flush=True)
            continue
        t0 = time.time()
        if kind == "prepared":
            src = open_slide(prepared=d, mpp=a.mpp)
            Y, panel, n_prot, n_out = slide_expression(d, sym2glob)
        else:
            src = open_slide(image=a.image, cells=a.cells, mpp=a.mpp)
            tmp = pathlib.Path(a.expression).parent
            Y, panel, n_prot, n_out = slide_expression(str(tmp), sym2glob)
        if len(Y) != len(src):
            raise SystemExit(f"{name}: expression has {len(Y)} rows but the slide has {len(src)} cells. "
                             f"Row i of each must be the same cell.")
        print(f"  [{name}] {len(src):,} cells, {len(panel)} usable genes "
              f"({n_prot} protein channels and {n_out} genes outside the head dropped)", flush=True)

        tiles = tile(src.pos, a.tile)
        dl = DataLoader(TileDS(src, tiles), batch_size=1, num_workers=a.workers,
                        collate_fn=lambda b: b[0], pin_memory=True,
                        prefetch_factor=4 if a.workers else None)
        E = np.zeros((len(src), 1536), np.float32)
        done = 0
        with torch.no_grad():
            for imgs, w, _pos, cells in dl:
                x = imgs.to(dev, non_blocking=True).float().div_(255.0)
                x = (x - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = pooled_feat(model, x, w.to(dev, non_blocking=True))
                E[cells.numpy()] = f.float().cpu().numpy()
                done += len(cells)
                if done % 50000 < len(cells):
                    print(f"      {done:>8,}/{len(src):,}", flush=True)
        p = save_bank_slide(a.bank_dir, name, E, Y, src.pos, panel, wid)
        print(f"  [{name}] -> {os.path.basename(p)} ({os.path.getsize(p)/1e9:.2f} GB, "
              f"{time.time()-t0:.0f}s)", flush=True)
        del E, Y

    names = list_bank(a.bank_dir)
    print(f"\n[bank] {a.bank_dir}: {len(names)} slides — {', '.join(names)}", flush=True)
    print("[note] retrieve with a bank that EXCLUDES the query slide; including it is self-retrieval.",
          flush=True)


if __name__ == "__main__":
    main()
