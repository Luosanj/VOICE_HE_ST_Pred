#!/usr/bin/env python
"""Fit reference-slide fusion weights. Input: same-tissue prepared references, model weights, and an excluding retrieval bank. Output: global-gene-ID to beta JSON."""
from __future__ import annotations
import os, sys, json, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import numpy as np
from voice import paths as _p
_p.hf_home()
import torch
from torch.utils.data import DataLoader

from voice.retrieval import crossR, list_bank
from voice.gate import fit_beta
from voice.panel import sym2glob_map
from predict.geometry import tile
from predict.inputs import open_slide
from predict.predict import TileDS, panel_global_ids
from predict.build_bank import weights_id, slide_expression


def main():
    ap = argparse.ArgumentParser(description="Fit the per-gene fusion gate on reference slides.")
    ap.add_argument("--bank", required=True, help="bank directory from predict/build_bank.py")
    ap.add_argument("--prepared", nargs="+", required=True,
                    help="reference slide directories, each with expression.npz + genes.tsv. Two or more.")
    ap.add_argument("--out", required=True, help="output gate JSON")
    ap.add_argument("--release", default=None)
    ap.add_argument("--stage1", default=None); ap.add_argument("--stage2", default=None)
    ap.add_argument("--genes", default=None, help="the head's gene table; taken from --release if absent")
    ap.add_argument("--global_genes", default=None, help="alias for --genes, for symmetry with build_bank.py")
    ap.add_argument("--knn", type=int, default=200); ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--tile", type=int, default=256); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if len(a.prepared) < 2:
        raise SystemExit("give at least two reference slides: each is retrieved from a bank that excludes it, "
                         "so a single reference leaves an empty bank.")
    genes_tsv = a.genes or a.global_genes
    in_bank = set(list_bank(a.bank))
    dev = torch.device(a.device)

    from voice.encoder import pooled_feat, MEAN, STD
    from voice.release import load_release
    model, se2, rcfg = load_release(a.release, device=dev, stage1=a.stage1, stage2=a.stage2)
    n_genes = int(rcfg["n_genes"])
    gid_head = panel_global_ids(a.release, genes_tsv, n_genes)
    wid = weights_id(a.release, a.stage1, a.stage2)
    sym2glob, _n = sym2glob_map(genes_tsv or os.path.join(a.release, "genes.tsv"))
    print(f"[model] {wid} | head {n_genes} genes | bank {a.bank}: {len(in_bank)} slides", flush=True)

    mean = MEAN.to(dev); std = STD.to(dev)
    refs = []
    for d in a.prepared:
        name = os.path.basename(os.path.abspath(d))
        others = sorted(in_bank - {name})
        if not others:
            raise SystemExit(f"{name}: the bank has no other slide to retrieve from.")
        if name not in in_bank:
            print(f"  [{name}] not in the bank — its own cells cannot leak in", flush=True)
        t0 = time.time()
        src = open_slide(prepared=d)
        Y_ref, panel_ref, _np_, _no = slide_expression(d, sym2glob)
        Y_ref = Y_ref[src.er]
        if len(Y_ref) != len(src):
            raise SystemExit(f"{name}: expression has {len(Y_ref)} rows but the slide has {len(src)} cells.")

        tiles = tile(src.pos, a.tile)
        dl = DataLoader(TileDS(src, tiles), batch_size=1, num_workers=a.workers,
                        collate_fn=lambda b: b[0], pin_memory=True,
                        prefetch_factor=4 if a.workers else None)
        A = np.zeros((len(src), n_genes), np.float32)
        E = np.zeros((len(src), 1536), np.float32)
        with torch.no_grad():
            for imgs, w, p, cells in dl:
                x = imgs.to(dev, non_blocking=True).float().div_(255.0)
                x = (x - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = pooled_feat(model, x, w.to(dev, non_blocking=True))
                    lm, _ = se2(f.float(), p.to(dev, non_blocking=True))
                idx = cells.numpy()
                A[idx] = lm.float().cpu().numpy(); E[idx] = f.float().cpu().numpy()
        print(f"  [{name}] A + embeddings done ({time.time()-t0:.0f}s); retrieving from "
              f"{len(others)} other slide(s)", flush=True)

        R, _cov = crossR(E, gid_head, a.bank, names=others, exclude={name}, K=a.knn, tau=a.tau,
                         weights_id=wid, device=a.device, verbose=False)

        col = {int(g): j for j, g in enumerate(gid_head)}
        keep = [(j, col[int(g)]) for j, g in enumerate(panel_ref) if int(g) in col]
        if not keep:
            raise SystemExit(f"{name}: none of its genes are in the model head.")
        yj = np.array([k[0] for k in keep]); hj = np.array([k[1] for k in keep])
        refs.append((A[:, hj], R[:, hj], Y_ref[:, yj], gid_head[hj]))
        print(f"  [{name}] {len(hj)} genes usable for fitting", flush=True)
        del A, E, R, Y_ref

    from voice.gate import fit_beta_reference
    gate = fit_beta_reference(refs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({str(k): float(v) for k, v in sorted(gate.items())}, f, indent=1)

    b = np.array(list(gate.values()), np.float32)
    print(f"\n[gate]  {len(gate)} genes -> {a.out}")
    print(f"        beta: mean {b.mean():.3f}  median {np.median(b):.3f}  "
          f"pure-A (beta=1) {int((b >= 0.999).sum())}  pure-R (beta=0) {int((b <= 0.001).sum())}")
    print("[note]  transfer this gate to a target slide with predict/predict.py --bank ... --gate "
          f"{os.path.basename(a.out)}; the target's own labels are never read.", flush=True)


if __name__ == "__main__":
    main()
