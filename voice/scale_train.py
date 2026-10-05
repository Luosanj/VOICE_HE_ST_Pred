#!/usr/bin/env python
"""Decode cell features and train expression prediction. Input: features, coordinates, counts, and gene panels. Output: log1p predictions and decoder checkpoints."""
from __future__ import annotations
import argparse, os, sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata

from voice.util import load_config, set_seed
from voice.genes import GeneSpace
from voice.cache_io import SourceCache
from voice.scale_dataset import GlobalSpatialDataset


from voice._se2_arch import UNIEncoder, SE2Transformer
from voice._nb_head import NBHead


class ScaleHE2Cell(nn.Module):
    def __init__(self, n_genes, feat_dim=1536, d_model=256, d_pair=64, n_heads=8, n_layers=4):
        super().__init__()
        self.encoder = UNIEncoder(d_model, load_uni=False, feat_dim=feat_dim)
        self.se2 = SE2Transformer(d_model, d_pair, n_heads, n_layers)
        self.head = NBHead(d_model, n_genes)

    def forward(self, feats, pos):
        x = self.encoder(feats)
        h = self.se2(x, pos)
        return self.head(h)


def nb_nll(y, mu, theta):
    t = theta.clamp(1e-4, 1e4)
    return -(torch.lgamma(y + t) - torch.lgamma(t) - torch.lgamma(y + 1)
             + t * (torch.log(t) - torch.log(t + mu)) + y * (torch.log(mu + 1e-8) - torch.log(t + mu))).mean()


def per_gene_pcc(P, Y):
    out = np.full(P.shape[1], np.nan)
    for g in range(P.shape[1]):
        a, b = P[:, g], Y[:, g]
        if a.std() > 1e-8 and b.std() > 1e-8:
            out[g] = np.corrcoef(a, b)[0, 1]
    return out


def morans_i(coords, Y, k=6):
    from sklearn.neighbors import NearestNeighbors
    from scipy import sparse
    n = coords.shape[0]; k = min(k, n - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    W = sparse.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T); rs = np.asarray(W.sum(1)).ravel(); rs[rs == 0] = 1
    Wn = sparse.diags(1 / rs) @ W
    Z = Y - Y.mean(0, keepdims=True)
    return np.asarray((Z * (Wn @ Z)).sum(0)) / np.clip((Z * Z).sum(0), 1e-9, None)


@torch.no_grad()
def evaluate(model, val_ds, device):
    """Per held-out slide: aggregate per-cell preds over patches, per-gene log1p PCC; macro-avg."""
    model.eval()
    n_slides = len(val_ds.slides)
    acc = {si: {"sum": None, "cnt": None} for si in range(n_slides)}
    for i in range(len(val_ds)):
        b = val_ds[i]; si = b["slide"]
        lm, _ = model(b["features"].to(device), b["pos"].to(device))
        lm = lm[:, val_ds.slide_panel[si]].float().cpu().numpy()
        ci = b["cell_idx"].numpy()
        n = val_ds.slide_feats[si].shape[0]; Gp = lm.shape[1]
        if acc[si]["sum"] is None:
            acc[si]["sum"] = np.zeros((n, Gp), np.float32); acc[si]["cnt"] = np.zeros(n, np.float32)
        acc[si]["sum"][ci] += lm; acc[si]["cnt"][ci] += 1
    res = {"ALL": [], "HVG50": [], "SVG50": []}; per_slide = []
    for si in range(n_slides):
        cnt = acc[si]["cnt"]; m = cnt > 0
        pred = acc[si]["sum"][m] / cnt[m, None]
        y = np.asarray(val_ds.slide_ycsr[si][m].todense(), np.float32); y = np.log1p(y)
        coords = val_ds.slide_pos[si][m]
        pcc = per_gene_pcc(pred, y)
        var = y.var(0); mi = morans_i(coords, y)
        hvg = np.argsort(-var)[:50]; svg = np.argsort(-np.nan_to_num(mi))[:50]
        a, h, s = np.nanmean(pcc), np.nanmean(pcc[hvg]), np.nanmean(pcc[svg])
        res["ALL"].append(a); res["HVG50"].append(h); res["SVG50"].append(s)
        per_slide.append((val_ds.slides[si][1], float(a), float(h), float(s)))
    out = {k: float(np.nanmean(v)) for k, v in res.items()}; out["per_slide"] = per_slide
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--held_slides", required=True,
                    help="TEST slides. Never trained on, and (since the 2026-07-12 fix) NEVER selected on either.")
    ap.add_argument("--val_slides", default=None,
                    help="WHOLE slides held out of TRAINING, used ONLY for checkpoint selection. Must NOT overlap "
                         "--held_slides. If omitted, NO selection is performed (last-step ckpt only) — because "
                         "selecting on --held_slides is a TEST-SET LEAK (that bug produced scaleA_real_base.pt).")
    ap.add_argument("--monitor_held", action="store_true",
                    help="ALSO evaluate on --held_slides each eval and PRINT it. MONITOR/diagnostic ONLY — it can "
                         "never influence which checkpoint is kept.")
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--n_layers", type=int, default=4); ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--tag", default=""); ap.add_argument("--limit_train_slides", type=int, default=None)
    ap.add_argument("--train_slides", default=None, help="comma-sep: restrict TRAIN to these slides (scale ablation)")
    ap.add_argument("--init_ckpt", default=None, help="warm-start model weights from this ckpt (7M-pretrain -> finetune)")
    ap.add_argument("--he_emb_dir", default=None, help="override feature cache dir (e.g. LoRA-FM features)")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(0); dev = torch.device("cuda")
    if args.he_emb_dir: cfg.paths.he_emb_dir = args.he_emb_dir
    gs = GeneSpace(cfg.paths.global_genes, cfg.scfoundation.gene_index_tsv, cfg.paths.panels_dir, universe="all")
    src = SourceCache(cfg)
    all_slides = [(t, s) for (t, s, _p, _n) in src.list_slides()]

    known = {s for (_t, s) in all_slides}
    held = set(args.held_slides.split(","))
    val = set(args.val_slides.split(",")) if args.val_slides else set()
    if val & held:
        raise SystemExit(f"[LEAK] --val_slides overlaps --held_slides: {sorted(val & held)}. "
                         f"Selecting on the test slides is exactly the bug this fix removes.")
    if val - known:
        raise SystemExit(f"--val_slides not found in the cache: {sorted(val - known)}")
    train_slides = [(t, s) for (t, s) in all_slides if s not in held and s not in val]
    val_slides = [(t, s) for (t, s) in all_slides if s in val]
    test_slides = [(t, s) for (t, s) in all_slides if s in held]
    if args.train_slides:
        keep = set(args.train_slides.split(","))
        train_slides = [(t, s) for (t, s) in all_slides if s in keep and s not in held and s not in val]
    if args.limit_train_slides:
        train_slides = train_slides[: args.limit_train_slides]
    print(f"train slides={len(train_slides)} | val(selection) slides={len(val_slides)} | test(held) slides={len(test_slides)} | n_global={gs.n_global}")
    if val_slides:
        print(f"  [SELECT ON]  {', '.join(s[:44] for (_t, s) in val_slides)}")
    else:
        print("  ⚠️  NO --val_slides GIVEN -> NO CHECKPOINT SELECTION. Only the last-step ckpt (*_latest.pt / *_final.pt) "
              "is written. This is the safe default: selecting on --held_slides would be a TEST-SET LEAK.")
    print(f"  [NEVER SELECTED ON] test: {', '.join(s[:44] for (_t, s) in test_slides)}"
          + ("   (monitored & printed only)" if args.monitor_held else ""))
    train_ds = GlobalSpatialDataset(cfg, train_slides, gs, mode="train")
    val_ds = GlobalSpatialDataset(cfg, val_slides, gs, mode="val") if val_slides else None
    test_ds = GlobalSpatialDataset(cfg, test_slides, gs, mode="val") if (args.monitor_held and test_slides) else None

    model = ScaleHE2Cell(gs.n_global, feat_dim=1536, d_model=args.d_model, n_layers=args.n_layers).to(dev)
    isd = None; partial = False
    if args.init_ckpt:
        isd = torch.load(args.init_ckpt, map_location=dev)
        sd = isd["model"]; msd = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        partial = len(keep) < len(msd)
        model.load_state_dict(keep, strict=False)
        fresh = sorted(set(k.split('.')[0] for k in msd if k not in keep))
        print(f"warm-started from {args.init_ckpt} (step {isd.get('step')}, val {isd.get('val',{}).get('ALL')}): "
              f"loaded {len(keep)}/{len(msd)} tensors{'; FRESH (head) -> '+','.join(fresh) if partial else ''}")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params (trainable) = {n_tr/1e6:.1f}M | SE2 layers={args.n_layers}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if isd is not None and isd.get("opt") is not None and not partial:
        try: opt.load_state_dict(isd["opt"]); print("  resumed optimizer state")
        except Exception as e: print(f"  opt resume skipped: {e}")
    order = np.random.permutation(len(train_ds)); ptr = 0; t0 = time.time(); best = -1
    ckpt = os.path.join(cfg.paths.ckpt_dir, f"scaleA{args.tag}.pt"); os.makedirs(cfg.paths.ckpt_dir, exist_ok=True)
    for step in range(args.max_steps):
        if ptr >= len(order):
            order = np.random.permutation(len(train_ds)); ptr = 0
        b = train_ds[order[ptr]]; ptr += 1
        model.train()
        lm, aux = model(b["features"].to(dev), b["pos"].to(dev))
        panel = b["panel"].to(dev); y = b["y_counts"].to(dev)
        lmp = lm[:, panel]; mup = aux["mu"][:, panel]; thp = aux["log_theta"][panel].exp()
        loss = F.mse_loss(lmp, torch.log1p(y)) + 0.5 * nb_nll(y, mup, thp)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            print(f"step {step:6d} | loss {float(loss):.4f} | {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.max_steps:
            blob = {"model": model.state_dict(), "n_genes": gs.n_global, "args": vars(args),
                    "step": step + 1, "opt": opt.state_dict(), "val": None}
            if val_ds is not None:
                m = evaluate(model, val_ds, dev)
                ps = " | ".join(f"{n[:18]} {a:.4f}/{h:.4f}/{s:.4f}" for n, a, h, s in m.get("per_slide", []))
                print(f"  [VAL @ {step+1}] macro ALL={m['ALL']:.4f} HVG50={m['HVG50']:.4f} SVG50={m['SVG50']:.4f} || {ps}", flush=True)
                blob["val"] = m
            torch.save(blob, ckpt.replace(".pt", "_latest.pt"))
            if val_ds is not None and blob["val"]["ALL"] > best:
                best = blob["val"]["ALL"]; torch.save(blob, ckpt)
            if test_ds is not None:
                mt = evaluate(model, test_ds, dev)
                pt = " | ".join(f"{n[:18]} {a:.4f}" for n, a, _h, _s in mt.get("per_slide", []))
                print(f"  [TEST-MONITOR @ {step+1}] macro ALL={mt['ALL']:.4f} || {pt}"
                      f"   *** DIAGNOSTIC ONLY — NOT used to pick any checkpoint ***", flush=True)
    torch.save({"model": model.state_dict(), "n_genes": gs.n_global, "args": vars(args),
                "step": args.max_steps, "opt": opt.state_dict(), "val": None},
               ckpt.replace(".pt", "_final.pt"))
    if val_ds is not None:
        print(f"done {time.time()-t0:.0f}s | best VAL PCC_ALL={best:.4f} -> {ckpt} | last-step -> {ckpt.replace('.pt','_final.pt')}")
    else:
        print(f"done {time.time()-t0:.0f}s | NO SELECTION performed | last-step -> {ckpt.replace('.pt','_final.pt')}")


if __name__ == "__main__":
    main()
