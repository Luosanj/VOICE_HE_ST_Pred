#!/usr/bin/env python
"""Train the spatial expression decoder with spatial validation. Input: prepared crops, masks, positions, gene panels, counts, and Stage-1 weights. Output: Stage-2 checkpoints."""
from __future__ import annotations
import sys, os, pathlib, time, math, argparse
from datetime import timedelta
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from voice import paths as _p
_p.hf_home()
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from scipy import sparse
from PIL import Image, ImageDraw
from voice.clip_lora import build_uni2, inject_lora, pooled_feat, trainable, MEAN, STD, EXCL
from voice.scale_train import ScaleHE2Cell, nb_nll

V2 = _p.v2_root(); SM = f"{V2}/sample_meta"; CROPS = f"{V2}/crops_raw"; CKDIR = _p.ckpt_dir()
LORA_CKPT = f"{V2}/ckpts/clip_lora_v2_final.pt"
GLOBAL_GENES = f"{V2}/global_genes_v2.tsv"; MANIFEST = f"{V2}/manifest_v2.csv"
AMP = torch.bfloat16
_HDR = {"gene", "gene_symbol", "symbol", "genes", "name"}


def read_genes(path):
    lines = [l.strip().split("\t")[0] for l in open(path) if l.strip()]
    return lines[1:] if lines and lines[0].lower() in _HDR else lines


def sym2glob_map():
    gg = pd.read_csv(GLOBAL_GENES, sep="\t")
    return dict(zip(gg["gene_symbol"].astype(str), gg["global_gene_index"].astype(int))), len(gg)


class Phase2V2DS(Dataset):
    def __init__(self, samples, sym2glob, patch_size=256, overlap=30, max_cells=256, min_cells=2, seed=0, rank0=True,
                 val_frac=0.0, val_margin=256, val_side="hi"):
        self.max_cells = max_cells; self.rng = np.random.RandomState(seed)
        self.cpaths = []; self.wpaths = []; self.pos = []; self.ycsr = []; self.panel = []; self.names = []; self.patches = []
        self.is_val = []
        self._C = self._W = None
        t0 = time.time(); tot = 0; dropped_panels = []; n_drop = 0; val_cells = 0; tr_cells = 0
        for s in samples:
            cp, wp = f"{CROPS}/{s}/crops.u8", f"{CROPS}/{s}/maskW.f16"
            if not (os.path.exists(cp) and os.path.exists(wp)):
                dropped_panels.append(s); continue
            bz = np.load(f"{SM}/{s}/patch_cell_boundaries.npz", allow_pickle=True)
            xp = bz["x_pixel"].astype(np.float32); yp = bz["y_pixel"].astype(np.float32); N = len(xp)
            nc = np.load(cp, mmap_mode="r").shape[0]; nw = np.load(wp, mmap_mode="r").shape[0]
            X = sparse.load_npz(f"{SM}/{s}/expression.npz").tocsr()
            assert nc == nw == N == X.shape[0], f"{s}: crops={nc} mask={nw} pos={N} expr={X.shape[0]} MISALIGNED"
            genes = read_genes(f"{SM}/{s}/genes.tsv")
            assert len(genes) == X.shape[1], f"{s}: genes.tsv={len(genes)} != expr cols={X.shape[1]}"
            cols = [i for i, g in enumerate(genes) if g in sym2glob]
            gid = np.array([sym2glob[genes[i]] for i in cols], np.int64)
            y = X[:, cols].tocsr().astype(np.float32)
            si = len(self.cpaths)
            self.cpaths.append(cp); self.wpaths.append(wp); self.pos.append(np.stack([yp, xp], 1))
            self.ycsr.append(y); self.panel.append(gid); self.names.append(s)
            cy, cx = yp, xp; H = cy.max() + patch_size; W = cx.max() + patch_size

            cut = float(np.quantile(cx, 1.0 - val_frac)) if val_frac > 0 else None
            for y0 in np.arange(0, H, patch_size - overlap):
                rm = (cy >= y0) & (cy < y0 + patch_size)
                if rm.sum() < min_cells: continue
                cyr = np.where(rm)[0]; cxr = cx[rm]
                for x0 in np.arange(0, W, patch_size - overlap):
                    cm = (cxr >= x0) & (cxr < x0 + patch_size)
                    if cm.sum() < min_cells: continue
                    cells = cyr[cm]
                    isval = False
                    if val_frac > 0:
                        xs = cx[cells]; lo, hi = float(xs.min()), float(xs.max())
                        in_band = (lo >= cut) if val_side == "hi" else (hi < cut)
                        past_margin = (hi < cut - val_margin) if val_side == "hi" else (lo >= cut + val_margin)
                        if in_band: isval = True
                        elif past_margin: isval = False
                        else: n_drop += 1; continue
                    if len(cells) > max_cells: cells = self.rng.choice(cells, max_cells, replace=False)
                    self.patches.append((si, np.sort(cells))); self.is_val.append(isval)
                    if isval: val_cells += len(cells)
                    else: tr_cells += len(cells)
            tot += N
        self.is_val = np.array(self.is_val, bool)
        if rank0:
            nv = int(self.is_val.sum()); nt = len(self.patches) - nv
            print(f"[data] {len(self.cpaths)} slides, {tot:,} cells, {len(self.patches):,} patches "
                  f"(max_cells={max_cells}) in {time.time()-t0:.0f}s", flush=True)
            if val_frac > 0:
                print(f"[val-split] band=x-quantile({1-val_frac:.2f}) side={val_side} margin={val_margin}px | "
                      f"train {nt:,} patches / {tr_cells:,} cells · val {nv:,} patches / {val_cells:,} cells "
                      f"({val_cells/max(1,tr_cells+val_cells)*100:.1f}%) · dropped {n_drop:,} straddling patches", flush=True)
            if dropped_panels: print(f"[data] NO crops yet (skipped): {dropped_panels}", flush=True)

    def _open(self):
        self._C = [np.load(p, mmap_mode="r") for p in self.cpaths]
        self._W = [np.load(p, mmap_mode="r") for p in self.wpaths]

    def __len__(self): return len(self.patches)

    def __getitem__(self, k):
        if self._C is None: self._open()
        si, cells = self.patches[k]
        img = np.ascontiguousarray(np.asarray(self._C[si][cells]).transpose(0, 3, 1, 2))
        W = np.asarray(self._W[si][cells], np.float32)
        pos = self.pos[si][cells]; y = np.asarray(self.ycsr[si][cells].todense(), np.float32)
        return (torch.from_numpy(img), torch.from_numpy(W), torch.from_numpy(pos),
                torch.from_numpy(y), torch.from_numpy(self.panel[si]))


def normalize(img_u8, dev):
    x = img_u8.to(dev, non_blocking=True).float().div_(255.0)
    return (x - MEAN.to(dev)) / STD.to(dev)


def freeze_first_lora(model, nblocks, frac):
    n = len(model.blocks); nfz = int(math.ceil(frac * nblocks)); fz = 0
    for i in range(n - nblocks, n - nblocks + nfz):
        for mod in (model.blocks[i].attn.qkv, model.blocks[i].attn.proj):
            for pn in ("A", "B"):
                if hasattr(mod, pn): getattr(mod, pn).requires_grad_(False); fz += 1
    return nfz, fz


@torch.no_grad()
def gate_mask_lossless(ds, model, dev, n=16):
    was_train = model.training; model.eval()
    si, cells = ds.patches[0]; s = ds.names[si]
    bz = np.load(f"{SM}/{s}/patch_cell_boundaries.npz", allow_pickle=True)
    ip = bz["indptr"]; vx = bz["vertex_x_patch"]; vy = bz["vertex_y_patch"]; osize = int(bz["output_size"]); tok = osize // 16
    C = np.load(ds.cpaths[si], mmap_mode="r"); Wc = np.load(ds.wpaths[si], mmap_mode="r")
    idx = cells[:n]; raster, cached, imgs = [], [], []
    for i in idx:
        a, b = int(ip[i]), int(ip[i + 1]); im = Image.new("L", (osize, osize), 0)
        if b - a >= 3: ImageDraw.Draw(im).polygon(list(zip(vx[a:b].tolist(), vy[a:b].tolist())), fill=1)
        w = np.asarray(im, np.float32).reshape(16, tok, 16, tok).mean((1, 3))
        if w.sum() <= 1e-6: w[8, 8] = 1.0
        raster.append((w / w.sum()).astype(np.float32)); cached.append(np.asarray(Wc[i], np.float32))
        imgs.append(np.asarray(C[i]).transpose(2, 0, 1))
    dW = max(float(np.abs(a - b).max()) for a, b in zip(raster, cached))
    x = normalize(torch.from_numpy(np.stack(imgs)), dev)
    with torch.autocast("cuda", dtype=AMP):
        f_c = pooled_feat(model, x, torch.from_numpy(np.stack(cached)).to(dev)).float().cpu().numpy()
        f_r = pooled_feat(model, x, torch.from_numpy(np.stack(raster)).to(dev)).float().cpu().numpy()
    cos = float(np.mean((f_c * f_r).sum(1) / (np.linalg.norm(f_c, axis=1) * np.linalg.norm(f_r, axis=1) + 1e-9)))
    ok = dW < 1e-3 and cos > 0.999
    print(f"[gate/mask] {s[:36]}: |maskW-raster|max={dW:.2e}  pooled_feat cos={cos:.6f}  -> {'OK' if ok else 'FAIL'}", flush=True)
    if was_train: model.train()
    return ok


@torch.no_grad()
def gate_val_disjoint(ds, val_margin, rank0=True):
    """Assert the split really is leak-free: for every slide, min x over VAL cells - max x over TRAIN cells > margin."""
    if not len(ds.patches) or not ds.is_val.any(): return True
    lo_v = {}; hi_t = {}
    for k, (si, cells) in enumerate(ds.patches):
        xs = ds.pos[si][cells][:, 1]
        if ds.is_val[k]: lo_v[si] = min(lo_v.get(si, np.inf), float(xs.min()))
        else: hi_t[si] = max(hi_t.get(si, -np.inf), float(xs.max()))
    gaps = [lo_v[si] - hi_t[si] for si in lo_v if si in hi_t]
    ok = len(gaps) > 0 and min(gaps) > val_margin - 1e-3
    if rank0:
        print(f"[gate/val] {len(gaps)} slides with both sides | min(train-val x gap)={min(gaps) if gaps else float('nan'):.1f}px "
              f"(margin={val_margin}) -> {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def _lora_cpu(model): return {n: p.detach().cpu() for n, p in trainable(model).items()}

def save_latest(path, step, epoch, model, se2, opt, args, best_val=float("inf")):
    obj = dict(step=step, epoch=epoch, lora=_lora_cpu(model), se2=se2.state_dict(), opt=opt.state_dict(),
               torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(), args=vars(args), best_val=best_val)
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def save_artifact(path, model, se2, args, meta, extra):
    obj = dict(lora=_lora_cpu(model), se2=se2.state_dict(), nblocks=args.nblocks, r=args.r, alpha=args.alpha,
               dropout=args.dropout, d_model=args.d_model, n_layers=args.n_layers, freeze_frac=args.freeze_frac,
               args=vars(args), meta=meta); obj.update(extra); torch.save(obj, path)


@torch.no_grad()
def evaluate(val_dl, model, se2, dev, world):
    """Mean per-patch val loss (same MSE[log1p] + 0.5*NB as training), summed across ranks. eval() disables LoRA dropout."""
    was_train = model.training
    model.eval(); se2.eval()
    tot = torch.zeros(2, device=dev)
    for img, W, pos, y, panel in val_dl:
        with torch.autocast("cuda", dtype=AMP):
            feats = pooled_feat(model, normalize(img, dev), W.to(dev, non_blocking=True))
            lm, aux = se2(feats.float(), pos.to(dev, non_blocking=True))
        p = panel.to(dev, non_blocking=True); yy = y.to(dev, non_blocking=True)
        loss = F.mse_loss(lm[:, p].float(), torch.log1p(yy)) + \
               0.5 * nb_nll(yy, aux["mu"][:, p].float(), aux["log_theta"][p].float().exp())
        tot[0] += loss.detach().float(); tot[1] += 1
    if world > 1: dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    if was_train: model.train(); se2.train()
    return float(tot[0] / tot[1].clamp(min=1)), int(tot[1].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2v3")
    ap.add_argument("--nblocks", type=int, default=12); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--freeze_frac", type=float, default=0.2)
    ap.add_argument("--d_model", type=int, default=512); ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--max_cells", type=int, default=256); ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=30); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr_lora", type=float, default=3e-5); ap.add_argument("--lr_se2", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05); ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--const_lr", type=float, default=0.0, help="if >0, flat lr = const_lr*base (no-spike extension)")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_ckpt", default=LORA_CKPT); ap.add_argument("--max_slides", type=int, default=0)
    ap.add_argument("--exclude_inslide", action="store_true", help="also hold out the 5 in-slide benchmark slices (clip_lora.EXCL)")
    ap.add_argument("--init_from", default="", help="warm-start TRAINED lora+se2 from a Phase-2 ckpt (fresh epoch counter/opt)")
    ap.add_argument("--max_steps", type=int, default=0)

    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="per-slide spatial band held out for validation (0 = no val, == old se2_lora_v2 behaviour)")
    ap.add_argument("--val_margin", type=float, default=256.0,
                    help="px buffer between train and val cells; patches straddling band-or-margin are dropped")
    ap.add_argument("--val_side", default="hi", choices=["hi", "lo"], help="which end of x the val band sits on")
    ap.add_argument("--val_every", type=int, default=2000, help="run validation every N optimizer steps")
    ap.add_argument("--val_max_patches", type=int, default=400,
                    help="fixed (seeded) subsample of val patches scored each time — keeps val cheap and comparable")
    ap.add_argument("--val_workers", type=int, default=3)
    ap.add_argument("--no_val_at_start", action="store_true", help="skip the step-0 baseline validation")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.save_every, args.workers, args.max_cells, args.max_slides, args.max_steps = 1, 4, 2, 32, 2, 8
        args.val_every, args.val_max_patches, args.val_workers = 4, 8, 0

    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0)); r0 = rank == 0
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(hours=4), device_id=torch.device(f"cuda:{local}"))
    else:
        torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    os.makedirs(CKDIR, exist_ok=True); torch.manual_seed(0); np.random.seed(0)

    sym2glob, n_global = sym2glob_map()
    mf = pd.read_csv(MANIFEST); samples = list(mf[mf.in_training]["sample"])
    if args.exclude_inslide:
        n0 = len(samples); samples = [s for s in samples if s not in EXCL]
        if r0: print(f"[exclude_inslide] held out {n0-len(samples)} in-slide benchmark slices -> {len(samples)} train slides", flush=True)
    if args.max_slides: samples = samples[:args.max_slides]
    if r0: print(f"[init] {len(samples)} train slides | global genes={n_global} | HF_HOME={os.environ['HF_HOME']}", flush=True)
    ds = Phase2V2DS(samples, sym2glob, args.patch_size, args.overlap, args.max_cells, rank0=r0,
                    val_frac=args.val_frac, val_margin=args.val_margin, val_side=args.val_side)


    tr_idx = np.where(~ds.is_val)[0]; va_idx = np.where(ds.is_val)[0]
    assert len(tr_idx) > 0, "no training patches left — val_frac/val_margin too aggressive"
    train_ds = Subset(ds, tr_idx.tolist())
    val_dl = None
    if args.val_frac > 0 and len(va_idx) > 0:
        if r0: assert gate_val_disjoint(ds, args.val_margin, rank0=True), "VAL SPLIT LEAKS (train/val x gap <= margin)"
        pick = np.random.RandomState(1234).permutation(va_idx)[:args.val_max_patches]
        pick = np.sort(pick)[rank::world]
        val_dl = DataLoader(Subset(ds, pick.tolist()), batch_size=1, shuffle=False, num_workers=args.val_workers,
                            pin_memory=True, persistent_workers=args.val_workers > 0,
                            prefetch_factor=4 if args.val_workers else None, collate_fn=lambda b: b[0])
        if r0: print(f"[val] scoring {min(len(va_idx), args.val_max_patches)} of {len(va_idx):,} val patches "
                     f"every {args.val_every} steps ({len(pick)} on rank0)", flush=True)


    model = build_uni2(dev); inject_lora(model, args.nblocks, args.r, args.alpha, args.dropout); model.to(dev)
    for p in model.parameters(): p.requires_grad = False
    for n, p in model.named_parameters():
        if ".A" in n or ".B" in n: p.requires_grad = True
    lck = torch.load(args.lora_ckpt, map_location=dev, weights_only=False)
    msd = dict(model.named_parameters())
    miss = [n for n in lck["lora"] if n not in msd]
    assert not miss, f"warm-start LoRA keys absent in model: {miss[:4]}"
    for n, v in lck["lora"].items(): msd[n].data.copy_(v.to(dev))
    nfz, _ = freeze_first_lora(model, args.nblocks, args.freeze_frac)
    model.set_grad_checkpointing(True)
    se2 = ScaleHE2Cell(n_global, feat_dim=1536, d_model=args.d_model, n_layers=args.n_layers).to(dev)
    lp = [p for p in model.parameters() if p.requires_grad]; sp = list(se2.parameters())
    if r0:
        print(f"[model] LoRA<-{os.path.basename(args.lora_ckpt)} | trainable LoRA={sum(p.numel() for p in lp)/1e3:.0f}K "
              f"(froze first {nfz}/{args.nblocks} blocks) | SE2 d{args.d_model}/L{args.n_layers}="
              f"{sum(p.numel() for p in sp)/1e6:.1f}M (fresh)", flush=True)

    if args.init_from:
        ick = torch.load(args.init_from, map_location=dev, weights_only=False)
        miss2 = [n for n in ick["lora"] if n not in msd]
        assert not miss2, f"init_from LoRA keys absent in model: {miss2[:4]}"
        for n, v in ick["lora"].items(): msd[n].data.copy_(v.to(dev))
        se2.load_state_dict(ick["se2"])
        if r0: print(f"[init_from] warm-start <- {os.path.basename(args.init_from)} "
                     f"(trained lora {len(ick['lora'])} overlay + se2 {len(ick['se2'])}); step->0, fresh opt", flush=True)

    if args.check_only:
        if r0:
            assert gate_mask_lossless(ds, model, dev), "MASK GATE FAILED"
            img, W, pos, y, panel = ds[int(tr_idx[0])]
            model.train(); se2.train()
            with torch.autocast("cuda", dtype=AMP):
                feats = pooled_feat(model, normalize(img, dev), W.to(dev)); lm, aux = se2(feats.float(), pos.to(dev))
            panel = panel.to(dev); y = y.to(dev)
            loss = F.mse_loss(lm[:, panel].float(), torch.log1p(y)) + 0.5 * nb_nll(y, aux["mu"][:, panel].float(), aux["log_theta"][panel].float().exp())
            loss.backward()
            gL = sum(float(p.grad.norm()) for p in lp if p.grad is not None); gS = sum(float(p.grad.norm()) for p in sp if p.grad is not None)
            print(f"[check_only] patch0 n_cells={img.shape[0]} genes={len(panel)} loss={float(loss.detach()):.4f} "
                  f"|gradLoRA|={gL:.3e} |gradSE2|={gS:.3e}  -> plumbing OK, safe to train.", flush=True)
        if world > 1: dist.destroy_process_group()
        return

    if world > 1:
        for p in list(model.parameters()) + list(se2.parameters()): dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW([{"params": lp, "lr": args.lr_lora}, {"params": sp, "lr": args.lr_se2}],
                            weight_decay=1e-4, fused=True)

    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, drop_last=True) if world > 1 else None
    dl = DataLoader(train_ds, batch_size=1, sampler=sampler, shuffle=(sampler is None), drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers else None, collate_fn=lambda b: b[0])
    spe = len(dl); total = spe * args.epochs
    if args.max_steps: total = min(total, args.max_steps)
    warmup = max(1, int(args.warmup_frac * total)); params = lp + sp
    latest = f"{CKDIR}/se2_lora_{args.tag}_latest.pt"; best_path = f"{CKDIR}/se2_lora_{args.tag}_best.pt"
    step, ep0, saved_ep, best_val = 0, 0, 0, float("inf")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev, weights_only=False)
        for n, v in ck["lora"].items(): msd[n].data.copy_(v.to(dev))
        se2.load_state_dict(ck["se2"]); opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"].cpu() if torch.is_tensor(ck["torch_rng"]) else ck["torch_rng"])
        np.random.set_state(ck["np_rng"]); step, ep0 = ck["step"], ck["epoch"]; saved_ep = step // max(1, spe)
        best_val = float(ck.get("best_val", float("inf")))
        if r0: print(f"[resume] step {step}/{total} epoch {ep0} best_val={best_val:.4f}", flush=True)
    if r0:
        print(f"[train] {spe} steps/epoch x {args.epochs} = {total} | world={world} | warmup={warmup} "
              f"| lr_lora={args.lr_lora} lr_se2={args.lr_se2}", flush=True)

    def lr_at(s):

        if args.const_lr > 0: return args.const_lr
        if s < warmup: return (s + 1) / warmup
        t = (s - warmup) / max(1, total - warmup); return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    def do_val(tag_txt):
        """Score val, print, and keep the best-val checkpoint. Runs on every rank (all_reduce inside evaluate)."""
        nonlocal best_val
        if val_dl is None: return
        tv = time.time(); vl, npatch = evaluate(val_dl, model, se2, dev, world)
        improved = vl < best_val - 1e-6
        if improved: best_val = vl
        if r0:
            print(f"  [val] {tag_txt} step {step} loss {vl:.4f} (best {best_val:.4f}{' *NEW*' if improved else ''}) "
                  f"on {npatch} patches in {time.time()-tv:.0f}s", flush=True)
            if improved:
                save_artifact(best_path, model, se2, args, f"Phase-2 SE2-LoRA BEST val={vl:.5f} @ step {step}",
                              dict(step=step, val_loss=vl, steps_per_epoch=spe, epoch_index=step // max(1, spe)))
        if world > 1: dist.barrier()

    def run_step(img, W, pos, y, panel):
        model.train(); se2.train()
        with torch.autocast("cuda", dtype=AMP):
            feats = pooled_feat(model, normalize(img, dev), W.to(dev, non_blocking=True))
            lm, aux = se2(feats.float(), pos.to(dev, non_blocking=True))
        p = panel.to(dev, non_blocking=True); yy = y.to(dev, non_blocking=True)
        loss = F.mse_loss(lm[:, p].float(), torch.log1p(yy)) + 0.5 * nb_nll(yy, aux["mu"][:, p].float(), aux["log_theta"][p].float().exp())
        loss.backward(); return float(loss.detach())

    if not args.no_val_at_start: do_val("start")
    t0 = time.time(); run = 0.0; step0 = step
    for ep in range(ep0, args.epochs):
        if sampler: sampler.set_epoch(ep)
        epoch_dl, skip = dl, max(step - ep * spe, 0) if ep == ep0 else 0
        if skip > 0 and sampler is not None:
            order = list(iter(sampler))[skip:]
            epoch_dl = DataLoader(train_ds, batch_size=1, sampler=order, drop_last=True, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=False, collate_fn=lambda b: b[0],
                                  prefetch_factor=4 if args.workers else None)
            if r0: print(f"[resume] fast-skip {skip} done batches of epoch {ep}", flush=True)
        for img, W, pos, y, panel in epoch_dl:
            for g, base in zip(opt.param_groups, (args.lr_lora, args.lr_se2)): g["lr"] = base * lr_at(step)
            opt.zero_grad(set_to_none=True)
            try:
                loss = run_step(img, W, pos, y, panel)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt.zero_grad(set_to_none=True)
                h = max(2, img.shape[0] // 2); sel = torch.randperm(img.shape[0])[:h]
                loss = run_step(img[sel], W[sel], pos[sel], y[sel], panel)
                if r0: print(f"[oom] step {step}: {img.shape[0]}->{h} cells", flush=True)
            if world > 1:
                for p in params:
                    if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
            nn.utils.clip_grad_norm_(params, args.grad_clip); opt.step()
            step += 1; run += loss
            if r0 and step <= 5:
                print(f"  step {step} loss {loss:.4f} ({(time.time()-t0)/max(1,step-step0):.1f}s/step, "
                      f"GPU {torch.cuda.max_memory_allocated()/1e9:.0f}GB, n_cells {img.shape[0]})", flush=True)
            if r0 and step % 50 == 0:
                print(f"  step {step}/{total} ep{step//max(1,spe)} loss {run/50:.4f} "
                      f"({(time.time()-t0)/max(1,step-step0):.2f}s/step)", flush=True); run = 0.0
            cur_ep = step // max(1, spe)
            at_epoch_end = cur_ep > saved_ep
            if args.val_every > 0 and (step % args.val_every == 0 or at_epoch_end or step >= total):
                do_val("epoch-end" if at_epoch_end else ("final" if step >= total else "periodic"))
            if r0 and at_epoch_end:
                ei = cur_ep - 1
                save_artifact(f"{CKDIR}/se2_lora_{args.tag}_epoch{ei}.pt", model, se2, args,
                              f"Phase-2 SE2-LoRA epoch {ei}", dict(epoch_index=ei, step=step, steps_per_epoch=spe))
                print(f"  [epoch {ei} done @ step {step}] saved se2_lora_{args.tag}_epoch{ei}.pt", flush=True)
            if at_epoch_end: saved_ep = cur_ep
            if r0 and (step % args.save_every == 0 or step == total):
                save_latest(latest, step, cur_ep, model, se2, opt, args, best_val)
            if step >= total: break
        if step >= total: break
    if r0:
        save_artifact(f"{CKDIR}/se2_lora_{args.tag}_final.pt", model, se2, args, "Phase-2 SE2-LoRA final",
                      dict(step=step, best_val=best_val))
        print(f"[done] step {step} | saved se2_lora_{args.tag}_final.pt"
              f"{f' | BEST val {best_val:.4f} -> se2_lora_{args.tag}_best.pt' if val_dl is not None else ''} "
              f"({(time.time()-t0)/3600:.1f}h)", flush=True)
        if args.smoke:
            ck = torch.load(latest, map_location=dev, weights_only=False)
            print(f"[smoke] resume-load OK: step={ck['step']} lora={len(ck['lora'])} se2_keys={len(ck['se2'])} "
                  f"best_val={ck.get('best_val')}", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
