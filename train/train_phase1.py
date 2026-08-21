#!/usr/bin/env python
"""Phase-1 LoRA-CLIP (spatial FM) — he <-> scFoundation InfoNCE with a LIVE LoRA-UNI2-h, WITH A VALIDATION SET.

Same model / data / GradCache / DDP / gates as the original Phase-1 script. The ONLY additions are validation:

  --val_frac F      per-slide SPATIAL band hold-out (NOT random cells), same rule as Phase-2's train_phase2.py:
                    cut at each slide's x-quantile so fraction F of ITS cells fall in the band. Every slide (hence
                    every tissue, including the 1-slide ones) contributes val cells.
  --val_margin PX   buffer between train and val cells. DEFAULT 256 >= the 224 px crop width, which is the point:
                    two cells closer than 224 px have PHYSICALLY OVERLAPPING crops, so a random cell split would put
                    near-duplicate images in train and val. The margin makes train and val crops pixel-disjoint.
  --val_every N     validate every N optimizer steps (plus at start, at each epoch boundary, and at the end).
  best checkpoint   lowest val InfoNCE -> clip_lora_<tag>_best.pt (same artifact format as epoch<N>/final).

Reported each validation: val InfoNCE (the training objective) and in-batch retrieval R@1 both directions
(he->scF, scF->he). R@1 is the interpretable number; InfoNCE is what selects the checkpoint.

⚠ InfoNCE depends on the number of in-batch negatives, so validation uses a FIXED --val_batch with drop_last, and a
FIXED seeded subset of val cells, or the numbers would not be comparable across steps.

--val_frac 0 reproduces clip_lora_v2.py exactly (no split, no val, no best.pt).

Run:   torchrun --nproc_per_node=2 train_phase1.py --tag v3 --epochs 3 --val_frac 0.1 --resume
Smoke: python train_phase1.py --smoke        (gates + split report only, no training)
"""
from __future__ import annotations
import os, sys, pathlib, time, json, argparse, math
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # repo root -> `voice` importable
from voice import paths as _p
_p.hf_home()                       # HF_HOME / *_OFFLINE, before timm or transformers is imported
from voice.clip_lora import (build_uni2, inject_lora, pooled_feat, norm, MEAN, STD, HeTower, ScfTower,
                       infonce, trainable, TEMP, AMP_DTYPE)                        # v1's validated numerics

V2 = _p.v2_root()
CROPS = f"{V2}/crops_raw"; SCF = f"{V2}/cell_emb_scf"; SM = f"{V2}/sample_meta"
V1_SCF = _p.scf_dir()
CKDIR = f"{V2}/ckpts"; S = 224


# ---------------------------------------------------------------- data
class MemmapCLIPDS(Dataset):
    """Flat index over cells. Zero decode, zero PIL: uint8 crop + fp16 mask + fp16 scF straight from disk.
    Memmaps are opened LAZILY per worker (a np.memmap does not survive fork cleanly)."""

    def __init__(self, samples, sid, row):
        self.samples = samples                       # list[(sample_id, crops_path, maskw_path, scf_path)]
        self.sid = sid; self.row = row               # int16[N], int32[N]
        self._c = self._w = self._s = None

    def _open(self):
        self._c = [np.load(s[1], mmap_mode="r") for s in self.samples]
        self._w = [np.load(s[2], mmap_mode="r") for s in self.samples]
        self._s = [np.load(s[3], mmap_mode="r") for s in self.samples]

    def __len__(self): return len(self.sid)

    def __getitem__(self, k):
        if self._c is None: self._open()
        i, r = int(self.sid[k]), int(self.row[k])
        img = np.asarray(self._c[i][r])                                   # (224,224,3) uint8, already decoded
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.asarray(self._w[i][r], np.float32)),
                torch.from_numpy(np.asarray(self._s[i][r], np.float32)))


def _cell_x(sid, sdir):
    """x_pixel of every cell of a sample, row-aligned with crops.u8 (same expr_row order the whole corpus uses).

    Cached to a tiny x_pixel.npy: patch_cell_boundaries.npz also holds the per-cell polygon vertices, so pulling
    x_pixel out of it costs ~11 s/slide (~15 min over the corpus) and that would be paid on EVERY launch/resume.
    With the cache the second launch is ~1 s. Cache misses are harmless (read-only dir -> just slow again)."""
    cache = f"{SM}/{sid}/x_pixel.npy"
    if os.path.exists(cache):
        try: return np.load(cache)
        except Exception: pass                                   # truncated/racing write -> fall through and redo
    for p in (f"{SM}/{sid}/patch_cell_boundaries.npz", os.path.join(str(sdir), "patch_cell_boundaries.npz")):
        if os.path.exists(p):
            x = np.load(p, allow_pickle=True)["x_pixel"].astype(np.float32)
            try:
                tmp = f"{cache}.{os.getpid()}.tmp.npy"; np.save(tmp, x); os.replace(tmp, cache)   # atomic, rank-safe
            except OSError:
                pass
            return x
    raise FileNotFoundError(f"{sid}: no patch_cell_boundaries.npz (needed for the spatial val split)")


def build_index(rank0=True, max_cells=0, val_frac=0.0, val_margin=256.0, val_side="hi"):
    """Flat (sample, row) index over training cells whose scF embedding is non-degenerate (v1 dropped these too).

    val_frac>0 additionally splits each sample by x: VAL if x >= cut, TRAIN if x < cut - margin, else DROPPED
    (cut = that sample's x-quantile(1-val_frac) over its good cells). Returns (train_ds, val_ds, samples)."""
    mf = pd.read_csv(f"{V2}/manifest_v2.csv"); tr = mf[mf.in_training]
    samples, sids, rows, vsids, vrows, skipped = [], [], [], [], [], []
    n_drop = 0; gaps = []
    for _, r in tr.iterrows():
        sid = r["sample"]
        cp = f"{CROPS}/{sid}/crops.u8"; wp = f"{CROPS}/{sid}/maskW.f16"
        sp = f"{SCF}/{sid}/scf_cellemb.npy"
        if not os.path.exists(sp): sp = f"{V1_SCF}/{sid}/scf_cellemb.npy"    # the 29 v1 samples reuse their embeddings
        if not (os.path.exists(cp) and os.path.exists(wp) and os.path.exists(sp)):
            skipped.append(sid); continue
        sc = np.load(sp, mmap_mode="r"); nc = np.load(cp, mmap_mode="r").shape[0]
        assert sc.shape[0] == nc, f"{sid}: scF {sc.shape[0]} != crops {nc} (row alignment broken)"
        good = np.where(np.abs(np.asarray(sc[:, :64], np.float32)).sum(1) > 0)[0]   # degenerate scF cells are all-zero
        i = len(samples); samples.append((sid, cp, wp, sp))
        if val_frac > 0:
            x = _cell_x(sid, r.get("sample_dir", ""))
            assert len(x) == nc, f"{sid}: boundaries {len(x)} != crops {nc} (row alignment broken)"
            xg = x[good]; cut = float(np.quantile(xg, 1.0 - val_frac))
            if val_side == "hi": vm = xg >= cut; tm = xg < cut - val_margin
            else:                vm = xg < cut;  tm = xg >= cut + val_margin
            n_drop += int((~vm & ~tm).sum())
            if vm.any() and tm.any():
                gaps.append(float(xg[vm].min() - xg[tm].max()) if val_side == "hi" else float(xg[tm].min() - xg[vm].max()))
            gtr, gva = good[tm], good[vm]
        else:
            gtr, gva = good, good[:0]
        sids.append(np.full(len(gtr), i, np.int16)); rows.append(gtr.astype(np.int32))
        vsids.append(np.full(len(gva), i, np.int16)); vrows.append(gva.astype(np.int32))
    sid = np.concatenate(sids); row = np.concatenate(rows)
    vsid = np.concatenate(vsids) if vsids else np.zeros(0, np.int16)
    vrow = np.concatenate(vrows) if vrows else np.zeros(0, np.int32)
    if max_cells and len(sid) > max_cells:                                  # deterministic subsample (scale ablations)
        k = np.sort(np.random.RandomState(0).choice(len(sid), max_cells, replace=False)); sid, row = sid[k], row[k]
    if rank0:
        print(f"[data] {len(samples)} samples ready, {len(skipped)} not yet pre-processed -> "
              f"{len(sid):,} train cells", flush=True)
        if val_frac > 0:
            ok = len(gaps) > 0 and min(gaps) > val_margin - 1e-3
            print(f"[val-split] band=x-quantile({1-val_frac:.2f}) side={val_side} margin={val_margin}px | "
                  f"train {len(sid):,} · val {len(vsid):,} ({len(vsid)/max(1,len(sid)+len(vsid))*100:.1f}%) · "
                  f"dropped {n_drop:,} cells in the margin", flush=True)
            print(f"[gate/val] {len(gaps)} samples with both sides | min(train-val x gap)="
                  f"{min(gaps) if gaps else float('nan'):.1f}px (margin={val_margin}, crop={S}) "
                  f"-> {'OK' if ok else 'FAIL'}", flush=True)
            assert ok, "VAL SPLIT LEAKS: some sample's train/val x gap <= margin"
        if skipped: print(f"[data] NOT READY: {skipped[:6]}{' ...' if len(skipped) > 6 else ''}", flush=True)
    return MemmapCLIPDS(samples, sid, row), MemmapCLIPDS(samples, vsid, vrow), samples


def scf_stats_cached(ds, path, rank0, world=1, chunk=100_000):
    """mu/sd over the scF bank — streamed once and cached (it is ~147 GB of reads). Computed over the dataset it is
    GIVEN, so passing the TRAIN dataset keeps val cells out of the input statistics.

    🔴 The first version fancy-indexed a WHOLE sample at once and cast to float64 (66 GB peak) and ran on BOTH ranks
    -> OOM killer took rank 0 (torchrun reports only `exitcode: -9`). Now rank 0 alone computes in chunk-row blocks
    (peak 100k x 3072 x 8 B = 2.4 GB), writes the cache, and other ranks poll the filesystem for it."""
    def _load():
        d = np.load(path)
        return (torch.from_numpy(d["mu"]).cuda(), torch.from_numpy(d["sd"]).cuda())

    if os.path.exists(path):
        return _load()
    if rank0:
        print(f"[scf] streaming mean/std over {len(ds):,} cells (chunked, rank0 only) -> {os.path.basename(path)} ...", flush=True)
        n = 0; s = np.zeros(3072, np.float64); ss = np.zeros(3072, np.float64)
        ds._open()
        for i in range(len(ds.samples)):
            r = ds.row[ds.sid == i]
            if not len(r): continue
            for k in range(0, len(r), chunk):
                A = np.asarray(ds._s[i][r[k:k + chunk]], np.float64)            # <= chunk x 3072 -> 2.4 GB
                s += A.sum(0); ss += np.einsum("ij,ij->j", A, A); n += len(A)   # einsum: no A*A temporary
                del A
        mu = s / n; sd = np.sqrt(np.maximum(ss / n - mu * mu, 0)) + 1e-6
        tmp = path + ".tmp.npz"
        np.savez(tmp, mu=mu.astype(np.float32), sd=sd.astype(np.float32)); os.replace(tmp, path)
        print(f"[scf] done over {n:,} cells -> {path}", flush=True)
    elif world > 1:
        # NOT dist.barrier(): NCCL's collective timeout is 600 s and this pass streams 147 GB. A filesystem poll has
        # no such deadline.
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > 4 * 3600:
                raise TimeoutError(f"rank>0 waited 4 h for {path}; rank 0 must have died")
            time.sleep(10)
        time.sleep(2)                                                           # let the os.replace land
    return _load()


# ---------------------------------------------------------------- gates
@torch.no_grad()
def gate_memmap_lossless(samples, model, dev, n=24):
    """The pre-decode claims to be LOSSLESS -> the memmap crop MUST be bit-identical to decoding the PNG, and the cached
    mask MUST equal the PIL rasterisation. Assert exact equality (stronger than cos~1), then compare live pooled_feat.

    ⚠️ Must NOT gate one of the 7 recovered slides: their stored PNGs are still ALL-BLACK (their crops were re-cut from
    the WSI), so "memmap == PNG" is guaranteed FALSE and would abort training on a bug that is not there."""
    pick = None
    for s in samples:
        mp = os.path.join(CROPS, s[0], "meta.json")
        try:
            if "WSI re-cut" not in str(json.load(open(mp)).get("source", "")): pick = s; break
        except Exception:
            continue
    if pick is None:
        print("[gate/memmap] no PNG-path sample to gate (all re-cut) — skipping", flush=True); return True
    sid, cp, wp, _sp = pick
    mfv = pd.read_csv(f"{V2}/manifest_v2.csv"); sdir = mfv[mfv["sample"] == sid].iloc[0]["sample_dir"]
    man = pd.read_csv(os.path.join(sdir, "manifest.csv.gz"),
                      usecols=["expr_row", "patch_path"]).drop_duplicates("expr_row").sort_values("expr_row")
    z = np.load(os.path.join(sdir, "patch_cell_boundaries.npz"), allow_pickle=True)
    ip = z["indptr"]; vx = z["vertex_x_patch"]; vy = z["vertex_y_patch"]
    C = np.load(cp, mmap_mode="r"); W = np.load(wp, mmap_mode="r")
    idx = np.random.RandomState(7).choice(len(man), n, replace=False)
    old_imgs, new_imgs, old_W, new_W = [], [], [], []
    for i in idx:
        p = os.path.join(sdir, str(man.patch_path.iloc[i]))
        img = np.asarray(Image.open(p).convert("RGB"), np.uint8)
        if img.shape[:2] != (S, S):
            img = np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BILINEAR), np.uint8)
        a, b = int(ip[i]), int(ip[i + 1])
        im = Image.new("L", (S, S), 0)
        if b - a >= 3: ImageDraw.Draw(im).polygon(list(zip(vx[a:b].tolist(), vy[a:b].tolist())), fill=1)
        w = np.asarray(im, np.float32).reshape(16, S // 16, 16, S // 16).mean((1, 3))
        if w.sum() <= 1e-6: w[8, 8] = 1.0
        w = w / w.sum()
        old_imgs.append(img); new_imgs.append(np.asarray(C[i])); old_W.append(w); new_W.append(np.asarray(W[i], np.float32))
    same_px = all(np.array_equal(a, b) for a, b in zip(old_imgs, new_imgs))
    dW = max(float(np.abs(a - b).max()) for a, b in zip(old_W, new_W))
    to = lambda L: torch.from_numpy(np.stack([x.transpose(2, 0, 1) for x in L])).contiguous()
    f_old = pooled_feat(model, norm(to(old_imgs)), torch.from_numpy(np.stack(old_W)).to(dev)).float().cpu().numpy()
    f_new = pooled_feat(model, norm(to(new_imgs)), torch.from_numpy(np.stack(new_W)).to(dev)).float().cpu().numpy()
    cos = float(np.mean(np.sum(f_old * f_new, 1) / (np.linalg.norm(f_old, axis=1) * np.linalg.norm(f_new, axis=1) + 1e-9)))
    print(f"[gate/memmap] {sid[:30]}: crops bit-identical={same_px}  |maskW diff|max={dW:.2e}  "
          f"pooled_feat cos(old,new)={cos:.6f}", flush=True)
    return same_px and dW < 1e-3 and cos > 0.999


def gate_gradcache(model, he_tower, scf_tower, ds, dev):
    """GradCache LoRA grads must ~= naive full-batch grads (v1's check, kept verbatim in spirit)."""
    b = [ds[i] for i in range(32)]
    img = torch.stack([x[0] for x in b]); W = torch.stack([x[1] for x in b]); scf = torch.stack([x[2] for x in b])
    p = next(iter(trainable(model).values()))
    model.zero_grad(set_to_none=True); he_tower.zero_grad(); scf_tower.zero_grad()
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        zi = he_tower(pooled_feat(model, norm(img), W.to(dev))).float(); zg = scf_tower(scf.to(dev)).float()
    infonce(zi, zg).backward(); g_naive = p.grad.detach().clone()
    opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad] +
                          list(he_tower.parameters()) + list(scf_tower.parameters()), lr=0.0)
    gradcache_step([(img[:16], W[:16], scf[:16]), (img[16:], W[16:], scf[16:])],
                   model, he_tower, scf_tower, opt, dev, world=1)
    rel = float((p.grad - g_naive).norm() / (g_naive.norm() + 1e-8))
    print(f"[gate/gradcache] rel-err vs naive = {rel:.2e}  ({'OK' if rel < 1e-2 else 'MISMATCH'})", flush=True)
    return rel < 1e-2


# ---------------------------------------------------------------- step (v1's GradCache + a single all-reduce)
def gradcache_step(chunks, model, he_tower, scf_tower, opt, dev, world=1, clip_norm=1.0):
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
        Zi = [he_tower(pooled_feat(model, norm(i), w.to(dev, non_blocking=True))).float() for i, w, _ in chunks]
        Zg = [scf_tower(s.to(dev, non_blocking=True)).float() for _, _, s in chunks]
    zi = torch.cat(Zi).detach().requires_grad_(True); zg = torch.cat(Zg).detach().requires_grad_(True)
    loss = infonce(zi, zg); loss.backward(); gi, gg = zi.grad, zg.grad          # negatives = this rank's eff_batch (== v1)
    opt.zero_grad(set_to_none=True); off = 0
    for img, W, scf in chunks:
        c = img.shape[0]
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            zi_c = he_tower(pooled_feat(model, norm(img), W.to(dev, non_blocking=True)))
            zg_c = scf_tower(scf.to(dev, non_blocking=True))
        torch.autograd.backward([zi_c.float(), zg_c.float()], [gi[off:off + c], gg[off:off + c]]); off += c
    params = [p for p in model.parameters() if p.requires_grad] + list(he_tower.parameters()) + list(scf_tower.parameters())
    if world > 1:                                                              # ONE all-reduce, after the last chunk
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
    torch.nn.utils.clip_grad_norm_(params, clip_norm); opt.step()
    return float(loss)


# ---------------------------------------------------------------- validation
@torch.no_grad()
def evaluate_clip(val_dl, model, he_tower, scf_tower, dev, world, chunk):
    """Val InfoNCE + in-batch retrieval R@1 (he->scF and scF->he). Towers already L2-normalize their output, so the
    similarity matrix is just zi @ zg.T — no re-normalisation (and no F.normalize p-vs-dim trap)."""
    was_train = model.training
    model.eval(); he_tower.eval(); scf_tower.eval()
    tot = torch.zeros(4, device=dev)                                            # [loss, r1_i2g, r1_g2i, n_batches]
    for img, W, scf in val_dl:
        Zi, Zg = [], []
        for i in range(0, len(img), chunk):                                     # chunked fwd: same memory knob as training
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                Zi.append(he_tower(pooled_feat(model, norm(img[i:i + chunk]), W[i:i + chunk].to(dev, non_blocking=True))).float())
                Zg.append(scf_tower(scf[i:i + chunk].to(dev, non_blocking=True)).float())
        zi = torch.cat(Zi); zg = torch.cat(Zg)
        sim = zi @ zg.t(); tgt = torch.arange(len(sim), device=dev)
        tot += torch.stack([infonce(zi, zg).float(),
                            (sim.argmax(1) == tgt).float().mean(),
                            (sim.argmax(0) == tgt).float().mean(),
                            torch.ones((), device=dev)])
    if world > 1: dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    n = tot[3].clamp(min=1)
    if was_train: model.train(); he_tower.train(); scf_tower.train()
    return float(tot[0] / n), float(tot[1] / n), float(tot[2] / n), int(tot[3].item())


# ---------------------------------------------------------------- ckpt
def save_ckpt(path, step, epoch, model, he_tower, scf_tower, opt, args, best_val=float("inf")):
    obj = dict(step=step, epoch=epoch, lora={n: p.detach().cpu() for n, p in trainable(model).items()},
               he_tower=he_tower.state_dict(), scf_tower=scf_tower.state_dict(), opt=opt.state_dict(),
               torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(), args=vars(args), best_val=best_val)
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)          # atomic
def save_artifact(path, model, he_tower, scf_tower, mu, sd, args, meta="", extra=None):
    obj = dict(lora={n: p.detach().cpu() for n, p in trainable(model).items()},
               he_tower=he_tower.state_dict(), scf_tower=scf_tower.state_dict(),
               scf_mu=mu.cpu(), scf_sd=sd.cpu(), nblocks=args.nblocks, r=args.r, alpha=args.alpha,
               dropout=args.dropout, args=vars(args), meta=meta)
    if extra: obj.update(extra)
    torch.save(obj, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--nblocks", type=int, default=12); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eff_batch", type=int, default=2048)                      # InfoNCE negatives PER RANK (== v1)
    ap.add_argument("--chunk", type=int, default=256)                           # GPU-memory knob; OOM auto-halves
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr_lora", type=float, default=1.5e-4); ap.add_argument("--lr_tower", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--max_cells", type=int, default=0)
    # ---- validation (the additions over clip_lora_v2.py) ----
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="per-slide spatial band held out for validation (0 = no val, == old clip_lora_v2 behaviour)")
    ap.add_argument("--val_margin", type=float, default=256.0,
                    help="px buffer between train and val cells; MUST be >= the 224px crop width or crops overlap")
    ap.add_argument("--val_side", default="hi", choices=["hi", "lo"], help="which end of x the val band sits on")
    ap.add_argument("--val_every", type=int, default=500, help="validate every N optimizer steps")
    ap.add_argument("--val_batch", type=int, default=0,
                    help="fixed InfoNCE batch for validation (0 = eff_batch). MUST stay fixed: the loss depends on it")
    ap.add_argument("--val_batches", type=int, default=8, help="how many fixed val batches to score each time")
    ap.add_argument("--val_workers", type=int, default=2)
    ap.add_argument("--no_val_at_start", action="store_true", help="skip the step-0 baseline validation")
    ap.add_argument("--scf_stats", default="", help="override the scF mu/sd cache path (default is split-aware)")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.val_batch <= 0: args.val_batch = args.eff_batch
    if args.val_frac > 0:
        assert args.val_margin >= S, f"--val_margin {args.val_margin} < crop {S}px: train/val crops would overlap"

    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0)); r0 = rank == 0
    if world > 1:
        torch.cuda.set_device(local)                                           # BEFORE init: pins this rank to its GPU
        dist.init_process_group("nccl", timeout=timedelta(hours=4), device_id=torch.device(f"cuda:{local}"))
    dev = torch.device(f"cuda:{local}")
    os.makedirs(CKDIR, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    ds, val_ds, samples = build_index(rank0=r0, max_cells=args.max_cells, val_frac=args.val_frac,
                                      val_margin=args.val_margin, val_side=args.val_side)
    if len(ds) == 0:
        if r0: print("[abort] no sample is fully pre-processed yet (crops + masks + scF)"); return

    model = build_uni2(dev); inject_lora(model, args.nblocks, args.r, args.alpha, args.dropout); model.to(dev)
    for p in model.parameters(): p.requires_grad = False
    for n, p in model.named_parameters():
        if ".A" in n or ".B" in n: p.requires_grad = True
    # scF stats over the TRAIN split only (val cells stay out of the input statistics). Split-aware cache path so a
    # val run never silently reuses the all-cell stats; --scf_stats <old path> opts back in and skips the 147 GB pass.
    stats_path = args.scf_stats or (f"{V2}/scf_stats.npz" if args.val_frac <= 0 else
                                    f"{V2}/scf_stats_train_vf{args.val_frac:g}_{args.val_side}.npz")
    mu, sd = scf_stats_cached(ds, stats_path, r0, world)
    he_tower = HeTower().to(dev); scf_tower = ScfTower(mu, sd).to(dev)

    if r0:                                                                     # GATES — abort on failure
        ok1 = gate_memmap_lossless(samples, model, dev)                        # LoRA B=0 here => frozen backbone
        ok2 = gate_gradcache(model, he_tower, scf_tower, ds, dev)
        if not (ok1 and ok2):
            print("[ABORT] a gate failed — refusing to train", flush=True)
            if world > 1: dist.destroy_process_group()
            return
    if args.smoke:
        if r0: print("[smoke] gates + split passed; not training.", flush=True)
        return
    if world > 1:                                                              # identical init on every rank
        for p in list(model.parameters()) + list(he_tower.parameters()) + list(scf_tower.parameters()):
            dist.broadcast(p.data, 0)

    # ---- fixed val batches (seeded subset, then sharded by rank so every rank scores whole batches) ----
    val_dl = None
    if args.val_frac > 0 and len(val_ds) >= args.val_batch:
        nb = min(args.val_batches, len(val_ds) // args.val_batch)
        pick = np.random.RandomState(1234).permutation(len(val_ds))[:nb * args.val_batch]
        pick = pick.reshape(nb, args.val_batch)[rank::world].reshape(-1)        # whole batches per rank
        if len(pick):
            val_dl = DataLoader(val_ds, batch_size=args.val_batch, sampler=pick.tolist(), drop_last=True,
                                num_workers=args.val_workers, pin_memory=True,
                                persistent_workers=args.val_workers > 0,
                                prefetch_factor=4 if args.val_workers else None)
        if r0: print(f"[val] {nb} fixed batches x {args.val_batch} negatives from {len(val_ds):,} val cells, "
                     f"every {args.val_every} steps", flush=True)

    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": list(he_tower.parameters()) + list(scf_tower.parameters()), "lr": args.lr_tower}], weight_decay=0.01)
    latest = f"{CKDIR}/clip_lora_{args.tag}_latest.pt"; best_path = f"{CKDIR}/clip_lora_{args.tag}_best.pt"
    step0, ep0, best_val = 0, 0, float("inf")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev, weights_only=False)
        msd = dict(model.named_parameters())
        for n, v in ck["lora"].items(): msd[n].data.copy_(v.to(dev))
        he_tower.load_state_dict(ck["he_tower"]); scf_tower.load_state_dict(ck["scf_tower"])
        opt.load_state_dict(ck["opt"]); step0, ep0 = ck["step"], ck["epoch"]
        best_val = float(ck.get("best_val", float("inf")))
        if r0: print(f"[resume] step {step0}, epoch {ep0}, best_val={best_val:.4f}", flush=True)

    per_rank = args.eff_batch
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    dl = DataLoader(ds, batch_size=per_rank, sampler=sampler, shuffle=(sampler is None), drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers else None)
    spe = len(dl); total = spe * args.epochs
    if r0:
        print(f"[train] {len(ds):,} cells | world={world} | eff_batch/rank={per_rank} chunk={args.chunk} "
              f"| {spe} steps/epoch x {args.epochs} = {total} steps", flush=True)

    step = step0; t0 = time.time(); ch = args.chunk

    def do_val(tag_txt):
        """Score val, print, keep the best-val checkpoint. Runs on every rank (all_reduce inside evaluate_clip)."""
        nonlocal best_val
        if val_dl is None: return
        tv = time.time(); vl, r_ig, r_gi, nb = evaluate_clip(val_dl, model, he_tower, scf_tower, dev, world, ch)
        improved = vl < best_val - 1e-6
        if improved: best_val = vl
        if r0:
            print(f"  [val] {tag_txt} step {step} InfoNCE {vl:.4f} (best {best_val:.4f}{' *NEW*' if improved else ''}) "
                  f"| R@1 he->scF {r_ig*100:.2f}% scF->he {r_gi*100:.2f}% | {nb} batches x {args.val_batch} "
                  f"in {time.time()-tv:.0f}s", flush=True)
            if improved:
                save_artifact(best_path, model, he_tower, scf_tower, mu, sd, args,
                              meta=f"v3 LoRA-CLIP BEST val={vl:.5f} @ step {step}",
                              extra=dict(step=step, val_infonce=vl, val_r1_he2scf=r_ig, val_r1_scf2he=r_gi))
        if world > 1: dist.barrier()

    # Resume lands mid-epoch. We must NOT re-iterate the epoch from batch 0, and must NOT "iterate and continue"
    # (the DataLoader still materialises every skipped batch: ~1.5 TB of reads here). Instead: take THIS rank's
    # deterministic index order for epoch ep0 and slice off the already-done prefix — zero skipped reads.
    skip = max(step0 - ep0 * spe, 0)
    if r0 and skip: print(f"[resume] fast-skip {skip} done batches of epoch {ep0} via sampler offset (0 data read)", flush=True)
    if not args.no_val_at_start: do_val("start")
    for ep in range(ep0, args.epochs):
        if sampler: sampler.set_epoch(ep)
        epoch_dl, iterskip = dl, 0
        if ep == ep0 and skip > 0:
            if sampler is not None:                                        # world>1: slice the sharded index order
                order = list(iter(sampler))[skip * per_rank:]
                epoch_dl = DataLoader(ds, batch_size=per_rank, sampler=order, drop_last=True,
                                      num_workers=args.workers, pin_memory=True, persistent_workers=False,
                                      prefetch_factor=4 if args.workers else None)
            else:
                iterskip = skip                                            # world==1 fallback (not our config)
        for bi, (img, W, scf) in enumerate(epoch_dl):
            if bi < iterskip: continue
            for g, base in zip(opt.param_groups, (args.lr_lora, args.lr_tower)):
                g["lr"] = base * min(1.0, (step + 1) / max(args.warmup, 1))
            while True:                                                        # OOM -> halve the chunk and retry
                try:
                    chunks = [(img[i:i + ch], W[i:i + ch], scf[i:i + ch]) for i in range(0, len(img), ch)]
                    loss = gradcache_step(chunks, model, he_tower, scf_tower, opt, dev, world)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if ch == 1: raise
                    ch = max(1, ch // 2)
                    if r0: print(f"[oom] chunk -> {ch}", flush=True)
            step += 1
            if r0 and step % 10 == 0:
                el = time.time() - t0
                print(f"  step {step}/{total} ep{ep} loss {loss:.4f} | {el/max(step-step0,1):.2f} s/step "
                      f"| ETA {(total-step)*el/max(step-step0,1)/3600:.1f} h", flush=True)
            if args.val_every > 0 and step % args.val_every == 0: do_val("periodic")
            if r0 and step % args.save_every == 0:
                save_ckpt(latest, step, ep, model, he_tower, scf_tower, opt, args, best_val)
        do_val(f"epoch{ep}-end")                                               # every rank (collective inside)
        if r0:                                                                 # ★ EVERY epoch, permanent, never overwritten
            save_artifact(f"{CKDIR}/clip_lora_{args.tag}_epoch{ep}.pt", model, he_tower, scf_tower, mu, sd, args,
                          meta=f"v3 LoRA-CLIP epoch {ep} ({len(ds):,} cells, world={world})",
                          extra=dict(epoch_index=ep, step=step))
            save_ckpt(latest, step, ep + 1, model, he_tower, scf_tower, opt, args, best_val)
            print(f"[epoch {ep} done @ step {step}] saved clip_lora_{args.tag}_epoch{ep}.pt", flush=True)
    do_val("final")
    if r0:
        save_artifact(f"{CKDIR}/clip_lora_{args.tag}_final.pt", model, he_tower, scf_tower, mu, sd, args,
                      meta="v3 final", extra=dict(step=step, best_val=best_val))
        print(f"[done] step {step} | saved clip_lora_{args.tag}_final.pt"
              f"{f' | BEST val {best_val:.4f} -> clip_lora_{args.tag}_best.pt' if val_dl is not None else ''}", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
