#!/usr/bin/env python
"""Train contrastive image-expression alignment. Input: cell crops, masks, and scFoundation embeddings. Output: LoRA adapters, projection towers, and training checkpoints."""
from __future__ import annotations
import sys, os, json, time, argparse, math
from voice import paths as _p
_p.hf_home()
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from voice.util import load_config
from voice.cache_io import SourceCache
from voice.encoder import build_uni2, pooled_feat, MEAN, STD

dev = torch.device("cuda")
SCF = _p.scf_dir()
DATAROOT = _p.data_root()
CKDIR = _p.ckpt_dir()
EXCL = {
    "breast_cancer_sample1_xenium_replicate_1_Xenium_FFPE_Human_Breast_Cancer_Rep1",
    "breast_cancer_sample1_xenium_replicate_2_Xenium_FFPE_Human_Breast_Cancer_Rep2",
    "breast_cancer_sample2_xenium_Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2",
    "lung_cancer_sample1_xenium_Xenium_V1_humanLung_Cancer_FFPE",
    "skin_melanoma_sample1_xenium_Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE",
}
TEMP = 0.07
AMP_DTYPE = torch.bfloat16 if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.05):
        super().__init__(); self.base = base
        for p in base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5); self.s = alpha / r; self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.s


def inject_lora(m, nblocks, r, alpha, dropout):
    n = len(m.blocks)
    for i in range(n - nblocks, n):
        m.blocks[i].attn.qkv = LoRALinear(m.blocks[i].attn.qkv, r, alpha, dropout)
        m.blocks[i].attn.proj = LoRALinear(m.blocks[i].attn.proj, r, alpha, dropout)


class HeTower(nn.Module):
    def __init__(self, d=128): super().__init__(); self.ln = nn.LayerNorm(1536); self.net = nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, d))
    def forward(self, x): return F.normalize(self.net(self.ln(x)), dim=-1)


class ScfTower(nn.Module):
    def __init__(self, mu, sd, d=128):
        super().__init__(); self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(3072, 512), nn.GELU(), nn.Linear(512, d))
    def forward(self, x): return F.normalize(self.net((x - self.mu) / self.sd), dim=-1)


class CLIPDS(Dataset):
    def __init__(self, paths, W, scf): self.paths = paths; self.W = W; self.scf = scf
    def __len__(self): return len(self.paths)
    def __getitem__(self, k):
        img = np.asarray(Image.open(self.paths[k]).convert("RGB").resize((224, 224), Image.BILINEAR), np.uint8)
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(self.W[k]), torch.from_numpy(self.scf[k].astype(np.float32)))


def norm(x_uint8): x = x_uint8.float().to(dev, non_blocking=True) / 255.0; return (x - MEAN.to(dev)) / STD.to(dev)


def scf_stats(scf, chunk=200000):
    n, d = scf.shape; s = np.zeros(d, np.float64); ss = np.zeros(d, np.float64)
    for i in range(0, n, chunk):
        c = np.asarray(scf[i:i + chunk], np.float32); s += c.sum(0); ss += (c * c).sum(0)
    mu = s / n; var = np.maximum(ss / n - mu * mu, 0)
    return torch.from_numpy(mu.astype(np.float32)).to(dev), torch.from_numpy((np.sqrt(var) + 1e-6).astype(np.float32)).to(dev)


def infonce(zi, zg):
    logits = (zi @ zg.t()) / TEMP; lab = torch.arange(len(zi), device=zi.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))


def he_embed(model, he_tower, img, W):
    return he_tower(pooled_feat(model, norm(img), W.to(dev, non_blocking=True)))


def gradcache_step(chunks, model, he_tower, scf_tower, opt, clip_norm=1.0):
    """One GradCache InfoNCE optimizer step over `chunks` (list of (img,W,scf)). Returns loss (float)."""

    with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
        Zi = [he_embed(model, he_tower, img, W).float() for img, W, scf in chunks]
        Zg = [scf_tower(scf.to(dev, non_blocking=True)).float() for img, W, scf in chunks]
    zi = torch.cat(Zi).detach().requires_grad_(True); zg = torch.cat(Zg).detach().requires_grad_(True)

    loss = infonce(zi, zg); loss.backward(); gi, gg = zi.grad, zg.grad

    opt.zero_grad(set_to_none=True); off = 0
    for img, W, scf in chunks:
        c = img.shape[0]
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            zi_c = he_embed(model, he_tower, img, W); zg_c = scf_tower(scf.to(dev, non_blocking=True))
        torch.autograd.backward([zi_c.float(), zg_c.float()], [gi[off:off + c], gg[off:off + c]]); off += c
    params = [p for p in model.parameters() if p.requires_grad] + list(he_tower.parameters()) + list(scf_tower.parameters())
    torch.nn.utils.clip_grad_norm_(params, clip_norm); opt.step()
    return float(loss)


def trainable(model): return {n: p for n, p in model.named_parameters() if p.requires_grad}


def save_ckpt(path, step, epoch, model, he_tower, scf_tower, opt, args):
    obj = dict(step=step, epoch=epoch,
               lora={n: p.detach().cpu() for n, p in trainable(model).items()},
               he_tower=he_tower.state_dict(), scf_tower=scf_tower.state_dict(), opt=opt.state_dict(),
               torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(), args=vars(args))
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def load_into(model, he_tower, scf_tower, ck):
    msd = dict(model.named_parameters())
    for n, v in ck["lora"].items(): msd[n].data.copy_(v.to(dev))
    he_tower.load_state_dict(ck["he_tower"]); scf_tower.load_state_dict(ck["scf_tower"])


def save_artifact(path, model, he_tower, scf_tower, scf_mu, scf_sd, args, meta="", extra=None):
    """Save an alignment checkpoint with LoRA adapters and projection towers."""
    obj = dict(lora={n: p.detach().cpu() for n, p in trainable(model).items()}, he_tower=he_tower.state_dict(),
               scf_tower=scf_tower.state_dict(), scf_mu=scf_mu.cpu(), scf_sd=scf_sd.cpu(),
               nblocks=args.nblocks, r=args.r, alpha=args.alpha, dropout=args.dropout, args=vars(args), meta=meta)
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_slide_npz(sample_dir, per, seed):
    z = np.load(os.path.join(sample_dir, "patch_cell_boundaries.npz"), allow_pickle=True)
    er_all = z["expr_rows"].astype(np.int64); ip = z["indptr"]; vx = z["vertex_x_patch"]; vy = z["vertex_y_patch"]
    osize = int(z["output_size"]); tok = osize // 16; n = len(er_all)
    sel = np.arange(n) if n <= per else np.sort(np.random.RandomState(seed).choice(n, per, replace=False))
    man = pd.read_csv(os.path.join(sample_dir, "manifest.csv.gz"), usecols=["expr_row", "patch_path"]).drop_duplicates("expr_row").set_index("expr_row")["patch_path"]
    paths, Ws, ers = [], [], []
    for i in sel:
        er = int(er_all[i]); a, b = int(ip[i]), int(ip[i + 1]); w = np.zeros((16, 16), np.float32)
        if b - a >= 3:
            im = Image.new("L", (osize, osize), 0); ImageDraw.Draw(im).polygon(list(zip(vx[a:b].tolist(), vy[a:b].tolist())), fill=1)
            w = np.asarray(im, np.float32).reshape(16, tok, 16, tok).mean((1, 3))
        if w.sum() <= 1e-6: w[8, 8] = 1.0
        Ws.append((w / w.sum()).astype(np.float32)); paths.append(os.path.join(sample_dir, str(man.loc[er]))); ers.append(er)
    return paths, np.stack(Ws), np.array(ers, np.int64)


@torch.no_grad()
def verify_fidelity(cfg, n2t, n2d, slide, model):
    fz = np.load(os.path.join(cfg.paths.he_emb_dir, n2t[slide], slide, "features.npz"), allow_pickle=True)
    fc = fz["feats"].astype(np.float32); er2i = {int(e): i for i, e in enumerate(fz["expr_rows"].astype(np.int64))}
    paths, W, er = load_slide_npz(n2d[slide], 32, 123)
    imgs = torch.stack([torch.from_numpy(np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR), np.uint8).transpose(2, 0, 1))) for p in paths])
    x = (imgs.float().to(dev) / 255.0 - MEAN.to(dev)) / STD.to(dev)
    f = pooled_feat(model, x, torch.from_numpy(W).to(dev)).float().cpu().numpy()
    cos = np.array([float(f[k] @ fc[er2i[int(e)]] / (np.linalg.norm(f[k]) * np.linalg.norm(fc[er2i[int(e)]]) + 1e-9)) for k, e in enumerate(er) if int(e) in er2i])
    print(f"[fidelity] {slide[:30]}: live pooled_feat vs cached frozen feats cos mean={cos.mean():.4f} min={cos.min():.4f}", flush=True)
    return cos.mean() > 0.99


def build_data(cfg, name2t, name2dir, max_cells, seed, max_slides=0, cache=None):
    if cache and os.path.exists(cache + ".W.npy"):
        W = np.load(cache + ".W.npy"); scf = np.load(cache + ".scf.npy", mmap_mode="r"); paths = list(np.load(cache + ".paths.npy", allow_pickle=True))
        print(f"[data] loaded cache {cache} -> {len(paths):,} cells (scf mmap, W in RAM)", flush=True); return paths, W, scf
    slides = [s for _t, s, _p, _n in SourceCache(cfg).list_slides() if s not in EXCL and s in name2dir]
    if max_slides: slides = slides[:max_slides]
    per = max(1, max_cells // len(slides)); t0 = time.time(); nsk = 0
    cap = sum(min(int(len(np.load(os.path.join(name2dir[s], "patch_cell_boundaries.npz"), allow_pickle=True)["expr_rows"])), per) for s in slides)
    W = np.empty((cap, 16, 16), np.float32); scf = np.empty((cap, 3072), np.float16); paths = []; off = 0
    for s in slides:
        try:
            p, w, er = load_slide_npz(name2dir[s], per, seed)
            sc = np.load(os.path.join(SCF, s, "scf_cellemb.npy"), mmap_mode="r")[er].astype(np.float16)
        except Exception as e:
            print(f"[data] SKIP {s[:45]}: {type(e).__name__}: {e}", flush=True); nsk += 1; continue
        idx = np.where(np.abs(sc).sum(1) > 0)[0]; k = len(idx)
        W[off:off + k] = w[idx]; scf[off:off + k] = sc[idx]; paths += [p[i] for i in idx]; off += k
        print(f"  [{s[:36]:36s}] +{k:>7d} cells ({time.time()-t0:.0f}s)", flush=True)
    W = W[:off]; scf = scf[:off]
    print(f"[data] {len(slides)-nsk}/{len(slides)} slides -> {len(paths):,} cells (skipped {nsk}) in {time.time()-t0:.0f}s", flush=True)
    if cache:
        np.save(cache + ".W.npy", W); np.save(cache + ".scf.npy", scf); np.save(cache + ".paths.npy", np.array(paths, object))
        print(f"[data] cached -> {cache}.*", flush=True)
    return paths, W, scf


def verify_gradcache(model, he_tower, scf_tower, ds):
    """Compare GradCache and naive full-batch LoRA gradients."""
    from torch.utils.data import DataLoader
    b = [ds[i] for i in range(32)]; img = torch.stack([x[0] for x in b]); W = torch.stack([x[1] for x in b]); scf = torch.stack([x[2] for x in b])
    p = next(iter(trainable(model).values()))

    model.zero_grad(set_to_none=True); he_tower.zero_grad(); scf_tower.zero_grad()
    zi = he_embed(model, he_tower, img, W).float(); zg = scf_tower(scf.to(dev)).float()
    infonce(zi, zg).backward(); g_naive = p.grad.detach().clone()

    opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad] + list(he_tower.parameters()) + list(scf_tower.parameters()), lr=0.0)
    chunks = [(img[:16], W[:16], scf[:16]), (img[16:], W[16:], scf[16:])]
    gradcache_step(chunks, model, he_tower, scf_tower, opt)
    g_gc = p.grad.detach().clone()
    rel = (g_gc - g_naive).norm() / (g_naive.norm() + 1e-8)
    print(f"[verify] GradCache vs naive LoRA-grad rel-err = {rel:.2e}  ({'OK' if rel < 1e-2 else 'MISMATCH!'})", flush=True)
    return rel < 1e-2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config_real.yaml"); ap.add_argument("--tag", default="v1")
    ap.add_argument("--nblocks", type=int, default=12); ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=float, default=32); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max_cells", type=int, default=600000); ap.add_argument("--eff_batch", type=int, default=2048); ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr_lora", type=float, default=1.5e-4); ap.add_argument("--lr_tower", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--save_every", type=int, default=50); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--max_slides", type=int, default=0)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    args = ap.parse_args()
    global AMP_DTYPE
    if args.amp_dtype == "bf16":
        AMP_DTYPE = torch.bfloat16
    elif args.amp_dtype == "fp16":
        AMP_DTYPE = torch.float16
    else:
        AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"[amp] dtype={AMP_DTYPE}", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    if args.smoke: args.max_cells, args.eff_batch, args.chunk, args.epochs, args.save_every, args.workers, args.max_slides = 1500, 64, 32, 1, 20, 2, 2
    cfg = load_config(args.config); name2t = {s: t for t, s, _p, _n in SourceCache(cfg).list_slides()}
    name2dir = {p.parent.name: str(p.parent) for p in Path(DATAROOT).glob("*/*/manifest.csv.gz")}
    tr = [s for _t, s, _p, _n in SourceCache(cfg).list_slides() if s not in EXCL and s in name2dir]
    print(f"[data] {len(tr)} train slides with crops (all-except-5 that have manifests)", flush=True)
    cache = None if args.smoke else os.path.join(CKDIR, f"clip_lora_data_{args.max_cells}_{args.max_slides}")
    paths, W, scf = build_data(cfg, name2t, name2dir, args.max_cells, 1, args.max_slides, cache)
    scf_mu, scf_sd = scf_stats(scf)
    ds = CLIPDS(paths, W, scf); N = len(ds)

    model = build_uni2(dev); inject_lora(model, args.nblocks, args.r, args.alpha, args.dropout); model.to(dev)
    if not args.smoke:
        assert verify_fidelity(cfg, name2t, name2dir, tr[0], model), "FEATURE FIDELITY FAILED — live pooled_feat != cached frozen feats (wrong mask/crop?); aborting"
    if not args.no_ckpt: model.set_grad_checkpointing(True)
    he_tower = HeTower().to(dev); scf_tower = ScfTower(scf_mu, scf_sd).to(dev)
    nlora = sum(p.numel() for p in trainable(model).values())
    print(f"[model] LoRA params={nlora/1e3:.0f}K on last {args.nblocks} blocks (qkv+proj) r={args.r} a={args.alpha}", flush=True)

    lp = list(trainable(model).values()); tp = list(he_tower.parameters()) + list(scf_tower.parameters())
    opt = torch.optim.AdamW([{"params": lp, "lr": args.lr_lora}, {"params": tp, "lr": args.lr_tower}], weight_decay=1e-4)
    steps_per_epoch = max(1, N // args.eff_batch); total = steps_per_epoch * args.epochs

    if args.smoke:
        assert verify_gradcache(model, he_tower, scf_tower, ds), "GradCache mismatch — aborting"

    os.makedirs(CKDIR, exist_ok=True); latest = os.path.join(CKDIR, f"clip_lora_{args.tag}_latest.pt")
    dl = DataLoader(ds, batch_size=args.chunk, shuffle=True, num_workers=args.workers,
                    persistent_workers=(args.workers > 0), pin_memory=True, drop_last=True)
    n_chunks = max(1, args.eff_batch // args.chunk); step, epoch = 0, 0
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev, weights_only=False)
        load_into(model, he_tower, scf_tower, ck); opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"].cpu() if torch.is_tensor(ck["torch_rng"]) else ck["torch_rng"]); np.random.set_state(ck["np_rng"])
        step, epoch = ck["step"], ck["epoch"]
        print(f"[resume] from {latest}: step {step}/{total} epoch {epoch}", flush=True)

    print(f"[train] N={N:,} eff_batch={args.eff_batch} chunk={args.chunk} n_chunks/step={n_chunks} steps/epoch={steps_per_epoch} total={total}", flush=True)
    t0 = time.time(); run = 0.0; it = iter(dl); step0 = step
    saved_ep = step // steps_per_epoch
    while step < total:
        chunks = []
        for _ in range(n_chunks):
            try: chunks.append(next(it))
            except StopIteration: it = iter(dl); chunks.append(next(it))
        for g, lr in zip(opt.param_groups, (args.lr_lora, args.lr_tower)):
            g["lr"] = lr * min(1.0, (step + 1) / max(1, args.warmup))
        loss = gradcache_step(chunks, model, he_tower, scf_tower, opt); step += 1; run += loss
        ep = step // steps_per_epoch
        if step <= 6: print(f"  step {step} loss {loss:.4f} ({(time.time()-t0)/step:.1f}s/step, GPU {torch.cuda.max_memory_allocated()/1e9:.0f}GB)", flush=True)
        if step % 10 == 0:
            print(f"  step {step}/{total} ep{ep} loss {run/10:.4f} ({(time.time()-t0)/max(1,step-step0):.1f}s/step)", flush=True); run = 0.0
        if ep > saved_ep:
            epoch_index = ep - 1
            save_artifact(os.path.join(CKDIR, f"clip_lora_{args.tag}_epoch{epoch_index}.pt"),
                          model, he_tower, scf_tower, scf_mu, scf_sd, args,
                          f"LoRA-CLIP zero-indexed epoch {epoch_index}",
                          dict(epoch_index=epoch_index, completed_epochs=ep, step=step, steps_per_epoch=steps_per_epoch))
            print(f"  [epoch {epoch_index} done @ step {step}] saved clip_lora_{args.tag}_epoch{epoch_index}.pt", flush=True); saved_ep = ep
        if step % args.save_every == 0 or step == total:
            save_ckpt(latest, step, ep, model, he_tower, scf_tower, opt, args)

    final = os.path.join(CKDIR, f"clip_lora_{args.tag}_final.pt")
    save_artifact(final, model, he_tower, scf_tower, scf_mu, scf_sd, args, "LoRA-CLIP he<->scf all-except-5 final")
    print(f"[done] step {step} | saved {final}", flush=True)
    if args.smoke:
        ck = torch.load(latest, map_location=dev, weights_only=False); print(f"[smoke] resume-load OK: step={ck['step']} lora keys={len(ck['lora'])}", flush=True)


if __name__ == "__main__":
    main()
