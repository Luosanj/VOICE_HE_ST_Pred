"""Stage 3, part one: the retrieval branch.

The direct branch predicts a cell's expression from its image alone. The retrieval branch answers a different
question — *which cells in a reference cohort look like this one, and what were they expressing?* — and
averages their measured profiles. The two make different mistakes, which is what makes fusing them worth doing
(`voice/gate.py`).

    R_i = sum_k softmax(cos(e_i, e_k) / tau)_k * y_k        over the K nearest bank cells

Three details are load-bearing.

**The bank is a set of reference slides with measured expression**, embedded with the *same* encoder as the
query. Mixing encoders puts query and bank in different spaces and the neighbours become meaningless — this is
not hypothetical, an earlier version of this project lost the effect that way. `build_bank` therefore records
which weights produced it, and `crossR` refuses a bank that does not match.

**Panels differ between slides, so retrieval is done per gene signature.** A query gene is predictable only
from bank slides that actually measure it. Genes are grouped by *which subset of the bank covers them*, and
each group retrieves against only that subset. A gene no bank slide measures gets NaN — never a silently
imputed zero.

**Bank expression is z-scored per slide before pooling.** Slides differ in depth and in panel composition, so
raw counts from two slides are not on one scale; the retrieval output therefore lives in z-space, not in
log1p-counts space. That is why the gate blends the two branches per gene rather than adding them.

Nearest neighbours are found with an exact torch top-K rather than faiss. `GpuIndexFlatIP` is brute force and
the project builds it with `useFloat16=True`, so fp16 scoring here returns the same neighbours; faiss is
avoided because initialising its cuBLAS handle after torch has touched the device aborts on some machines
(`blasStatus == CUBLAS_STATUS_SUCCESS` failed), even with the GPU otherwise idle.
"""
from __future__ import annotations
import os, json, glob
from collections import defaultdict

import numpy as np
import torch

K_DEFAULT = 200
TAU_DEFAULT = 0.03


# ----------------------------------------------------------------------------- exact top-K
@torch.no_grad()
def topk_cosine(Q, bank_E, K=K_DEFAULT, qchunk=1024, bchunk=1_000_000, device="cuda"):
    """Exact cosine top-K. Returns (D [Nq,K] float32, I [Nq,K] int64), descending.

    The bank is held on the GPU in fp16 blocks and scored block by block, keeping a running top-K, so a bank of
    several million cells never needs to fit in one allocation.

    ⚠️ `F.normalize`'s second POSITIONAL argument is `p`, not `dim`; both are named here.
    """
    dev = torch.device(device)
    Nq, M = len(Q), len(bank_E)
    Qt = torch.nn.functional.normalize(
        torch.from_numpy(np.ascontiguousarray(Q, np.float32)), p=2, dim=1).half()
    blocks = []
    for s in range(0, M, bchunk):
        be = torch.from_numpy(np.ascontiguousarray(bank_E[s:s + bchunk], np.float32))
        blocks.append((torch.nn.functional.normalize(be, p=2, dim=1).half().to(dev), s))
        del be
    D = np.empty((Nq, K), np.float32); I = np.empty((Nq, K), np.int64)
    for i in range(0, Nq, qchunk):
        qc = Qt[i:i + qchunk].to(dev)
        ds, is_ = [], []
        for blk, off in blocks:
            sc = qc @ blk.T                              # fp16, same precision as faiss useFloat16
            k = min(K, sc.shape[1])
            d, ii = torch.topk(sc, k, dim=1)
            ds.append(d); is_.append(ii + off); del sc
        dcat = torch.cat(ds, 1); icat = torch.cat(is_, 1)
        d, sel = torch.topk(dcat, K, dim=1)
        D[i:i + qchunk] = d.float().cpu().numpy()
        I[i:i + qchunk] = torch.gather(icat, 1, sel).cpu().numpy()
        del qc, ds, is_, dcat, icat, d, sel
    for blk, _ in blocks:
        del blk
    del blocks, Qt
    torch.cuda.empty_cache()
    return D, I


@torch.no_grad()
def knn_R(Q, bank_E, bank_Y, K=K_DEFAULT, tau=TAU_DEFAULT, chunk=1024, device="cuda"):
    """[Nq, G] softmax(cos/tau)-weighted mean of the K neighbours' expression."""
    D, I = topk_cosine(Q, bank_E, K, qchunk=chunk, device=device)
    dev = torch.device(device)
    bY = torch.from_numpy(np.ascontiguousarray(bank_Y, np.float32)).half().to(dev)
    P = np.empty((len(Q), bank_Y.shape[1]), np.float32)
    for i in range(0, len(Q), chunk):
        d = torch.from_numpy(D[i:i + chunk]).to(dev)
        w = torch.softmax(d / tau, dim=1)                # exp((d - max)/tau) normalised is exactly this
        P[i:i + chunk] = torch.einsum(
            "nk,nkg->ng", w, bY[torch.from_numpy(I[i:i + chunk]).to(dev)].float()).float().cpu().numpy()
        del d, w
    del bY
    torch.cuda.empty_cache()
    return P


# ----------------------------------------------------------------------------- the bank
BANK_SUFFIX = ".bank.npz"


def bank_path(bank_dir, name):
    return os.path.join(bank_dir, f"{name}{BANK_SUFFIX}")


def save_bank_slide(bank_dir, name, E, Y_log1p, pos, panel, weights_id):
    """One reference slide's contribution: embeddings, measured log1p expression, coordinates, gene ids."""
    os.makedirs(bank_dir, exist_ok=True)
    E = np.ascontiguousarray(E, np.float32)
    Y_log1p = np.ascontiguousarray(Y_log1p, np.float32)
    panel = np.asarray(panel, np.int64)
    if not (E.shape[0] == Y_log1p.shape[0] == len(pos)):
        raise ValueError(f"{name}: cell axis disagrees — E {E.shape[0]}, Y {Y_log1p.shape[0]}, pos {len(pos)}")
    if Y_log1p.shape[1] != len(panel):
        raise ValueError(f"{name}: Y has {Y_log1p.shape[1]} genes but panel lists {len(panel)}")
    p = bank_path(bank_dir, name)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, E1536=E, Y=Y_log1p, pos=np.asarray(pos, np.float32),
                        panel=panel, weights_id=np.array(str(weights_id)))
    os.replace(tmp, p)
    return p


def list_bank(bank_dir):
    return sorted(os.path.basename(p)[:-len(BANK_SUFFIX)]
                  for p in glob.glob(os.path.join(bank_dir, f"*{BANK_SUFFIX}")))


def _load(bank_dir, name, need=("E1536", "Y", "panel")):
    z = np.load(bank_path(bank_dir, name), allow_pickle=True)
    return tuple(z[k] for k in need) + (str(z["weights_id"]) if "weights_id" in z.files else "",)


def bank_panels(bank_dir, names):
    """{slide: {global_gene_id: column}} — read without touching the embeddings."""
    out = {}
    for s in names:
        z = np.load(bank_path(bank_dir, s), allow_pickle=True)
        out[s] = {int(g): j for j, g in enumerate(z["panel"].astype(np.int64))}
    return out


# ----------------------------------------------------------------------------- retrieval over a bank
def crossR(Eq, panel_q, bank_dir, names=None, K=K_DEFAULT, tau=TAU_DEFAULT,
           weights_id=None, device="cuda", verbose=True):
    """Retrieval prediction for one query slide.

    Eq       : [Nq, 1536] query embeddings, from the SAME encoder that built the bank
    panel_q  : [Gq] global gene ids of the query's panel
    Returns  : (R [Nq, Gq] float32, NaN where no bank slide measures that gene; covered gene indices)

    Genes are grouped by which bank slides cover them; each group's bank is loaded, used and freed, so peak
    memory is one group's bank rather than the whole cohort.
    """
    import gc
    names = names or list_bank(bank_dir)
    if not names:
        raise SystemExit(f"no bank slides in {bank_dir} — build one with predict/build_bank.py")
    panels = bank_panels(bank_dir, names)

    if weights_id is not None:
        for s in names:
            wid = _load(bank_dir, s, need=())[-1]
            if wid and wid != str(weights_id):
                raise SystemExit(
                    f"bank slide '{s}' was embedded with '{wid}' but the query uses '{weights_id}'.\n"
                    f"  Query and bank must share an encoder; different weights put them in different spaces "
                    f"and the neighbours become meaningless. Rebuild the bank with the query's weights.")

    panel_q = np.asarray(panel_q, np.int64)
    sig = {}
    for gi, g in enumerate(panel_q):
        hits = tuple(s for s in names if int(g) in panels[s])
        if hits:
            sig[gi] = hits
    covered = np.array(sorted(sig), dtype=np.int64)
    groups = defaultdict(list)
    for gi in covered:
        groups[sig[int(gi)]].append(int(gi))
    if verbose:
        print(f"[R] {len(names)} bank slides | {len(covered)}/{len(panel_q)} query genes covered "
              f"| {len(groups)} signature group(s)", flush=True)

    R = np.full((len(Eq), len(panel_q)), np.nan, np.float32)
    for hits, gis in groups.items():
        gg = panel_q[gis]
        Es, Ys = [], []
        for s in hits:
            E, Y, panel, _w = _load(bank_dir, s)
            cols = np.array([panels[s][int(g)] for g in gg], np.int64)
            Ysub = np.asarray(Y[:, cols], np.float32)
            # z-score per slide: depth and panel composition differ, so raw values are not on one scale
            Ys.append((Ysub - Ysub.mean(0, keepdims=True)) / (Ysub.std(0, keepdims=True) + 1e-6))
            Es.append(np.asarray(E, np.float32))
            del E, Y, Ysub
            gc.collect()
        bE = np.concatenate(Es); bY = np.concatenate(Ys)
        del Es, Ys
        gc.collect()
        if verbose:
            print(f"    [{len(gis):4d} genes] bank {len(bE):,} cells from {len(hits)} slide(s)", flush=True)
        R[:, gis] = knn_R(Eq, bE, bY, K, tau, device=device)
        del bE, bY
        gc.collect()
    return R, covered
