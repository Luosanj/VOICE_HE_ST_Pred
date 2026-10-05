"""Analyze reference fusion weights. Input: reference A/R/Y caches and target prediction caches. Output: support, variability, correlations, and fusion-variant TSVs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
GRID = np.linspace(0, 1, 41).astype(np.float32)
TISSUES = []
TARGETS = {}
EXCL_PREFIX = ()
DUPLICATES = ()
SAME_SPECIMEN = []

def ctr(X):
    return X - X.mean(0, keepdims=True)


def pcc_cols(P, Y):
    Pc, Yc = ctr(P), ctr(Y); num = (Pc * Yc).sum(0); den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    return np.where(den > 1e-8, num / np.where(den > 1e-8, den, 1.0), 0.0)


def grid_curve(A, R, Y):
    """[41, G] per-gene PCC of w*A+(1-w)*R for every grid w (inputs already centered)."""
    return np.stack([pcc_cols(w * A + (1 - w) * R, Y) for w in GRID])


def best_w(A, R, Y):
    """Choose the first grid point reaching the strict maximum PCC."""
    best = np.full(A.shape[1], -2.0); bw = np.ones(A.shape[1], np.float32)
    for w in GRID:
        r = pcc_cols(w * A + (1 - w) * R, Y); t = r > best; best = np.where(t, r, best); bw = np.where(t, w, bw)
    return bw.astype(np.float32)


def same_specimen(a, b):
    return any((a.startswith(x) and b.startswith(y)) or (a.startswith(y) and b.startswith(x)) for x, y in SAME_SPECIMEN)


def load_refs(refset):
    files = sorted(CACHE.glob("*.npz"))
    meta = pd.DataFrame([dict(f=f, tissue=f.name.split("__")[0], slide=str(np.load(f)["slide"]),
                              n=int(np.load(f)["n_cells"])) for f in files])
    meta = meta[~meta.slide.str.startswith(EXCL_PREFIX)]
    if refset != "paper":
        meta = meta[~meta.slide.str.startswith(DUPLICATES)]
    if refset in ("paper", "paper_dedup"):
        meta = meta.sort_values("n", ascending=False).groupby("tissue").head(NREF)
    refs = []
    for f in sorted(meta.f):
        z = np.load(f, allow_pickle=True)
        cov = z["covered"].astype(int); panel = z["panel"].astype(int)
        refs.append(dict(tissue=str(z["tissue"]), slide=str(z["slide"]), gid=panel[cov],
                         A=ctr(z["A"][:, cov]), R=ctr(np.nan_to_num(z["R"][:, cov])), Y=ctr(z["Y"][:, cov]),
                         n_cells=int(z["n_cells"])))
    return refs


def pooled_beta(refs):
    """Per gene: grid argmax of PCC on the cells of all refs measuring the gene, concatenated (each ref centered).
    Vectorised over genes that share the same set of measuring refs."""
    col = [{int(g): k for k, g in enumerate(r["gid"])} for r in refs]
    sig = {}
    for g in sorted(set(np.concatenate([r["gid"] for r in refs]))):
        sig.setdefault(tuple(i for i, c in enumerate(col) if int(g) in c), []).append(int(g))
    out = {}
    for members, genes in sig.items():
        cat = lambda key: np.concatenate([refs[i][key][:, [col[i][g] for g in genes]] for i in members])
        w = best_w(cat("A"), cat("R"), cat("Y"))
        out.update(zip(genes, w.tolist()))
    return out


def scalar_beta(refs):
    """One beta for every gene and tissue: grid argmax of the mean per-gene PCC over all refs."""
    curves = np.concatenate([grid_curve(r["A"], r["R"], r["Y"]) for r in refs], 1)
    return float(GRID[int(np.argmax(curves.mean(1)))])


def metrics_from_pcc(pcc, pg):
    s = pd.Series(pcc, index=pg.gene.values)
    out = {"ALL": float(np.nan_to_num(s.values).mean())}
    for k in (20, 50, 100):
        out[f"H{k}"] = float(np.nan_to_num(s.reindex(pg.sort_values("hvg_rank").gene.head(k)).values).mean())
        out[f"S{k}"] = float(np.nan_to_num(s.reindex(pg.sort_values("svg_rank").gene.head(k)).values).mean())
    return out


def stage_analyze(refset):
    refs = load_refs(refset)
    OUT = OUT_ROOT / f"refs_{refset}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] refset={refset}: {len(refs)} reference slides "
          f"{pd.Series([r['tissue'] for r in refs]).value_counts().to_dict()}", flush=True)

    rows = []
    for r in refs:
        w = best_w(r["A"], r["R"], r["Y"])
        r["beta"] = dict(zip(r["gid"].tolist(), w.tolist()))
        rows += [dict(tissue=r["tissue"], slide=r["slide"], gid=g, beta=b) for g, b in r["beta"].items()]
    sb = pd.DataFrame(rows); sb.to_csv(OUT / "slide_level_beta.tsv.gz", sep="\t", index=False)


    pgs = sb.groupby(["tissue", "gid"]).beta.agg(n_slides="count", mean="mean", sd="std", var="var",
                                                    min="min", max="max").reset_index()
    pgs["side_disagree"] = (pgs["min"] < 0.5) & (pgs["max"] > 0.5)
    pgs.to_csv(OUT / "pergene_beta_stability.tsv.gz", sep="\t", index=False)
    trows = []
    for t, d in pgs.groupby("tissue"):
        m = d[d.n_slides >= 2]
        trows.append(dict(tissue=t, n_ref_slides=sum(r["tissue"] == t for r in refs), n_genes_gated=len(d),
                          n_slides_distribution=d.n_slides.value_counts().sort_index().to_dict(),
                          n_genes_ge2_slides=len(m), median_sd=m.sd.median(), mean_sd=m.sd.mean(),
                          p90_sd=m.sd.quantile(0.9), frac_sd_gt_02=(m.sd > 0.2).mean(),
                          frac_side_disagree=m.side_disagree.mean(), tissue_mean_beta=d["mean"].mean()))
    pd.DataFrame(trows).to_csv(OUT / "tissue_summary.tsv", sep="\t", index=False)


    srows = []
    for r in refs:
        same = [o for o in refs if o["tissue"] == r["tissue"] and o["slide"] != r["slide"]]
        own = np.array([r["beta"][g] for g in r["gid"]], np.float32)
        loso = np.array([np.mean([o["beta"][g] for o in same if g in o["beta"]]) if any(g in o["beta"] for o in same)
                         else 1.0 for g in r["gid"]], np.float32)
        pA = pcc_cols(r["A"], r["Y"]); pR = pcc_cols(r["R"], r["Y"])
        pS_own = pcc_cols(own * r["A"] + (1 - own) * r["R"], r["Y"])
        pS_loso = pcc_cols(loso * r["A"] + (1 - loso) * r["R"], r["Y"])
        srows.append(dict(tissue=r["tissue"], slide=r["slide"], n_cells=r["n_cells"], n_genes_gated=len(r["gid"]),
                          mean_beta=own.mean(), median_beta=float(np.median(own)), frac_beta_lt_05=float((own < 0.5).mean()),
                          pcc_A=pA.mean(), pcc_R=pR.mean(), pcc_S_ownbeta=pS_own.mean(), pcc_S_losobeta=pS_loso.mean(),
                          gain_S_own_vs_A=(pS_own - pA).mean(), gain_S_loso_vs_A=(pS_loso - pA).mean()))
    pd.DataFrame(srows).to_csv(OUT / "slide_summary.tsv", sep="\t", index=False)


    prs = []
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            sh = sorted(set(a["beta"]) & set(b["beta"]))
            if len(sh) >= 20:
                x = pd.Series([a["beta"][g] for g in sh]); y = pd.Series([b["beta"][g] for g in sh])
                prs.append(dict(tissue_a=a["tissue"], tissue_b=b["tissue"], slide_a=a["slide"], slide_b=b["slide"],
                                same_tissue=a["tissue"] == b["tissue"], same_specimen=same_specimen(a["slide"], b["slide"]),
                                n_shared=len(sh), spearman=x.corr(y, method="spearman"), pearson=x.corr(y),
                                mean_abs_diff=float(np.abs(x - y).mean())))
    pd.DataFrame(prs).to_csv(OUT / "pairwise_beta_correlation.tsv", sep="\t", index=False)
    print("[analyze] stability tables written", flush=True)


    for r in refs:
        r["st"] = suff_stats(r["A"], r["R"], r["Y"])
    chk = [float((best_from_stats(r["st"]) == np.array([r["beta"][g] for g in r["gid"]], np.float32)).mean()) for r in refs]
    print(f"[analyze] slide-level beta: stats vs grid agreement min {min(chk):.4f}", flush=True)

    def pooled(rs):
        acc = {}
        for r in rs:
            for k, g in enumerate(r["gid"].tolist()):
                acc[g] = acc.get(g, 0) + r["st"][k]
        g = np.array(sorted(acc)); st = np.stack([acc[x] for x in g])
        return dict(zip(g.tolist(), best_from_stats(st).tolist()))

    V = {}
    V["mean_slide_beta"] = {t: sb[sb.tissue == t].groupby("gid").beta.mean().to_dict() for t in TISSUES}
    V["pooled_tissue_beta"] = {t: pooled([r for r in refs if r["tissue"] == t]) for t in TISSUES}
    gl = pooled(refs)
    V["global_pergene_pooled_beta"] = {t: gl for t in TISSUES}
    curves = np.concatenate([pcc_curve_from_stats(r["st"]) for r in refs], 1)
    sc = float(GRID[int(np.argmax(np.nanmean(curves, 1)))])
    V["global_scalar_beta"] = {t: {"__scalar__": sc} for t in TISSUES}
    V["mean_slide_beta + global fallback"] = {t: {**gl, **V["mean_slide_beta"][t]} for t in TISSUES}
    json.dump({"global_scalar_beta": sc}, open(OUT / "global_scalar_beta.json", "w"))
    bt = [dict(variant=vn, tissue=t, gid=g, beta=b) for vn, vb in V.items() if vn != "global_scalar_beta"
          for t, d in vb.items() for g, b in d.items()]
    pd.DataFrame(bt).to_csv(OUT / "beta_variants.tsv.gz", sep="\t", index=False)
    print(f"[analyze] variants built (global scalar beta = {sc})", flush=True)


    erows, check = [], []
    for stem, (tissue, tsv) in TARGETS.items():
        T = target_stats(stem, tsv)
        pg = T["pg"]; pA = np.nan_to_num(pg.pcc_A.values.astype(float))
        pos = {int(g): i for i, g in enumerate(pg.global_id.values)}
        ci = np.array([pos[int(g)] for g in T["gid"]])
        pR = pA.copy(); pR[ci] = np.nan_to_num(pcc_curve_from_stats(T["st"])[0])
        erows.append(dict(target=stem, tissue=tissue, variant="direct only", n_gated=0, **metrics_from_pcc(pA, pg)))
        erows.append(dict(target=stem, tissue=tissue, variant="retrieval where covered", n_gated=len(ci),
                          **metrics_from_pcc(pR, pg)))
        for vn, vb in V.items():
            d = vb[tissue]
            bet = np.array([d.get("__scalar__", d.get(int(g), np.nan)) for g in T["gid"]], float)
            ok = ~np.isnan(bet)
            pS = pA.copy()
            pS[ci[ok]] = np.nan_to_num(pcc_at(T["st"][ok], bet[ok]))
            erows.append(dict(target=stem, tissue=tissue, variant=vn, n_gated=int(ok.sum()), **metrics_from_pcc(pS, pg)))
            if vn == "mean_slide_beta":
                pb = pg.beta.values[ci]; both = ok & ~np.isnan(pb)
                check.append(dict(target=stem, n_gated_recomputed=int(ok.sum()), n_gated_paper=int((~np.isnan(pb)).sum()),
                                  beta_exact_match=float((np.abs(bet[both] - pb[both]) < 1e-4).mean()),
                                  beta_spearman=pd.Series(bet[both]).corr(pd.Series(pb[both]), method="spearman"),
                                  beta_mean_abs_diff=float(np.abs(bet[both] - pb[both]).mean()),
                                  pccA_covered_max_abs_diff=float(np.nanmax(np.abs(T["pA_cov"] - pg.pcc_A.values[ci]))),
                                  ALL_recomputed=float(pS.mean()), ALL_paper=float(np.nan_to_num(pg.pcc_STACK.values).mean())))
        print(f"  target {stem[:30]} done", flush=True)
    pd.DataFrame(erows).to_csv(OUT / "variant_eval_targets.tsv", sep="\t", index=False)
    pd.DataFrame(check).to_csv(OUT / "reproduction_check.tsv", sep="\t", index=False)
    print(pd.DataFrame(check).to_string())


def suff_stats(A, R, Y):
    """[G, 6] sums over cells of A*A, R*R, A*R, A*Y, R*Y, Y*Y (inputs already centered)."""
    A, R, Y = (x.astype(np.float64) for x in (A, R, Y))
    return np.stack([(A * A).sum(0), (R * R).sum(0), (A * R).sum(0), (A * Y).sum(0), (R * Y).sum(0), (Y * Y).sum(0)], 1)


def pcc_at(st, w):
    aa, rr, ar, ay, ry, yy = st.T
    num = w * ay + (1 - w) * ry; var = w * w * aa + 2 * w * (1 - w) * ar + (1 - w) ** 2 * rr
    den = np.sqrt(np.maximum(var, 0) * yy)
    return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1), 0.0)


def pcc_curve_from_stats(st):
    return np.stack([pcc_at(st, float(w)) for w in GRID])


def best_from_stats(st):
    return GRID[np.argmax(pcc_curve_from_stats(st), 0)].astype(np.float32)


def target_stats(stem, tsv):
    """Per-target sufficient statistics on bank-covered genes (cached, refset-independent)."""
    f = OUT_ROOT / "target_stats" / f"{stem}.npz"
    pg = pd.read_csv(Z / "pergene_pcc" / tsv, sep="\t")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        za = np.load(Z / f"Across_{stem}.npz", allow_pickle=True); zr = np.load(Z / f"Rcross_{stem}__full.npz", allow_pickle=True)
        gid_a = za["gid"].astype(int); cols_a = {int(g): j for j, g in enumerate(gid_a)}
        want = set(pg.global_id.astype(int))
        cov = [(int(zr["panel"][j]), int(j)) for j in zr["covered"] if int(zr["panel"][j]) in want and int(zr["panel"][j]) in cols_a]
        gid = np.array([g for g, _ in cov])
        A = za["Apred"]; ja = [cols_a[g] for g in gid]; A = ctr(np.asarray(A[:, ja], np.float32))
        Y = ctr(np.asarray(za["Ylog"][:, ja], np.float32)); del za
        R = ctr(np.asarray(zr["R"][:, [j for _, j in cov]], np.float32)); del zr
        st = suff_stats(A, R, Y); pA_cov = pcc_cols(A, Y)
        np.savez(f, gid=gid, st=st, pA_cov=pA_cov)
    z = np.load(f)
    return dict(gid=z["gid"], st=z["st"], pA_cov=z["pA_cov"], pg=pg)


def main():
    global CACHE, Z, OUT_ROOT, TISSUES, TARGETS, NREF, EXCL_PREFIX, DUPLICATES, SAME_SPECIMEN
    ap = argparse.ArgumentParser()
    ap.add_argument('--references', required=True, help='reference NPZ directory from cache_references.py')
    ap.add_argument('--predictions', required=True, help='Across_*, Rcross_*, and pergene_pcc/ directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--targets', required=True, help='JSON mapping target cache stem to [tissue, per-gene TSV name]')
    ap.add_argument('--refset', choices=['paper', 'paper_dedup', 'all_dedup'], default='paper')
    ap.add_argument('--nref', type=int, default=4)
    ap.add_argument('--exclude', nargs='*', default=[], help='reference slide prefixes to exclude')
    ap.add_argument('--duplicates', nargs='*', default=[], help='duplicate prefixes excluded in dedup modes')
    ap.add_argument('--specimen_pairs', help='JSON list of same-specimen reference slide-prefix pairs')
    a=ap.parse_args()
    CACHE, Z, OUT_ROOT, NREF = Path(a.references), Path(a.predictions), Path(a.out), a.nref
    TARGETS=json.loads(Path(a.targets).read_text())
    EXCL_PREFIX, DUPLICATES = tuple(a.exclude), tuple(a.duplicates)
    SAME_SPECIMEN = json.loads(Path(a.specimen_pairs).read_text()) if a.specimen_pairs else []
    TISSUES = sorted(set(v[0] for v in TARGETS.values()))
    stage_analyze(a.refset)


if __name__ == '__main__':
    main()
