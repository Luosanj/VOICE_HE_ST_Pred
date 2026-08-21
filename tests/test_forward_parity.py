#!/usr/bin/env python
"""Parity test: the packaged `voice` forward path must reproduce the predictions the paper was scored on.

The repository vendors code that used to live across three directories. Vendoring can silently change behaviour
-- a different LoRA scale, the wrong token prefix, a mask that is normalised at the wrong point -- and the
symptom would be numbers that are plausible but not the published ones. This test rules that out by rerunning a
sample of real cells through the packaged path and comparing against the stored `Across_<slide>.npz` that the
paper's cross-slide table was computed from.

It needs the original preprocessed slide and the stored predictions, so it is a maintainer test, not something a
user can run after cloning. Point it at your own paths with the two environment variables below; it skips
cleanly when they are absent.

    VOICE_TEST_SLIDE_DIR   test_preprocessed/<group>/<slide>/   (patches/, manifest.csv.gz, patch_cell_boundaries.npz)
    VOICE_TEST_APRED       the matching Across_<slide>.npz

Run: python tests/test_forward_parity.py [--n 512]
"""
from __future__ import annotations
import os, sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from voice import paths as _p
_p.hf_home()
import torch
from PIL import Image
import pandas as pd

from voice.encoder import build_uni2, inject_lora, load_lora, pooled_feat, MEAN, STD
from voice.scale_train import ScaleHE2Cell
from predict.geometry import polygon_mask, tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="cells to test (sampled from whole tiles)")
    ap.add_argument("--slide_dir", default=os.environ.get("VOICE_TEST_SLIDE_DIR"))
    ap.add_argument("--apred", default=os.environ.get("VOICE_TEST_APRED"))
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--release", default=None,
                    help="a directory from tools/export_release.py; tests the SAFETENSORS weights instead")
    a = ap.parse_args()
    if not a.slide_dir or not a.apred:
        print("SKIP: set VOICE_TEST_SLIDE_DIR and VOICE_TEST_APRED (maintainer test)"); return 0
    # VOICE_CKPT_DIR is only consulted when neither --release nor explicit paths were given
    if a.release or (a.stage1 and a.stage2):
        s1, s2 = a.stage1, a.stage2
    else:
        ck_dir = _p.ckpt_dir()
        s1 = a.stage1 or os.path.join(ck_dir, "clip_lora_v2_final.pt")
        s2 = a.stage2 or os.path.join(ck_dir, "se2_lora_p2v2_noinslide_epoch0.pt")
    dev = torch.device("cuda")

    d = a.slide_dir
    bz = np.load(f"{d}/patch_cell_boundaries.npz", allow_pickle=True)
    er = bz["expr_rows"].astype(np.int64); ip = bz["indptr"]
    vx = bz["vertex_x_patch"]; vy = bz["vertex_y_patch"]
    osize = int(bz["output_size"])
    pos = np.stack([bz["y_pixel"], bz["x_pixel"]], 1).astype(np.float32)
    man = (pd.read_csv(f"{d}/manifest.csv.gz", usecols=["expr_row", "patch_path"])
             .drop_duplicates("expr_row").set_index("expr_row")["patch_path"])

    # whole tiles, so every sampled cell sees the same neighbourhood the reference run gave it
    tiles = tile(pos, 256)
    rs = np.random.RandomState(0); order = rs.permutation(len(tiles))
    sel, take = [], 0
    for t in order:
        sel.append(tiles[t]); take += len(tiles[t])
        if take >= a.n: break
    print(f"[test] {len(sel)} whole tiles, {take} cells from {os.path.basename(d)}", flush=True)

    if a.release:
        from voice.release import load_release
        model, se2, rcfg = load_release(a.release, device=dev)
        print(f"[test] weights from RELEASE {a.release} ({rcfg.get('name')})", flush=True)
    else:
        ck = torch.load(s2, map_location=dev, weights_only=False)
        model = build_uni2(dev); inject_lora(model, ck["nblocks"], ck["r"], ck["alpha"], ck["dropout"]); model.to(dev)
        load_lora(model, torch.load(s1, map_location=dev, weights_only=False)["lora"], ck["lora"]); model.eval()
        n_genes = int(ck["se2"]["head.mu_lin.weight"].shape[0])
        se2 = ScaleHE2Cell(n_genes, feat_dim=1536, d_model=ck["d_model"], n_layers=ck["n_layers"]).to(dev)
        se2.load_state_dict(ck["se2"]); se2.eval()

    got = {}
    mean = MEAN.to(dev); std = STD.to(dev)
    with torch.no_grad():
        for cells in sel:
            imgs = np.stack([np.ascontiguousarray(np.asarray(
                Image.open(os.path.join(d, str(man.loc[int(er[c])]))).convert("RGB").resize((224, 224), Image.BILINEAR),
                np.uint8).transpose(2, 0, 1)) for c in cells])
            W = np.stack([polygon_mask(vx[int(ip[c]):int(ip[c+1])], vy[int(ip[c]):int(ip[c+1])], osize) for c in cells])
            x = torch.from_numpy(imgs).to(dev).float().div_(255.0); x = (x - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = pooled_feat(model, x, torch.from_numpy(W).to(dev))
                lm, _ = se2(f.float(), torch.from_numpy(pos[cells]).to(dev))
            for j, c in enumerate(cells):
                got[int(c)] = lm[j].float().cpu().numpy()

    z = np.load(a.apred, allow_pickle=True)
    gid = z["gid"].astype(np.int64); cov = gid >= 0
    idx = np.array(sorted(got)); new = np.stack([got[i] for i in idx])
    ref = z["Apred"][idx]                                     # reference is already gid-indexed
    newg = np.zeros_like(ref); newg[:, cov] = new[:, gid[cov]]

    d_abs = np.abs(newg - ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    print(f"[cmp]  {len(idx)} cells x {ref.shape[1]} genes")
    print(f"       max |Δ|      = {d_abs.max():.3e}")
    print(f"       mean |Δ|     = {d_abs.mean():.3e}")
    print(f"       max rel |Δ|  = {(d_abs / denom).max():.3e}")
    r = np.corrcoef(newg.ravel(), ref.ravel())[0, 1]
    print(f"       correlation  = {r:.10f}")
    # bf16 autocast is not bit-deterministic across batch composition, so require agreement, not equality
    ok = (r > 0.9999) and (d_abs.mean() < 1e-3)
    print(f"\n{'PASS' if ok else 'FAIL'}: packaged forward {'reproduces' if ok else 'DOES NOT reproduce'} the reference predictions")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
