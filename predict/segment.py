#!/usr/bin/env python
"""Segment nuclei in an H&E slide. Input: whole-slide image and resolution. Output: cells.npz containing centroids and polygons."""
from __future__ import annotations
import os, sys, argparse, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from predict.slide_io import SlideReader

MPP_REF = 0.2125
MIN_AREA_PX = 15


def _require(mod, pip):
    try:
        return __import__(mod)
    except ImportError:
        raise SystemExit(f"{mod} is required by predict/segment.py: pip install {pip}")


def load_model(gpu: bool = True):
    _require("cellpose", "cellpose>=4.0")
    from cellpose import models
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")
    return models.CellposeModel(gpu=gpu, pretrained_model=path)


def segment_window(model, rgb, channel: int, niter: int, flow_threshold: float, cellprob_threshold: float):
    """rgb uint8 [h,w,3] -> int32 label image. Channel 3 (blue) carries the haematoxylin signal in H&E."""
    ch = rgb if rgb.ndim == 2 else rgb[:, :, channel - 1]
    masks, _flows, _ = model.eval(ch, niter=niter, flow_threshold=flow_threshold,
                                  cellprob_threshold=cellprob_threshold)
    return np.asarray(masks, np.int32)


def polygons_from_labels(lab, min_area: int):
    """label image -> [(cx, cy, area, contour[n,2]), ...] in window pixel coords."""
    cv2 = _require("cv2", "opencv-python-headless")
    out = []
    n = int(lab.max())
    if n == 0:
        return out

    from scipy import ndimage as ndi
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = (lab[sl] == i).astype(np.uint8)
        area = int(sub.sum())
        if area < min_area:
            continue
        y0, x0 = sl[0].start, sl[1].start
        ys, xs = np.nonzero(sub)
        cx, cy = float(xs.mean()) + x0, float(ys.mean()) + y0
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        c[:, 0] += x0; c[:, 1] += y0
        out.append((cx, cy, area, c))
    return out


def main():
    ap = argparse.ArgumentParser(description="Segment nuclei in an H&E slide -> cells.npz")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mpp", type=float, default=None,
                    help="microns per pixel of --image. Windows are rescaled so nuclei match the size Cellpose "
                         "expects. Read from the slide metadata when omitted.")
    ap.add_argument("--window", type=int, default=2048, help="window side in slide pixels")
    ap.add_argument("--overlap", type=int, default=128,
                    help="context margin around each window, in slide pixels. Must comfortably exceed one "
                         "nucleus diameter, or nuclei on a seam are segmented without their surroundings.")
    ap.add_argument("--channel", type=int, default=3, help="1-based channel; 3 = blue = haematoxylin")
    ap.add_argument("--niter", type=int, default=250)
    ap.add_argument("--flow_threshold", type=float, default=0.4)
    ap.add_argument("--cellprob_threshold", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows (smoke test)")
    a = ap.parse_args()

    rd = SlideReader(a.image)
    mpp = a.mpp if a.mpp is not None else rd.mpp
    scale = (mpp / MPP_REF) if mpp else 1.0
    print(f"[slide] {a.image} {rd.width}x{rd.height} via {rd.backend} | "
          f"mpp={mpp if mpp else 'unknown'} -> window scale {scale:.3f}", flush=True)
    if mpp is None:
        print("[warn]  no resolution found; assuming the reference resolution. Pass --mpp if that is wrong -- "
              "nuclei at the wrong apparent size are the usual cause of poor segmentation.", flush=True)
    if a.overlap < 64:
        print(f"[warn]  --overlap {a.overlap} px is small; nuclei on window seams may be cut.", flush=True)

    model = load_model(gpu=not a.cpu)
    W, ov = a.window, a.overlap
    xs = list(range(0, rd.width, W)); ys = list(range(0, rd.height, W))
    total = len(xs) * len(ys)
    print(f"[plan]  {len(xs)}x{len(ys)} = {total} windows of {W} px (+{ov} px context)", flush=True)

    cy_all, cx_all, area_all, polys = [], [], [], []
    t0 = time.time(); done = 0
    for y0 in ys:
        for x0 in xs:
            done += 1
            if a.limit and done > a.limit:
                break
            side = W + 2 * ov
            rgb = rd.region(x0 - ov, y0 - ov, side)
            if rgb.max() == 0:
                continue
            if abs(scale - 1.0) > 1e-3:
                from PIL import Image
                s = max(32, int(round(side * scale)))
                rgb_s = np.asarray(Image.fromarray(rgb).resize((s, s), Image.BILINEAR))
            else:
                rgb_s, s = rgb, side
            lab = segment_window(model, rgb_s, a.channel, a.niter, a.flow_threshold, a.cellprob_threshold)
            inv = side / float(s)
            for cx, cy, area, c in polygons_from_labels(lab, MIN_AREA_PX):
                gx = (x0 - ov) + cx * inv
                gy = (y0 - ov) + cy * inv

                if not (x0 <= gx < x0 + W and y0 <= gy < y0 + W):
                    continue
                if not (0 <= gx < rd.width and 0 <= gy < rd.height):
                    continue
                cx_all.append(gx); cy_all.append(gy); area_all.append(area * inv * inv)
                cc = c.copy(); cc[:, 0] = (x0 - ov) + cc[:, 0] * inv; cc[:, 1] = (y0 - ov) + cc[:, 1] * inv
                polys.append(cc)
            if done % 20 == 0 or done == total:
                print(f"    window {done}/{total}  {len(cx_all):,} nuclei  ({time.time()-t0:.0f}s)", flush=True)
        if a.limit and done > a.limit:
            break
    rd.close()

    n = len(cx_all)
    if n == 0:
        raise SystemExit("no nuclei found. Check --channel (3 = blue for H&E) and --mpp.")
    indptr = np.zeros(n + 1, np.int64)
    for i, c in enumerate(polys):
        indptr[i + 1] = indptr[i] + len(c)
    vx = np.concatenate([c[:, 0] for c in polys]).astype(np.float32)
    vy = np.concatenate([c[:, 1] for c in polys]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        y_pixel=np.asarray(cy_all, np.float32), x_pixel=np.asarray(cx_all, np.float32),
        indptr=indptr, vertex_x=vx, vertex_y=vy,
        cell_id=np.array([f"cell_{i}" for i in range(n)]),
        area_px=np.asarray(area_all, np.float32),
        mpp=np.float32(mpp if mpp else np.nan), image=os.path.basename(a.image))
    diam = 2.0 * np.sqrt(np.asarray(area_all) / np.pi)
    print(f"[out]   {a.out}  {n:,} nuclei, {len(vx):,} vertices ({time.time()-t0:.0f}s)", flush=True)
    print(f"        median diameter {np.median(diam):.1f} px"
          + (f" = {np.median(diam)*mpp:.1f} um" if mpp else "")
          + "  (nuclei are typically 6-12 um; far from that means --mpp or --channel is wrong)", flush=True)


if __name__ == "__main__":
    main()
